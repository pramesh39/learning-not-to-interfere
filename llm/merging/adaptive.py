"""Adaptive merging: AdaMerging and AdaRank (AdaRank-TA variant).

Both learn how to merge by test-time entropy minimisation on unlabeled
test-partition prompts (no labels):

    AdaMerging: merged_t = theta0_t + sum_i w[i, t] * tau_{t,i}
    AdaRank   : as AdaMerging, but each attention/MLP task vector is replaced by
                U diag(gate * s) V^T, where gate is a learned per-singular-value mask
    loss      : mean next-token predictive entropy of the merged model over prompts

w has one entry per (task, parameter tensor) and is initialised to `prior`.
The base model and all five task vectors stay resident and differentiable, so
the model is spread across several GPUs (device_map="balanced").

    python -m llm.merging.adaptive --family llama-3.1-8b-instruct --experts ri \
        --method adamerging --output_dir merged/adamerging_ri
"""
import argparse
import gc
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.func import functional_call
from transformers import AutoModelForCausalLM

from llm.config import DOMAINS, MAX_SEQ_LENGTH, family_cfg, merging_cfg, models_cfg
from llm.merging.common import expert_ids, save_merged
from llm.ri.data import build_dataset, build_loader, cycle
from llm.ri.models import load_tokenizer


def token_entropy(logits, mask, chunk=64):
    """Mean next-token entropy over masked positions (chunked over time)."""
    tot = torch.zeros((), device=logits.device, dtype=torch.float32)
    n = 0.0
    for s in range(0, logits.shape[1], chunk):
        logp = F.log_softmax(logits[:, s:s + chunk].float(), dim=-1)
        H = -(logp.exp() * logp).sum(-1)
        m = mask[:, s:s + chunk].to(H.dtype)
        tot = tot + (H * m).sum()
        n += m.sum().item()
    return tot / max(n, 1.0)


def is_svd_key(name, tensor):
    """AdaRank decomposes attention and MLP weight matrices only."""
    return tensor.ndim == 2 and "norm" not in name and ("self_attn" in name or ".mlp." in name)


def main():
    ap = argparse.ArgumentParser(description="AdaMerging / AdaRank")
    ap.add_argument("--family", required=True, choices=sorted(models_cfg()["families"]))
    ap.add_argument("--experts", required=True, help="'vanilla', 'ri', or 5 comma-separated paths")
    ap.add_argument("--method", required=True, choices=["adamerging", "adarank"])
    ap.add_argument("--base_model", default=None, help="default: the family's base model (e.g. a local copy)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    a = ap.parse_args()

    cfg = merging_cfg()[a.method]
    fam = family_cfg(a.family)
    dtype = getattr(torch, fam["merge_dtype"])
    torch.manual_seed(a.seed)
    n_tasks = len(DOMAINS)
    adarank = a.method == "adarank"
    mask_temp, rank_ratio = 0.5, 1.0 / n_tasks          # AdaRank defaults (straight-through gate)

    base_path = a.base_model or fam["base"]
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype, device_map="balanced")
    model.eval()
    model.requires_grad_(False)
    if adarank:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base = {n: p.detach() for n, p in model.named_parameters()}
    names = list(base)
    svd_keys = {n for n in names if adarank and is_svd_key(n, base[n])}

    # task vectors: SVD factors for AdaRank's attn/MLP matrices, dense otherwise
    dense = {n: [] for n in names if n not in svd_keys}
    svd = {n: [] for n in svd_keys}
    for path in expert_ids(a.family, a.experts):
        print(f"[{a.method}] task vector from {path}", flush=True)
        exp = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
        ep = dict(exp.named_parameters())
        with torch.no_grad():
            for n in names:
                tau = ep[n].detach().to(base[n].device) - base[n]
                if n in svd:
                    U, s, Vt = torch.linalg.svd(tau.float(), full_matrices=False)
                    svd[n].append((U.to(dtype), s.float(), Vt.to(dtype)))
                else:
                    dense[n].append(tau)
        del exp, ep
        gc.collect()
        torch.cuda.empty_cache()

    dev0 = base[names[0]].device
    merge_weight = torch.nn.Parameter(torch.full((n_tasks, len(names)), cfg["prior"], device=dev0))
    merge_mask = {}
    for n in svd_keys:
        dim = svd[n][0][1].shape[0]
        m = torch.full((n_tasks, dim), 0.1, device=base[n].device)
        m[:, int(dim * rank_ratio):] = -0.1
        merge_mask[n] = torch.nn.Parameter(m)
    opt = torch.optim.Adam([merge_weight] + list(merge_mask.values()), lr=cfg["lr"])

    tok = load_tokenizer(base_path)
    iters = []
    for d in DOMAINS:
        ds = build_dataset(d, tok, fam["chat_template"], min(MAX_SEQ_LENGTH[d], cfg["max_seq_length"]), split="test")
        iters.append(cycle(build_loader(ds, tok, cfg["batch_size"])))

    def gated_tau(n, i):
        U, s, Vt = svd[n][i]
        gate = (merge_mask[n][i] / mask_temp).sigmoid()
        s_m = gate * s + (((gate > 0.5).float() - gate) * s).detach()   # straight-through
        return (U * s_m.to(U.dtype)) @ Vt

    def merged_state(final=False):
        sd = {}
        for t, n in enumerate(names):
            acc = base[n].detach().clone()
            if final:
                acc = acc.float()
            for i in range(n_tasks):
                tv = gated_tau(n, i) if n in svd_keys else dense[n][i]
                acc = acc + merge_weight[i, t].to(device=acc.device, dtype=acc.dtype) * tv.to(acc.dtype)
            sd[n] = acc
        return sd

    t0 = time.time()
    for step in range(1, cfg["steps"] + 1):
        opt.zero_grad(set_to_none=True)
        merged = merged_state()
        total, losses = 0.0, 0.0
        for di in range(n_tasks):
            b = next(iters[di])
            ids, msk = b["input_ids"].to(dev0), b["attention_mask"].to(dev0)
            out = functional_call(model, merged, args=(), kwargs={
                "input_ids": ids, "attention_mask": msk, "use_cache": False})
            loss = token_entropy(out.logits, msk) / n_tasks
            total += loss.item()
            if adarank:
                losses = losses + loss                                # one backward over all tasks
            else:
                loss.backward(retain_graph=di < n_tasks - 1)          # free each task's activations
        if adarank:
            losses.backward()
        opt.step()
        del merged
        if step % a.log_every == 0 or step == 1:
            print(f"[{a.method}] step {step}/{cfg['steps']} entropy={total:.4f} "
                  f"mean_w={merge_weight.detach().mean().item():.4f} "
                  f"({(time.time() - t0) / step:.1f}s/step)", flush=True)

    with torch.no_grad():
        final = {n: v.cpu() for n, v in merged_state(final=True).items()}
    state = {"domains": DOMAINS, "merge_weight": merge_weight.detach().cpu().tolist()}
    save_merged(final, a.output_dir, a.family,
                extra_files={f"{a.method}_state.json": json.dumps(state)})


if __name__ == "__main__":
    main()
