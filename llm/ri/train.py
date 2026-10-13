"""Resolving Interference (RI): adapt one expert before merging.

    loss = L_own + alpha * mean_j L_other_j

    L_own     = KL(original expert || student) on the expert's OWN benchmark
    L_other_j = KL(base model     || student) on each of the 4 OTHER benchmarks

The student starts as a full copy of the expert. The original expert and the
shared base model are frozen teachers whose logits are computed on the fly at
every step. All data comes from the TEST partition (configs/splits.yaml);
ground-truth labels are never used.

By default each domain's text is (prompt + expert completion), so the student
is distilled across the expert's whole response. Completions are taken from
the local cache, then the published dataset, then generated
(see llm/ri/completions.py). --no_expert_completions trains on prompts only.

Example:
    python -m llm.ri.train --family llama-3.1-8b-instruct --domain math --output_dir runs/ri
"""
import argparse
import json
import os
import time

import torch

from llm.config import DOMAINS, MAX_SEQ_LENGTH, family_cfg, models_cfg
from llm.ri.completions import get_completions
from llm.ri.data import build_dataset, build_loader, cycle
from llm.ri.losses import masked_kl_divergence
from llm.ri.models import load_frozen_model, load_student, load_tokenizer, save_student


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", required=True, choices=sorted(models_cfg()["families"]))
    ap.add_argument("--domain", required=True, choices=DOMAINS, help="expert to adapt")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--expert_model", default=None, help="default: the family's MergeBench expert")
    ap.add_argument("--base_model", default=None, help="default: the family's base model")

    ap.add_argument("--expert_completions", action=argparse.BooleanOptionalAction, default=True,
                    help="distil on prompt + expert completion (default) or prompt only")
    ap.add_argument("--completions_dir", default="./expert_completions")
    ap.add_argument("--completions_source", default="auto", choices=["auto", "local", "hub", "generate"])

    ap.add_argument("--alpha", type=float, default=1.0, help="weight of the interference loss")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--micro_batch_size", type=int, default=None, help="default: from configs/models.yaml")
    ap.add_argument("--grad_accum_steps", type=int, default=None, help="default: from configs/models.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=0, help="0 = save only the final checkpoint")
    return ap.parse_args()


def main():
    a = parse_args()
    fam = family_cfg(a.family)
    expert = a.expert_model or fam["experts"][a.domain]
    base = a.base_model or fam["base"]
    mbs = a.micro_batch_size or fam["ri"]["micro_batch_size"]
    accum = a.grad_accum_steps or fam["ri"]["grad_accum_steps"]
    others = [d for d in DOMAINS if d != a.domain]
    template = fam["chat_template"]
    torch.manual_seed(a.seed)
    device = "cuda"
    os.makedirs(a.output_dir, exist_ok=True)

    print(f"[RI] {a.family}/{a.domain}: expert={expert} base={base}")
    print(f"[RI] effective batch {mbs} x {accum} = {mbs * accum}; steps={a.steps} lr={a.lr} alpha={a.alpha}; "
          f"expert_completions={a.expert_completions}")

    tokenizer = load_tokenizer(base)

    def loader(domain):
        comps = (get_completions(a.family, domain, a.completions_dir, a.completions_source)
                 if a.expert_completions else None)
        ds = build_dataset(domain, tokenizer, template, MAX_SEQ_LENGTH[domain], completions=comps, split="test")
        print(f"[RI] {domain}: {len(ds)} test-partition examples (max_len {MAX_SEQ_LENGTH[domain]})")
        return cycle(build_loader(ds, tokenizer, mbs))

    own_iter = loader(a.domain)
    other_iters = {d: loader(d) for d in others}

    student = load_student(expert, device=device)
    own_teacher = load_frozen_model(expert, device=device)
    base_teacher = load_frozen_model(base, device=device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    def logits(model, batch):
        return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits

    log = open(os.path.join(a.output_dir, f"{a.domain}_loss_log.jsonl"), "a")
    t0 = time.time()
    for step in range(1, a.steps + 1):
        optimizer.zero_grad()
        acc = {"L_own": 0.0, "L_other": 0.0, **{d: 0.0 for d in others}}
        for _ in range(accum):
            b = {k: v.to(device) for k, v in next(own_iter).items()}
            with torch.no_grad():
                t = logits(own_teacher, b)
            L_own = masked_kl_divergence(logits(student, b), t, b["attention_mask"])

            terms = []
            for d in others:
                ob = {k: v.to(device) for k, v in next(other_iters[d]).items()}
                with torch.no_grad():
                    t = logits(base_teacher, ob)
                terms.append(masked_kl_divergence(logits(student, ob), t, ob["attention_mask"]))
            L_other = torch.stack(terms).mean()

            ((L_own + a.alpha * L_other) / accum).backward()
            acc["L_own"] += L_own.item() / accum
            acc["L_other"] += L_other.item() / accum
            for d, x in zip(others, terms):
                acc[d] += x.item() / accum
        optimizer.step()

        log.write(json.dumps({"step": step, **acc, "elapsed_s": time.time() - t0}) + "\n")
        log.flush()
        if step % a.log_every == 0:
            print(f"[RI] step {step}/{a.steps} L_own={acc['L_own']:.4f} L_other={acc['L_other']:.4f} "
                  f"({(time.time() - t0) / step:.1f}s/step)", flush=True)
        if step == a.steps or (a.save_every and step % a.save_every == 0):
            path = os.path.join(a.output_dir, f"{a.family}_{a.domain}_RI_step{step}")
            save_student(student, tokenizer, path)
            print(f"[RI] saved {path}")
    log.close()


if __name__ == "__main__":
    main()
