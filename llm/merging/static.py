"""Coefficient-based merging: Task Arithmetic, TIES and TSV-M.

Each method reduces the experts' task vectors tau_i = theta_i - theta_0 to one
merged delta, and the merged model is theta_0 + lambda * delta. The delta does
not depend on lambda, so a sweep computes it once and writes every coefficient
from it.

    python -m llm.merging.static --family llama-3.1-8b-instruct --experts ri \
        --method ties --coefficients 0.5 --output_dir merged/ties_ri
    python -m llm.merging.static --family gemma-2-9b-it --experts vanilla \
        --method tsvm --sweep --output_dir merged/tsvm_vanilla

Output: <output_dir>/<lambda>/ (a Hugging Face checkpoint per coefficient).
"""
import argparse

import torch

from llm.config import family_cfg, grid, merging_cfg, models_cfg
from llm.merging.common import coef_name, load_base_and_experts, save_merged


# ------------------------------------------------------------- Task Arithmetic
def task_arithmetic_delta(base, experts):
    """sum_i tau_i, accumulated in the load dtype (configs/models.yaml: merge_dtype)."""
    delta = {}
    for n in base:
        acc = None
        for e in experts:
            tau = e[n] - base[n]
            acc = tau if acc is None else acc + tau
        delta[n] = acc
    return delta


def task_arithmetic_apply(base, delta, coef):
    return {n: base[n] + coef * delta[n] for n in base}


# ------------------------------------------------------------------------ TIES
def _topk_mask(flat, k_percent):
    k = int(flat.numel() * k_percent / 100.0)
    if k < 1:
        return torch.zeros_like(flat)
    thresh = flat.abs().kthvalue(flat.numel() - k).values
    return flat * (flat.abs() > thresh)


def ties_key(taus, k_percent, device):
    """Trim to top-k% by magnitude, elect a per-entry sign, average the agreeing entries."""
    shape = taus[0].shape
    stacked = torch.stack([_topk_mask(t.to(device).float().flatten(), k_percent) for t in taus])
    sign = torch.sign(stacked.sum(dim=0))
    sign[sign == 0] = torch.sign(sign.sum())
    keep = torch.where(sign.unsqueeze(0) > 0, stacked > 0, stacked < 0)
    selected = stacked * keep
    return (selected.sum(dim=0) / (selected != 0).sum(dim=0).float().clamp_min(1.0)).reshape(shape)


# ----------------------------------------------------------------------- TSV-M
def tsvm_key(taus, device):
    """Task Singular Vectors: per matrix, keep each task's top 1/N singular
    components, then decorrelate the stacked bases by orthogonalisation.
    Non-matrix parameters (norms) are averaged."""
    if taus[0].dim() != 2:
        return sum(t.to(device).float() for t in taus) / len(taus)
    n = len(taus)
    us, ss, vs = zip(*(torch.linalg.svd(t.to(device).float(), full_matrices=False) for t in taus))
    r = int(ss[0].shape[0] / n)
    sum_u, sum_s, sum_v = torch.zeros_like(us[0]), torch.zeros_like(ss[0]), torch.zeros_like(vs[0])
    for i in range(n):
        sum_u[:, i * r:(i + 1) * r] = us[i][:, :r]
        sum_s[i * r:(i + 1) * r] = ss[i][:r]
        sum_v[i * r:(i + 1) * r, :] = vs[i][:r, :]
    u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
    u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
    return torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v))


def per_key_delta(base, experts, fn):
    delta = {}
    for i, n in enumerate(base):
        taus = [e[n] - base[n] for e in experts]      # in bf16, as in the paper
        delta[n] = fn(taus).cpu()
        if (i + 1) % 50 == 0:
            print(f"[merge] {i + 1}/{len(base)} tensors", flush=True)
    return delta


def float32_apply(base, delta, coef):
    return {n: base[n].float() + coef * delta[n] for n in base}


# ------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="Task Arithmetic / TIES / TSV-M merging")
    ap.add_argument("--family", required=True, choices=sorted(models_cfg()["families"]))
    ap.add_argument("--experts", required=True,
                    help="'vanilla', 'ri', or 5 comma-separated paths (instruction,math,multilingual,coding,safety)")
    ap.add_argument("--method", required=True, choices=["task_arithmetic", "ties", "tsvm"])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--coefficients", type=float, nargs="+")
    g.add_argument("--sweep", action="store_true", help="the paper's grid (configs/merging.yaml)")
    ap.add_argument("--top_k", type=float, default=None, help="TIES only; default from configs/merging.yaml")
    ap.add_argument("--base_model", default=None, help="default: the family's base model (e.g. a local copy)")
    ap.add_argument("--output_dir", required=True)
    a = ap.parse_args()

    cfg = merging_cfg()[a.method]
    coefs = grid(cfg["coefficients"]) if a.sweep else a.coefficients
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if a.method == "task_arithmetic":
        dtype = getattr(torch, family_cfg(a.family)["merge_dtype"])
        base, experts = load_base_and_experts(a.family, a.experts, dtype, a.base_model)
        delta, apply = task_arithmetic_delta(base, experts), task_arithmetic_apply
    else:
        base, experts = load_base_and_experts(a.family, a.experts, torch.bfloat16, a.base_model)
        if a.method == "ties":
            k = a.top_k if a.top_k is not None else cfg["top_k"]
            delta = per_key_delta(base, experts, lambda t: ties_key(t, k, device))
        else:
            delta = per_key_delta(base, experts, lambda t: tsvm_key(t, device))
        apply = float32_apply
    del experts

    for c in coefs:
        save_merged(apply(base, delta, c), f"{a.output_dir}/{coef_name(c)}", a.family)


if __name__ == "__main__":
    main()
