"""Loading experts and writing merged checkpoints."""
import os

import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from llm.config import DOMAINS, family_cfg


def expert_ids(family, experts):
    """experts: 'vanilla' (original MergeBench experts) or 'ri' (RI-adapted), or a
    comma-separated list of 5 local paths in DOMAINS order."""
    fam = family_cfg(family)
    if experts == "vanilla":
        return [fam["experts"][d] for d in DOMAINS]
    if experts == "ri":
        return [fam["ri_experts"][d] for d in DOMAINS]
    paths = [p.strip() for p in experts.split(",")]
    if len(paths) != len(DOMAINS):
        raise ValueError(f"expected {len(DOMAINS)} expert paths ({', '.join(DOMAINS)}), got {len(paths)}")
    return paths


def load_params(path, dtype):
    """name -> tensor (CPU) for every parameter. Tied weights appear once."""
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, low_cpu_mem_usage=True)
    params = {n: p.detach() for n, p in m.named_parameters()}
    del m
    return params


def load_base_and_experts(family, experts, dtype, base_model=None):
    base = load_params(base_model or family_cfg(family)["base"], dtype)
    expert_params = []
    for path in expert_ids(family, experts):
        print(f"[merge] loading {path}", flush=True)
        p = load_params(path, dtype)
        if p.keys() != base.keys():
            raise ValueError(f"parameter names differ between base and {path}")
        expert_params.append(p)
    return base, expert_params


def save_merged(params, out_dir, family, extra_files=None, base_model=None):
    """Write a loadable Hugging Face checkpoint in bfloat16.

    Weights are always stored in bfloat16, the dtype every evaluation loads
    them in. Merges computed in float16 (configs/models.yaml: merge_dtype) are
    cast here; casting at save time or at load time gives identical tensors.
    Config, tokenizer and generation config come from the base model that was
    merged onto (`base_model`, default: the family's base on the Hub).
    """
    base = base_model or family_cfg(family)["base"]
    os.makedirs(out_dir, exist_ok=True)
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in params.items()},
              os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    cfg = AutoConfig.from_pretrained(base)
    cfg.torch_dtype = torch.bfloat16
    cfg.save_pretrained(out_dir)
    AutoTokenizer.from_pretrained(base).save_pretrained(out_dir)
    try:
        GenerationConfig.from_pretrained(base).save_pretrained(out_dir)
    except OSError:
        pass
    for name, text in (extra_files or {}).items():
        with open(os.path.join(out_dir, name), "w") as f:
            f.write(text)
    print(f"[merge] saved {out_dir}", flush=True)


def coef_name(c):
    return f"{c:g}"
