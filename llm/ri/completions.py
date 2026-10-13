"""Expert completions: the distillation targets for RI's own-task loss.

For each domain, the original expert greedily completes every prompt of its
own benchmark. RI then distils the expert on (prompt + completion), so the
student matches the expert throughout its response, not just on the prompt.

get_completions() resolves them in this order:
  1. local cache    <cache_dir>/<family>/<domain>/<benchmark>_completions.json
  2. the published  dataset on the Hugging Face Hub (configs/models.yaml: completions_repo)
  3. generation     with the original expert

The published completions are the exact ones used in the paper. Regenerating
reproduces the procedure but is not guaranteed to match them byte for byte
(see README, "Expert completions").

CLI (pre-generate or pre-download outside of training):
    python -m llm.ri.completions --family llama-3.1-8b-instruct --domain math
    python -m llm.ri.completions --family gemma-2-9b-it --domain all --source generate
"""
import argparse
import json
import os
import time

import torch

from llm.config import BENCHMARK, DOMAINS, family_cfg, models_cfg
from llm.ri.data import full_prompts
from llm.ri.models import load_frozen_model, load_tokenizer

GEN_CONFIG = {"do_sample": False, "max_new_tokens": 512, "max_prompt_length": 512}


def _local_path(cache_dir, family, domain):
    return os.path.join(cache_dir, family, domain, f"{BENCHMARK[domain]}_completions.json")


def _validate(completions, domain, chat_template, path):
    n = len(full_prompts(domain, chat_template))
    if not isinstance(completions, list) or len(completions) != n:
        raise ValueError(f"{path}: expected a list of {n} completions for {domain}, "
                         f"got {type(completions).__name__} of length {len(completions)}")
    return completions


def _write(path, completions, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(completions, f)
    with open(path.replace(".json", ".meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def _download(family, domain, cache_dir):
    from huggingface_hub import hf_hub_download
    repo = models_cfg()["completions_repo"]
    rel = f"{family}/{domain}/{BENCHMARK[domain]}_completions.json"
    src = hf_hub_download(repo, rel, repo_type="dataset")
    with open(src) as f:
        completions = json.load(f)
    return completions, {"source": "hub", "repo": repo, "file": rel}


@torch.no_grad()
def generate(family, domain, batch_size=8, device="cuda", expert_model=None, chat_template=None):
    """Greedy completions from the original expert, batched with left padding.
    expert_model / chat_template override the family defaults (e.g. a local copy)."""
    fam = family_cfg(family)
    expert = expert_model or fam["experts"][domain]
    template = chat_template or fam["chat_template"]
    prompts = full_prompts(domain, template)
    tok = load_tokenizer(expert)
    tok.padding_side = "left"
    model = load_frozen_model(expert, device=device)
    out_texts = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True,
                  max_length=GEN_CONFIG["max_prompt_length"]).to(device)
        out = model.generate(**enc, do_sample=False, max_new_tokens=GEN_CONFIG["max_new_tokens"],
                             pad_token_id=tok.pad_token_id)
        out_texts += tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"[completions] {family}/{domain}: {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return out_texts, {"source": "generated", "expert": expert, "generation": GEN_CONFIG,
                       "chat_template": template, "created": time.strftime("%Y-%m-%d %H:%M:%S")}


def get_completions(family, domain, cache_dir, source="auto", batch_size=8):
    """source: 'auto' (local -> hub -> generate), 'local', 'hub' or 'generate'."""
    template = family_cfg(family)["chat_template"]
    path = _local_path(cache_dir, family, domain)
    if source in ("auto", "local") and os.path.exists(path):
        with open(path) as f:
            print(f"[completions] {family}/{domain}: local cache {path}")
            return _validate(json.load(f), domain, template, path)
    if source == "local":
        raise FileNotFoundError(path)
    if source in ("auto", "hub"):
        try:
            completions, meta = _download(family, domain, cache_dir)
            _validate(completions, domain, template, "hub")
            _write(path, completions, meta)
            print(f"[completions] {family}/{domain}: downloaded from {meta['repo']}")
            return completions
        except Exception as e:
            if source == "hub":
                raise
            print(f"[completions] {family}/{domain}: hub unavailable ({type(e).__name__}); generating")
    completions, meta = generate(family, domain, batch_size=batch_size)
    _validate(completions, domain, template, "generated")
    _write(path, completions, meta)
    return completions


def main():
    ap = argparse.ArgumentParser(description="Download or generate expert completions")
    ap.add_argument("--family", required=True, choices=sorted(models_cfg()["families"]))
    ap.add_argument("--domain", required=True, choices=DOMAINS + ["all"])
    ap.add_argument("--cache_dir", default="./expert_completions")
    ap.add_argument("--source", default="auto", choices=["auto", "local", "hub", "generate"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--expert_model", default=None, help="with --source generate: override the expert (e.g. a local copy)")
    a = ap.parse_args()
    for d in (DOMAINS if a.domain == "all" else [a.domain]):
        if a.source == "generate":
            c, meta = generate(a.family, d, batch_size=a.batch_size, expert_model=a.expert_model)
            _write(_local_path(a.cache_dir, a.family, d), c, meta)
        else:
            get_completions(a.family, d, a.cache_dir, source=a.source, batch_size=a.batch_size)


if __name__ == "__main__":
    main()
