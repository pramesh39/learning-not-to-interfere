"""Loads the YAML configs under llm/configs and defines per-domain constants."""
import os

import yaml

LLM_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(LLM_DIR)
CONFIG_DIR = os.path.join(LLM_DIR, "configs")

DOMAINS = ["instruction", "math", "multilingual", "coding", "safety"]

# Benchmark that defines each domain's data, and the file name its expert
# completions are stored under (locally and on the Hub).
BENCHMARK = {
    "instruction": "ifeval",
    "math": "gsm8k",
    "multilingual": "arc",
    "coding": "mbppplus",
    "safety": "xstest",
}

# Maximum tokenized length (prompt + completion) used during RI. IFEval is the
# only domain whose prompt+completion regularly exceeds 512 tokens (~25% of
# examples), so it alone gets 1024.
MAX_SEQ_LENGTH = {
    "instruction": 1024,
    "math": 512,
    "multilingual": 512,
    "coding": 512,
    "safety": 512,
}


def _load(name):
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return yaml.safe_load(f)


def models_cfg():
    return _load("models.yaml")


def family_cfg(family):
    fams = models_cfg()["families"]
    if family not in fams:
        raise ValueError(f"unknown family {family!r}; expected one of {sorted(fams)}")
    return fams[family]


def splits_cfg():
    return _load("splits.yaml")


def merging_cfg():
    return _load("merging.yaml")


def grid(spec):
    """{start, stop, step} -> list of floats, both endpoints included."""
    n = int(round((spec["stop"] - spec["start"]) / spec["step"]))
    return [round(spec["start"] + i * spec["step"], 6) for i in range(n + 1)]
