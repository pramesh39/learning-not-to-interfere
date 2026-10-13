"""Prompts and datasets for the five-benchmark subset.

Every prompt is formatted exactly as the corresponding evaluation formats it,
so RI adapts each expert on the same text the evaluation later scores. This
module is the single source of prompts for both RI training and expert
completion generation.

Domain -> benchmark:
    instruction  -> IFEval   (google/IFEval)
    math         -> GSM8K    (openai/gsm8k, test)
    multilingual -> ARC-ml   (alexandrainst/m_arc, test; de/es/fr/ru)
    coding       -> MBPP+    (evalplus/mbppplus, test)
    safety       -> XSTest   (safety-eval's exaggerated_safety.json)
"""
import json
import os
import re
from functools import partial

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from llm.config import REPO_ROOT, splits_cfg
from llm.ri.chat_templates import xstest_chat_wrap

ARC_LANGS = ["de", "es", "fr", "ru"]
XSTEST_PATH = os.path.join(REPO_ROOT, "third_party", "safety-eval", "evaluation", "tasks",
                           "generation", "xstest", "exaggerated_safety.json")


# ----------------------------------------------------------------- prompts
def _ifeval_prompts():
    return list(load_dataset("google/IFEval", split="train")["prompt"])


def _gsm8k_prompts():
    # lm-eval task gsm8k_cot_zeroshot_boxed: doc_to_text
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [f"Q: {q}\nA: Let's think step by step." for q in ds["question"]]


_BRACKETS = re.compile(r"\[.*?\]")


def _okapi_preprocess(text):
    # lm-eval okapi/arc_multilingual/utils.py
    text = text.strip().replace(" [title]", ". ")
    text = _BRACKETS.sub("", text)
    return text.replace("  ", " ")


def _arc_prompts_by_lang():
    return {lang: [f"Question: {_okapi_preprocess(x)}\nAnswer:"
                   for x in load_dataset("alexandrainst/m_arc", lang, split="test")["instruction"]]
            for lang in ARC_LANGS}


def _mbppplus_rows():
    try:
        ds = load_dataset("evalplus/mbppplus", split="test")
        return list(ds["prompt"]), [t[0] for t in ds["test_list"]]
    except TypeError:
        # Some `datasets` versions fail to parse this dataset's metadata;
        # read the parquet shard directly instead.
        import glob
        import pyarrow.parquet as pq
        from huggingface_hub import snapshot_download
        d = snapshot_download("evalplus/mbppplus", repo_type="dataset", allow_patterns="data/test-*.parquet")
        rows = pq.read_table(sorted(glob.glob(os.path.join(d, "data", "test-*.parquet")))[0]).to_pylist()
        return [r["prompt"] for r in rows], [r["test_list"][0] for r in rows]


def _mbppplus_prompts():
    # bigcode-evaluation-harness MBPPPlus.get_prompt(): description + first test, as a docstring
    descriptions, first_tests = _mbppplus_rows()
    return [f'"""\n{d}\n{t}\n"""\n' for d, t in zip(descriptions, first_tests)]


def _xstest_prompts(chat_template):
    with open(XSTEST_PATH) as f:
        data = json.load(f)
    wrap = xstest_chat_wrap(chat_template)
    return [wrap.format(instruction=ex["prompt"]) for ex in data]


def full_prompts(domain, chat_template):
    """All prompts of a benchmark's full test set, in loader order."""
    if domain == "instruction":
        return _ifeval_prompts()
    if domain == "math":
        return _gsm8k_prompts()
    if domain == "multilingual":
        by_lang = _arc_prompts_by_lang()
        return [p for lang in ARC_LANGS for p in by_lang[lang]]
    if domain == "coding":
        return _mbppplus_prompts()
    if domain == "safety":
        return _xstest_prompts(chat_template)
    raise ValueError(f"unknown domain {domain!r}")


def split_indices(domain, split):
    """Indices into full_prompts(domain) belonging to `split` ('val' or 'test')."""
    cfg = splits_cfg()[domain]
    if domain == "multilingual":
        idx, offset = [], 0
        for lang in cfg["lang_order"]:
            b = cfg["per_lang"][lang][split]
            idx.extend(range(offset + b["start"], offset + b["end"]))
            offset += cfg["lang_sizes"][lang]
        return idx
    b = cfg[split]
    return list(range(b["start"], b["end"]))


# ---------------------------------------------------------------- datasets
class TokenizedDataset(Dataset):
    """prompt (+ " " + completion) + EOS, truncated to max_length."""

    def __init__(self, prompts, completions, tokenizer, max_length):
        if completions is None:
            texts = [p + tokenizer.eos_token for p in prompts]
        else:
            texts = [p + " " + c + tokenizer.eos_token for p, c in zip(prompts, completions)]
        self.encodings = [tokenizer(t, truncation=True, max_length=max_length) for t in texts]

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, i):
        e = self.encodings[i]
        return {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]}


def build_dataset(domain, tokenizer, chat_template, max_length, completions=None, split="test"):
    """Tokenized dataset for `split` of `domain`.

    completions: None (prompt-only) or the FULL, unsliced list of expert
    completions aligned with full_prompts(domain); it is sliced here so the
    two stay index-aligned.
    """
    prompts = full_prompts(domain, chat_template)
    if completions is not None and len(completions) != len(prompts):
        raise ValueError(f"{domain}: {len(completions)} completions for {len(prompts)} prompts")
    idx = split_indices(domain, split)
    p = [prompts[i] for i in idx]
    c = None if completions is None else [completions[i] for i in idx]
    return TokenizedDataset(p, c, tokenizer, max_length)


def _collate(batch, pad_id):
    ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(b["input_ids"]) for b in batch],
                                          batch_first=True, padding_value=pad_id)
    attn = torch.nn.utils.rnn.pad_sequence([torch.tensor(b["attention_mask"]) for b in batch],
                                           batch_first=True, padding_value=0)
    return {"input_ids": ids, "attention_mask": attn}


def build_loader(dataset, tokenizer, batch_size):
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                      collate_fn=partial(_collate, pad_id=tokenizer.pad_token_id))


def cycle(loader):
    while True:
        for batch in loader:
            yield batch
