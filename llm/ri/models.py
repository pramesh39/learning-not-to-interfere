"""Model and tokenizer loading for RI."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(path):
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_frozen_model(path, dtype=torch.bfloat16, device="cuda"):
    """A frozen teacher: the original expert, or the shared base model."""
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_student(path, dtype=torch.bfloat16, device="cuda"):
    """A fully trainable copy of an expert (full-parameter adaptation, not LoRA).

    Weights, gradients and AdamW state are all kept in bf16, and gradient
    checkpointing is enabled to bound activation memory.
    """
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    # Some expert generation configs set sampling fields with do_sample=False,
    # which save_pretrained rejects. This only affects saved metadata.
    gc = model.generation_config
    if not gc.do_sample and (gc.temperature != 1.0 or gc.top_p != 1.0):
        gc.do_sample = True
    return model


def save_student(model, tokenizer, path):
    # Gemma-2 sets cache_implementation="hybrid", which save_pretrained rejects
    # for a model trained with use_cache=False. Generation-time only; harmless.
    gc = getattr(model, "generation_config", None)
    if gc is not None and getattr(gc, "cache_implementation", None):
        gc.cache_implementation = None
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
