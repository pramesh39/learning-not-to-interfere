"""Chat-template wrappers for XSTest prompts.

XSTest is the only benchmark whose prompts are wrapped in a chat template. The
wrappers here are byte-identical to the templates the safety evaluation applies
(third_party/patches/safety-eval-fork_chat-templates.patch), so RI adaptation
sees exactly the prompt text the evaluation scores.

The leading BOS token is deliberately omitted: the tokenizer prepends it
(add_special_tokens=True). Including it here would produce a double BOS.
"""

XSTEST_CHAT_WRAPS = {
    "llama3": (
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    ),
    "gemma": "<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n",
}


def xstest_chat_wrap(template):
    if template not in XSTEST_CHAT_WRAPS:
        raise ValueError(f"unknown chat template {template!r}; expected one of {sorted(XSTEST_CHAT_WRAPS)}")
    return XSTEST_CHAT_WRAPS[template]
