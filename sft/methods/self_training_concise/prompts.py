"""Prompt templates and answer extraction for Concise SFT sampling.

Uses RAGEN-compatible XML format (<think>/<answer> tags) for compatibility
with RAGEN's evaluation and RL training pipelines.

Paper: https://arxiv.org/abs/2502.20122
Code:  https://github.com/TergelMunkhbat/concise-reasoning
"""

import re

_LATENT_RE = re.compile(r"<\|latent_\d+\|>")


def strip_latent_tokens(text: str) -> str:
    """Remove <|latent_N|> tokens from model output."""
    return _LATENT_RE.sub("", text)


SYSTEM_PROMPTS = {
    "gsm8k": (
        "You are solving a math word problem. "
        "Read the problem carefully and compute the final numerical answer. "
        "Think step by step before answering. "
        "Write your reasoning in <think> tags, then give your final answer in <answer> tags. "
        "Example: <think>Step-by-step reasoning...</think><answer>42</answer>"
    ),
    "math": (
        "You are solving a competition math problem. "
        "Read the problem carefully and solve it step by step. "
        "Write your reasoning in <think> tags, then give your final answer in <answer> tags. "
        "Example: <think>Step-by-step reasoning...</think><answer>\\frac{1}{2}</answer>"
    ),
}

DEFAULT_NUM_FEW_SHOT = 8


def extract_think_answer(text: str) -> tuple[str | None, str | None]:
    """Extract <think> and <answer> content from model output.

    Latent tokens (``<|latent_N|>``) are stripped before matching so that
    dual-vocab model outputs are handled correctly.  For dual-vocab models
    the "think" phase is encoded entirely as latent tokens — the visible
    output typically starts with ``</think>`` followed by the answer.  In
    that case we treat the text between ``</think>`` and ``<answer>`` as the
    visible reasoning (the latent tokens *are* the actual thinking).
    """
    clean = strip_latent_tokens(text)

    answer_match = re.search(r"<answer>(.*?)</answer>", clean, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else None

    think_match = re.search(r"<think>(.*?)</think>", clean, re.DOTALL)
    if think_match:
        think = think_match.group(1).strip()
    else:
        # Dual-vocab fallback: latent tokens replaced the <think> content,
        # so visible text starts with </think>.  Treat text between
        # </think> and <answer> as the visible reasoning summary.
        dv_match = re.search(r"</think>(.*?)<answer>", clean, re.DOTALL)
        if dv_match:
            think = dv_match.group(1).strip() or "(latent)"
        else:
            # Legacy fallback: unclosed <think> tag
            fallback = re.search(r"<think>(.*?)<answer>", clean, re.DOTALL)
            think = fallback.group(1).strip() if fallback else None

    return think, answer


def get_system_prompt(benchmark: str) -> str:
    """Return system prompt for the benchmark."""
    return SYSTEM_PROMPTS.get(benchmark, "")
