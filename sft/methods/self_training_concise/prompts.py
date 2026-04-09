"""Prompt templates and answer extraction for Concise SFT sampling.

Supports two formats:
  - xml: RAGEN-compatible (<think>/<answer> tags)
  - paper: Original paper format ('The answer is')

Paper: https://arxiv.org/abs/2502.20122
Code:  https://github.com/TergelMunkhbat/concise-reasoning
"""

import re

_LATENT_RE = re.compile(r"<\|latent_\d+\|>")


def strip_latent_tokens(text: str) -> str:
    """Remove <|latent_N|> tokens from model output."""
    return _LATENT_RE.sub("", text)


XML_SYSTEM_PROMPTS = {
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

PAPER_SYSTEM_PROMPTS = {
    "gsm8k": (
        "Your task is to answer the question below. "
        "Give step by step reasoning before you answer, and when you're ready to answer, "
        "please use the format 'The answer is'"
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


def extract_numeric_answer(text: str) -> tuple[str | None, str | None]:
    """Extract answer from 'The answer is ...' style text (paper format).

    Returns (think, answer) tuple to match extract_think_answer signature.
    """
    think = text.strip()
    parts = text.lower().split("answer is")
    answer_flag = len(parts) > 1
    candidate_text = parts[-1] if answer_flag else text.lower()
    candidate_text = candidate_text.replace(",", "")
    matches = re.findall(r"-?\d+\.?\d*", candidate_text)
    if not matches:
        return think, None
    answer = matches[0] if answer_flag else matches[-1]
    if re.match(r"^-?\d+\.\d+$", answer):
        answer = answer.rstrip("0").rstrip(".")
    return think, answer


def get_system_prompt(benchmark: str, fmt: str = "xml") -> str:
    """Return system prompt for the benchmark."""
    prompts = XML_SYSTEM_PROMPTS if fmt == "xml" else PAPER_SYSTEM_PROMPTS
    return prompts.get(benchmark, "")
