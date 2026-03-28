"""Prompt templates and answer extraction for Concise SFT sampling.

Uses RAGEN-compatible XML format (<think>/<answer> tags) for compatibility
with RAGEN's evaluation and RL training pipelines.

Paper: https://arxiv.org/abs/2502.20122
Code:  https://github.com/TergelMunkhbat/concise-reasoning
"""

import re


SYSTEM_PROMPTS = {
    "gsm8k": (
        "You are solving a math word problem. "
        "Read the problem carefully and compute the final numerical answer. "
        "Think step by step before answering. "
        "Write your reasoning in <think> tags, then give your final answer in <answer> tags. "
        "Example: <think>Step-by-step reasoning...</think><answer>42</answer>"
    ),
}

DEFAULT_NUM_FEW_SHOT = 8


def extract_think_answer(text: str) -> tuple[str | None, str | None]:
    """Extract <think> and <answer> content from model output."""
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else None

    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if think_match:
        think = think_match.group(1).strip()
    else:
        # Fallback: text between <think> and <answer> (unclosed think tag)
        fallback = re.search(r"<think>(.*?)<answer>", text, re.DOTALL)
        think = fallback.group(1).strip() if fallback else None

    return think, answer


def get_system_prompt(benchmark: str) -> str:
    """Return system prompt for the benchmark."""
    return SYSTEM_PROMPTS.get(benchmark, "")
