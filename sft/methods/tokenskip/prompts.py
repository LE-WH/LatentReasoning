"""Prompt helpers and answer extraction for TokenSkip."""

from __future__ import annotations

import re

from token_skip_utils import append_inline_ratio_condition


RAW_REASONING_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

RAW_CHAT_SYSTEM_PROMPT = "You are a helpful assistant."


def get_raw_chat_system_prompt() -> str:
    """Return the system prompt used by the paper-aligned chat setup."""
    return RAW_CHAT_SYSTEM_PROMPT


def build_tokenskip_raw_question(
    question: str,
    ratio: float | int | str | None,
    *,
    model_family: str = "qwen",
) -> str:
    """Build the raw-text user prompt used for TokenSkip SFT/eval."""
    conditioned_question = append_inline_ratio_condition(
        question,
        ratio,
        model_family=model_family,
    )
    return f"{RAW_REASONING_INSTRUCTION}\n{conditioned_question}"


def build_tokenskip_raw_chat_messages(
    question: str,
    ratio: float | int | str | None,
    *,
    model_family: str = "qwen",
) -> list[dict[str, str]]:
    """Build system/user chat turns for paper-faithful TokenSkip prompting."""
    return [
        {"role": "system", "content": get_raw_chat_system_prompt()},
        {
            "role": "user",
            "content": build_tokenskip_raw_question(
                question,
                ratio,
                model_family=model_family,
            ),
        },
    ]


def extract_boxed_or_numeric_answer(text: str) -> str | None:
    """Extract a GSM8K-style final answer from boxed or free-form text."""
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip().replace(",", "")

    parts = re.split(r"the final answer is:|the answer is", text, flags=re.IGNORECASE)
    candidate_text = parts[-1] if len(parts) > 1 else text
    candidate_text = candidate_text.replace(",", "")
    matches = re.findall(r"-?\d+\.?\d*", candidate_text)
    if not matches:
        return None
    answer = matches[0] if len(parts) > 1 else matches[-1]
    if re.match(r"^-?\d+\.\d+$", answer):
        answer = answer.rstrip("0").rstrip(".")
    return answer


def extract_raw_reasoning_and_answer(text: str) -> tuple[str | None, str | None]:
    """Split raw TokenSkip output into rationale and final answer."""
    answer = extract_boxed_or_numeric_answer(text)

    reasoning = text.strip()
    split_patterns = [
        r"\n\nThe final answer is:",
        r"\nThe final answer is:",
        r"\n\nThe answer is",
        r"\nThe answer is",
    ]
    for pattern in split_patterns:
        parts = re.split(pattern, reasoning, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            reasoning = parts[0].strip()
            break

    return (reasoning or None), answer
