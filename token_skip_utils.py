"""Shared helpers for TokenSkip-style ratio conditioning."""

from __future__ import annotations

from typing import Optional


def normalize_compression_ratio(ratio: float | int | str | None) -> Optional[float]:
    """Parse and validate a compression ratio in (0, 1]."""
    if ratio is None:
        return None

    value = float(ratio)
    if value <= 0.0 or value > 1.0:
        raise ValueError(f"Compression ratio must be in (0, 1], got {ratio}.")
    return value


def format_compression_ratio(ratio: float | int | str) -> str:
    """Format a compression ratio for inline prompt injection."""
    value = normalize_compression_ratio(ratio)
    assert value is not None
    return f"{value:g}"


def infer_model_family(model_name_or_path: str | None) -> str:
    """Infer model family from a checkpoint path/name."""
    if not model_name_or_path:
        return "qwen"

    lower = model_name_or_path.lower()
    if "qwen" in lower:
        return "qwen"
    if "llama-3" in lower or "llama3" in lower:
        return "llama3"
    return "qwen"


def get_ratio_boundary_token(
    model_family: str,
    boundary_token: str | None = None,
) -> str:
    """Return the separator token used for inline ratio conditioning."""
    if boundary_token:
        return boundary_token

    family = (model_family or "qwen").lower()
    if family == "qwen":
        return "<|eot_id|>"
    if family == "llama3":
        return "<|eot_id|>"
    raise ValueError(f"Unsupported TokenSkip model family: {model_family}")


def build_inline_ratio_suffix(
    ratio: float | int | str | None,
    *,
    model_family: str = "qwen",
    boundary_token: str | None = None,
) -> str:
    """Build an inline suffix such as ``<|eot_id|>0.5<|eot_id|>``."""
    value = normalize_compression_ratio(ratio)
    if value is None or abs(value - 1.0) < 1e-9:
        return ""

    boundary = get_ratio_boundary_token(model_family, boundary_token=boundary_token)
    ratio_text = format_compression_ratio(value)
    return f"{boundary}{ratio_text}{boundary}"


def append_inline_ratio_condition(
    text: str,
    ratio: float | int | str | None,
    *,
    model_family: str | None = None,
    model_name_or_path: str | None = None,
    boundary_token: str | None = None,
) -> str:
    """Append TokenSkip inline ratio conditioning to a prompt string."""
    family = model_family or infer_model_family(model_name_or_path)
    suffix = build_inline_ratio_suffix(
        ratio,
        model_family=family,
        boundary_token=boundary_token,
    )
    return text if not suffix else f"{text}{suffix}"
