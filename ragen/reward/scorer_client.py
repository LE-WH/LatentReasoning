"""HTTP client for the gold log-probability scorer server.

Used by the RL training loop to compute continuous rewards via the remote
scorer server, replacing binary correctness from env.step().

Migrated from scalable-latent-reasoning/dual_cot/reward_math_gold_logprob_remote.py

Environment variables:
    SCORER_URL:     Server endpoint (default: http://127.0.0.1:8009)
    SCORER_TIMEOUT: Request timeout in seconds (default: 120)
"""
from __future__ import annotations

import os
from typing import Any, List

import requests


SCORER_URL = os.environ.get("SCORER_URL", "http://127.0.0.1:8009")
TIMEOUT = float(os.environ.get("SCORER_TIMEOUT", "120"))


def compute_score(
    prompt_messages: list[dict[str, str]],
    solution_str: str,
    ground_truth: str,
) -> float:
    """Score a single (prompt, CoT, answer) triple via the remote server."""
    payload = {
        "prompt_messages": prompt_messages,
        "solution_str": solution_str,
        "ground_truth": ground_truth,
    }

    resp = requests.post(
        f"{SCORER_URL.rstrip('/')}/score",
        json=payload,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return float(resp.json()["score"])


def compute_score_batch(
    items: List[dict[str, Any]],
) -> List[float]:
    """Score a batch of items via the remote server.

    Each item should have keys: prompt_messages, solution_str, ground_truth.
    """
    payload = {"items": items}

    resp = requests.post(
        f"{SCORER_URL.rstrip('/')}/score_batch",
        json=payload,
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["scores"]
