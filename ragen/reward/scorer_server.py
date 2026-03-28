"""Gold-answer log-probability scorer server.

Computes P(gold_answer | prompt + latent_CoT) as a continuous reward signal
for RL training. The model scores how likely the correct answer is given the
reasoning trajectory, providing a smoother gradient than binary correctness.

Migrated from scalable-latent-reasoning/dual_cot/scorer_server.py

Usage:
    python -m ragen.reward.scorer_server \
        --model_dir ./checkpoints/dual_qwen_3b \
        --port 8009
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .math_answer_utils import normalize_final_answer


def _resolve_dtype(name: str) -> torch.dtype:
    name = (name or "bfloat16").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


class ScoreRequest(BaseModel):
    prompt_messages: list[dict[str, str]]
    solution_str: str
    ground_truth: str


class BatchScoreRequest(BaseModel):
    items: list[ScoreRequest]


class GoldLogProbServer:
    """Scores CoT trajectories by computing mean token log-prob of the gold answer.

    Given: prompt + latent reasoning (up to </think>) + gold answer tokens
    Returns: mean log P(answer_token_i | prefix) across answer tokens

    This rewards reasoning that makes the correct answer more predictable,
    providing a continuous signal rather than binary correct/incorrect.
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_concurrency: int = 8,
    ) -> None:
        self.model_dir = model_dir
        self.device = torch.device(device)
        self.dtype = _resolve_dtype(dtype)
        self.semaphore = asyncio.Semaphore(max_concurrency)

        meta_path = Path(model_dir) / "dual_vocab_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"dual_vocab_meta.json not found in {model_dir}")

        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.think_end_token = str(self.meta["think_end_token"])

        self.tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.eval()

        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    def _render_prompt(self, prompt_messages: list[dict[str, str]]) -> str:
        return self.tok.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _extract_latent_prefix(self, solution_str: str) -> tuple[str, bool]:
        if self.think_end_token in solution_str:
            pos = solution_str.find(self.think_end_token)
            end = pos + len(self.think_end_token)
            return solution_str[:end], True
        return solution_str, False

    def _score_answer_variant(self, prefix_text: str, answer_text: str) -> float:
        prefix_ids = self.tok(prefix_text, add_special_tokens=False)["input_ids"]
        answer_ids = self.tok(answer_text, add_special_tokens=False)["input_ids"]

        if len(answer_ids) == 0:
            return -100.0

        input_ids = prefix_ids + answer_ids
        input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids_t).logits

        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        targets = input_ids_t[:, 1:]

        prefix_len = len(prefix_ids)
        answer_start = max(prefix_len - 1, 0)
        answer_end = len(input_ids) - 1

        if answer_end <= answer_start:
            return -100.0

        selected = log_probs[0, answer_start:answer_end, :]
        target_slice = targets[0, answer_start:answer_end]
        token_lp = selected.gather(-1, target_slice.unsqueeze(-1)).squeeze(-1)

        return float(token_lp.mean().item())

    def score_one(
        self,
        prompt_messages: list[dict[str, str]],
        solution_str: str,
        ground_truth: str,
    ) -> float:
        gold = normalize_final_answer(ground_truth)
        if not gold:
            return -100.0

        prefix_rendered = self._render_prompt(prompt_messages)
        latent_prefix, has_end = self._extract_latent_prefix(solution_str)

        penalty = -2.0 if not has_end else 0.0
        full_prefix = prefix_rendered + latent_prefix

        variants = [gold]
        if not gold.startswith(" "):
            variants.append(" " + gold)

        best = max(self._score_answer_variant(full_prefix, v) for v in variants)
        return best + penalty


def build_app(server: GoldLogProbServer) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"ok": True, "model_dir": server.model_dir, "device": str(server.device)}

    @app.post("/score")
    async def score(req: ScoreRequest) -> dict[str, Any]:
        async with server.semaphore:
            score_value = await asyncio.to_thread(
                server.score_one,
                req.prompt_messages,
                req.solution_str,
                req.ground_truth,
            )
        return {"score": float(score_value)}

    @app.post("/score_batch")
    async def score_batch(req: BatchScoreRequest) -> dict[str, Any]:
        async with server.semaphore:
            scores = await asyncio.to_thread(
                lambda: [
                    float(server.score_one(x.prompt_messages, x.solution_str, x.ground_truth))
                    for x in req.items
                ]
            )
        return {"scores": scores}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Gold log-probability scorer server")
    parser.add_argument("--model_dir", type=str, default="./checkpoints/dual_qwen_3b")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_concurrency", type=int, default=8)
    args = parser.parse_args()

    server = GoldLogProbServer(
        model_dir=args.model_dir,
        device=args.device,
        dtype=args.dtype,
        max_concurrency=args.max_concurrency,
    )
    app = build_app(server)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
