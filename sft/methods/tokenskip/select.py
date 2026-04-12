"""Build TokenSkip SFT samples from original and compressed CoTs."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from sft.data.base import SFTSample
from sft.methods.base import BaseSFTMethod
from sft.methods.tokenskip.prompts import build_tokenskip_raw_question

logger = logging.getLogger(__name__)


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _parse_ratio_pool(raw: str | list[float]) -> list[float]:
    if isinstance(raw, list):
        return [float(x) for x in raw]
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


class TokenSkipMethod(BaseSFTMethod):
    """TokenSkip SFT method with inline ratio conditioning."""

    name = "tokenskip"

    def build_samples(
        self,
        benchmark: str,
        raw_data: list[dict],
        **kwargs,
    ) -> list[SFTSample]:
        original_path = kwargs.get("original_cot_path")
        compressed_dir = kwargs.get("compressed_cot_dir")
        fmt = kwargs.get("fmt", "tokenskip_paper")
        model_family = kwargs.get("model_family", "qwen")
        ratio_pool = _parse_ratio_pool(kwargs.get("ratio_pool", "1.0,0.9,0.8,0.7,0.6,0.5"))
        seed = int(kwargs.get("seed", 42))

        if not original_path:
            raise ValueError("TokenSkipMethod requires --original-cot-path.")
        if not compressed_dir:
            raise ValueError("TokenSkipMethod requires --compressed-cot-dir.")

        original_records = {
            record["source_id"]: record
            for record in _load_jsonl(original_path)
            if record.get("is_correct") and record.get("reasoning")
        }

        compressed_by_ratio: dict[float, dict[str, dict]] = {}
        compressed_dir_path = Path(compressed_dir)
        for ratio in ratio_pool:
            if abs(ratio - 1.0) < 1e-9:
                continue
            ratio_tag = f"{ratio:g}"
            ratio_path = compressed_dir_path / f"compressed_ratio_{ratio_tag}.jsonl"
            if not ratio_path.exists():
                raise FileNotFoundError(
                    f"Missing compressed TokenSkip file for ratio {ratio_tag}: {ratio_path}"
                )
            compressed_by_ratio[ratio] = {
                record["source_id"]: record for record in _load_jsonl(str(ratio_path))
            }

        rng = random.Random(seed)
        samples = []
        skipped = 0

        if fmt == "xml":
            raise ValueError(
                "TokenSkip currently only supports response_format=tokenskip_paper."
            )

        for item in raw_data:
            source_id = item["source_id"]
            ratio = rng.choice(ratio_pool)

            if abs(ratio - 1.0) < 1e-9:
                selected = original_records.get(source_id)
                if selected is None:
                    skipped += 1
                    continue
                reasoning = selected["reasoning"]
                answer = selected.get("answer") or item["answer"]
            else:
                selected = compressed_by_ratio[ratio].get(source_id)
                if selected is None:
                    skipped += 1
                    continue
                reasoning = selected["compressed_reasoning"]
                answer = selected.get("answer") or item["answer"]

            question = build_tokenskip_raw_question(
                item["question"],
                ratio,
                model_family=model_family,
            )

            samples.append(
                SFTSample(
                    question=question,
                    answer=answer,
                    reasoning=reasoning,
                    benchmark=benchmark,
                    method=self.name,
                    source_id=source_id,
                    extra_metadata={
                        "compression_ratio": ratio,
                        "prompt_format": fmt,
                        "conditioned": abs(ratio - 1.0) >= 1e-9,
                    },
                )
            )

        logger.info(
            "TokenSkip SFT: built %d samples from %d questions (skipped=%d)",
            len(samples),
            len(raw_data),
            skipped,
        )
        return samples
