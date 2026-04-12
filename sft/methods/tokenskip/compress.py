"""Compress original CoTs with LLMLingua for TokenSkip."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL into memory."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], path: str) -> None:
    """Save a list of dicts as JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_ratio_pool(raw: str) -> list[float]:
    """Parse a comma-separated ratio pool."""
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress CoTs with LLMLingua")
    parser.add_argument("--input", type=str, required=True, help="Original CoT jsonl")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--ratio-pool", type=str, default="0.9,0.8,0.7,0.6,0.5")
    parser.add_argument(
        "--model-family",
        type=str,
        default="qwen",
        choices=["qwen", "llama3"],
        help="Model family for TokenSkip-specific compression settings.",
    )
    parser.add_argument(
        "--max-cot-tokens",
        type=int,
        default=500,
        help="Filter out original CoTs longer than this token count before compression.",
    )
    parser.add_argument(
        "--llmlingua-path",
        type=str,
        required=True,
        help="Path or HF id for LLMLingua-2 weights",
    )
    args = parser.parse_args()

    try:
        from llmlingua import PromptCompressor
    except ImportError as exc:
        raise ImportError(
            "TokenSkip compression requires `llmlingua`. Install it first "
            "(for example: `pip install llmlingua`)."
        ) from exc

    original_records = load_jsonl(args.input)
    skipped_for_length = 0
    filtered_records = []
    for record in original_records:
        if not (record.get("is_correct") and record.get("reasoning")):
            continue
        reasoning_tokens = record.get("reasoning_token_count")
        if reasoning_tokens is not None and reasoning_tokens > args.max_cot_tokens:
            skipped_for_length += 1
            continue
        filtered_records.append(record)
    original_records = [
        record for record in filtered_records
    ]
    logger.info(
        "Loaded %d correct original CoTs after filtering (skipped_for_length=%d, max_cot_tokens=%d)",
        len(original_records),
        skipped_for_length,
        args.max_cot_tokens,
    )

    compressor = PromptCompressor(
        model_name=args.llmlingua_path,
        use_llmlingua2=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ratio in parse_ratio_pool(args.ratio_pool):
        compressed_records = []
        for record in original_records:
            compress_kwargs = {"rate": ratio}
            if args.model_family == "llama3":
                compress_kwargs.update(
                    force_tokens=["Step", ":"],
                    force_reserve_digit=True,
                    drop_consecutive=True,
                )
            result = compressor.compress_prompt(record["reasoning"], **compress_kwargs)
            compressed_records.append(
                {
                    "benchmark": record["benchmark"],
                    "source_id": record["source_id"],
                    "question": record["question"],
                    "gold_answer": record["gold_answer"],
                    "answer": record["answer"],
                    "compression_ratio": ratio,
                    "original_reasoning": record["reasoning"],
                    "compressed_reasoning": result["compressed_prompt"],
                    "original_reasoning_token_count": result["origin_tokens"],
                    "compressed_reasoning_token_count": result["compressed_tokens"],
                    "actual_compression_rate": result["rate"],
                }
            )

        ratio_tag = f"{ratio:g}"
        out_path = output_dir / f"compressed_ratio_{ratio_tag}.jsonl"
        save_jsonl(compressed_records, str(out_path))
        logger.info("Saved ratio=%s compressed CoTs to %s", ratio_tag, out_path)


if __name__ == "__main__":
    main()
