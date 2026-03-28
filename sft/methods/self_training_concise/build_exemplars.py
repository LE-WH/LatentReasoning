"""Build few-shot exemplars from Naive BoN CoT samples.

Reads the zero-shot CoT samples file, selects the N shortest correct
reasoning traces across all questions, and saves them as a JSON file
for use with --few-shot-path in sample.py.

Following the original paper: select 128 shortest correct samples,
then at inference time randomly draw 8 per prompt.

Usage:
    python -m sft.methods.self_training_concise.build_exemplars \
        --cot-samples-path data/sft/cot_samples/gsm8k_cot_zero-shot.jsonl \
        --output data/sft/cot_samples/gsm8k_few_shot_exemplars.json \
        --num-exemplars 128
"""

import argparse
import json
import logging
from pathlib import Path

from ragen.env.reward_funcs import init_reward_func

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build few-shot exemplars from Naive BoN CoT samples"
    )
    parser.add_argument("--cot-samples-path", type=str, required=True,
                        help="Path to zero-shot CoT samples (output of sample.py)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSON path for exemplars")
    parser.add_argument("--num-exemplars", type=int, default=128,
                        help="Number of exemplars to select (default: 128)")
    parser.add_argument("--max-think-tokens", type=int, default=None,
                        help="Max think tokens filter (paper: 511 for GSM8k)")
    args = parser.parse_args()

    # Reward function for scoring candidates (shared with RL pipeline).
    reward_func = init_reward_func({
        "type": "linear",
        "max_length": args.max_think_tokens or 2048,
    })

    # Collect all correct samples with their metadata
    all_correct = []
    seen_texts = set()  # for exact string dedup (paper requirement)
    dedup_count = 0
    with open(args.cot_samples_path) as f:
        for line in f:
            record = json.loads(line)
            for s in record["samples"]:
                if not (
                    s["is_correct"]
                    and s.get("think") is not None
                    and s["think"].strip()
                    and s.get("think_token_count", 0) > 0
                ):
                    continue

                # Exact string dedup (paper: correctness -> dedup -> cutoff)
                text_key = s["think"].strip()
                if text_key in seen_texts:
                    dedup_count += 1
                    continue
                seen_texts.add(text_key)

                # Hard cutoff (paper: token_count < 511 for GSM8k)
                if (args.max_think_tokens is not None
                        and s.get("think_token_count", 0) >= args.max_think_tokens):
                    continue

                all_correct.append({
                    "benchmark": record.get("benchmark"),
                    "source_id": record["source_id"],
                    "question": record["question"],
                    "gold_answer": record["gold_answer"],
                    "reasoning": s["think"],
                    "answer": s["answer"],
                    "solution": s.get("response_text") or s.get("text"),
                    "think_token_count": s["think_token_count"],
                })

    logger.info(f"Found {len(all_correct)} correct samples total (dedup removed: {dedup_count})")

    # Per question, keep the best candidate by reward_func
    by_source: dict[str, dict] = {}
    for item in all_correct:
        sid = item["source_id"]
        item_score = reward_func(1.0, item["think_token_count"])
        if sid not in by_source:
            by_source[sid] = (item, item_score)
        else:
            _, best_score = by_source[sid]
            if item_score > best_score:
                by_source[sid] = (item, item_score)

    unique_best = [item for item, _score in by_source.values()]
    logger.info(f"Unique questions with correct samples: {len(unique_best)}")

    # Sort by reward (descending) and pick the N best
    unique_best.sort(
        key=lambda x: reward_func(1.0, x["think_token_count"]),
        reverse=True,
    )
    selected = unique_best[:args.num_exemplars]

    if not selected:
        raise ValueError("No valid exemplars selected; check correctness and max-think-tokens filter.")

    logger.info(
        f"Selected {len(selected)} exemplars, "
        f"token count range: {selected[0]['think_token_count']}-{selected[-1]['think_token_count']}"
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
