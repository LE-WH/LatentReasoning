"""Select the top-K shortest correct SoT samples per question.

Reads sharded ``collect.py`` outputs, groups by ``question_source_id``,
keeps up to ``--top-k`` correct samples sorted by reasoning length
ascending, drops questions with zero correct samples, writes one JSONL
row per kept sample (carrying the original ``response_text`` and
``reasoning``).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True,
                        help="Glob (pass quoted) for sharded jsonls e.g. 'data/sft/sot/collected/math/shard_*.jsonl'")
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    paths = sorted(Path().glob(args.input_glob))
    if not paths:
        raise SystemExit(f"No files matched glob: {args.input_glob}")
    logger.info("Reading %d shards", len(paths))

    by_q: dict[str, list[dict]] = defaultdict(list)
    n_total = 0
    for p in paths:
        with open(p) as f:
            for line in f:
                rec = json.loads(line)
                n_total += 1
                if rec.get("is_correct"):
                    by_q[rec["question_source_id"]].append(rec)
    n_questions = len(by_q)
    logger.info("Loaded %d total samples; %d questions have ≥1 correct", n_total, n_questions)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_kept = 0
    n_questions_kept = 0
    think_lens = []
    with open(args.output, "w") as f:
        for qid, recs in by_q.items():
            recs.sort(key=lambda r: r["reasoning_token_count"])
            keep = recs[: args.top_k]
            if not keep:
                continue
            n_questions_kept += 1
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_kept += 1
                think_lens.append(r["reasoning_token_count"])
    if think_lens:
        think_lens.sort()
        mean = sum(think_lens) / len(think_lens)
        med = think_lens[len(think_lens) // 2]
        p90 = think_lens[int(0.9 * len(think_lens))]
        logger.info("kept %d rows from %d questions → %s",
                    n_kept, n_questions_kept, args.output)
        logger.info("kept reasoning_token_count: mean=%.0f median=%d p90=%d", mean, med, p90)


if __name__ == "__main__":
    main()
