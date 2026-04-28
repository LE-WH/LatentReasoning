"""Filter compressed CoTs to top-X% per question using pre-computed scores.

Reads ``scores_ratio_R.jsonl`` (from quality_score.py) and the original
``compressed_ratio_R.jsonl``, groups records by ``question_source_id``, keeps
the top ``--top`` fraction per question (by ``logprob_answer``), and writes a
new compressed file the rest of the pipeline can consume.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter compressed CoTs to top-X% per question")
    parser.add_argument("--scores", type=str, required=True,
                        help="scores_ratio_R.jsonl from quality_score.py")
    parser.add_argument("--compressed", type=str, required=True,
                        help="compressed_ratio_R.jsonl (unfiltered)")
    parser.add_argument("--output", type=str, required=True,
                        help="Filtered output JSONL")
    parser.add_argument("--top", type=float, default=0.25,
                        help="Fraction to keep per question (e.g. 0.25 for top 25%%)")
    parser.add_argument("--score-key", type=str, default="logprob_post_cot",
                        choices=["logprob_answer", "logprob_boxed", "logprob_post_cot"])
    parser.add_argument("--min-per-question", type=int, default=1,
                        help="Always keep at least this many per question")
    args = parser.parse_args()

    if not (0.0 < args.top <= 1.0):
        raise ValueError("--top must be in (0, 1]")

    score_rows = _load_jsonl(args.scores)
    compressed_rows = _load_jsonl(args.compressed)

    score_by_sid = {row["source_id"]: row for row in score_rows if not row.get("skipped")}
    skipped_score_rows = sum(1 for row in score_rows if row.get("skipped"))

    # Group score rows by question_source_id.
    by_q: dict[str, list[dict]] = defaultdict(list)
    for row in score_rows:
        if row.get("skipped"):
            continue
        by_q[row["question_source_id"]].append(row)

    keep_sids: set[str] = set()
    kept_per_q_counts: list[int] = []
    n_per_q_counts: list[int] = []
    for qsid, rows in by_q.items():
        rows.sort(key=lambda r: r[args.score_key], reverse=True)
        n = len(rows)
        k = max(args.min_per_question, round(n * args.top))
        k = min(k, n)
        kept_per_q_counts.append(k)
        n_per_q_counts.append(n)
        for r in rows[:k]:
            keep_sids.add(r["source_id"])

    # Stats
    total_n = sum(n_per_q_counts)
    total_kept = sum(kept_per_q_counts)
    logger.info(
        "Questions: %d, total scored: %d, kept: %d (%.1f%%), avg per-q n=%.2f, avg per-q k=%.2f",
        len(by_q), total_n, total_kept,
        100 * total_kept / max(total_n, 1),
        total_n / max(len(by_q), 1),
        total_kept / max(len(by_q), 1),
    )
    if skipped_score_rows:
        logger.info("Score rows marked skipped (e.g. too long): %d", skipped_score_rows)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing_score = 0
    with open(out_path, "w") as f:
        for record in compressed_rows:
            sid = record.get("source_id")
            if sid not in score_by_sid:
                missing_score += 1
                continue
            if sid in keep_sids:
                # Attach the score for traceability downstream.
                enriched = dict(record)
                enriched["quality_score"] = score_by_sid[sid][args.score_key]
                enriched["quality_score_key"] = args.score_key
                f.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                written += 1

    logger.info("Wrote %d filtered records to %s (missing_score=%d)",
                written, out_path, missing_score)


if __name__ == "__main__":
    main()
