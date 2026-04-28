"""Summarize a quality-score file: distributions, per-question spreads, and
side-by-side examples of best vs. worst CoT for diverse questions.

Usage:
    PYTHONPATH=. python -m sft.methods.tokenskip.quality_report \
        --scores data/sft/tokenskip/scores/math_raw/scores_ratio_0.5.jsonl \
        --compressed data/sft/tokenskip/compressed/math_raw/compressed_ratio_0.5.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict


def _load_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _pctile(sorted_vals, p):
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    return sorted_vals[min(int(p / 100 * n), n - 1)]


def _show_dist(vals, label):
    s = sorted(vals)
    print(f"\n{label} (N={len(s)})")
    for p in (0, 1, 5, 25, 50, 75, 95, 99, 100):
        v = _pctile(s, p) if 0 <= p < 100 else s[-1]
        print(f"  p{p:3d}: {v:9.3f}")
    print(f"  mean: {statistics.mean(s):9.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--compressed", default=None,
                        help="Optional. If provided, prints best/worst CoT examples.")
    parser.add_argument("--top", type=float, default=0.25)
    parser.add_argument("--score-key", default="logprob_post_cot",
                        choices=["logprob_answer", "logprob_boxed", "logprob_post_cot"])
    parser.add_argument("--n-examples", type=int, default=4)
    args = parser.parse_args()

    scores = _load_jsonl(args.scores)
    scores = [r for r in scores if not r.get("skipped")]
    print(f"Loaded {len(scores)} scored rows from {args.scores}")

    by_q = defaultdict(list)
    for r in scores:
        by_q[r["question_source_id"]].append(r)
    n_dist = [len(v) for v in by_q.values()]
    print(f"Questions: {len(by_q)}, avg CoTs/q={statistics.mean(n_dist):.2f}, "
          f"min={min(n_dist)}, max={max(n_dist)}")

    # Per-record distributions
    for k in ("logprob_answer", "logprob_boxed", "logprob_post_cot"):
        _show_dist([r[k] for r in scores], f"{k} (per record)")

    # Within-question spreads
    print("\n=== Within-question spread (max-min per metric) ===")
    multi = [v for v in by_q.values() if len(v) >= 2]
    print(f"(over {len(multi)} questions with ≥2 CoTs)")
    for k in ("logprob_answer", "logprob_boxed", "logprob_post_cot"):
        spreads = [max(r[k] for r in rows) - min(r[k] for r in rows) for rows in multi]
        flat = sum(1 for d in spreads if d < 0.001)
        print(f"  {k:18s}: mean={statistics.mean(spreads):7.3f}  "
              f"median={statistics.median(spreads):7.3f}  "
              f"max={max(spreads):7.3f}  flat<0.001={flat}/{len(spreads)}")

    # Top-X% per question stats
    print(f"\n=== Top {args.top*100:.0f}% per-question filter (score_key={args.score_key}) ===")
    kept_n = 0
    total_n = 0
    kept_per_q_dist = []
    for qsid, rows in by_q.items():
        n = len(rows)
        k = max(1, round(n * args.top))
        kept_n += k
        total_n += n
        kept_per_q_dist.append(k)
    kept_dist_counts = defaultdict(int)
    for k in kept_per_q_dist:
        kept_dist_counts[k] += 1
    print(f"Total kept: {kept_n}/{total_n} ({100*kept_n/total_n:.1f}%)")
    for k in sorted(kept_dist_counts):
        print(f"  k={k}: {kept_dist_counts[k]} questions")

    # Show n_examples questions: best vs worst CoT, side by side
    if args.compressed and args.n_examples > 0:
        compressed_rows = _load_jsonl(args.compressed)
        comp_by_sid = {r["source_id"]: r for r in compressed_rows}

        # Pick questions with most CoTs and largest spread on score_key
        candidates = []
        for qsid, rows in by_q.items():
            if len(rows) < 4:
                continue
            v = [r[args.score_key] for r in rows]
            spread = max(v) - min(v)
            candidates.append((spread, qsid, rows))
        candidates.sort(reverse=True)

        print(f"\n=== Top {args.n_examples} questions by spread on {args.score_key} ===")
        for spread, qsid, rows in candidates[: args.n_examples]:
            rows = sorted(rows, key=lambda r: r[args.score_key], reverse=True)
            qrec = comp_by_sid.get(rows[0]["source_id"])
            if not qrec:
                continue
            print(f"\n--- {qsid}: spread={spread:.3f}, n={len(rows)}, gold={qrec['answer']!r} ---")
            print(f"Question: {qrec['question'][:150]}")
            print(f"\nBest (lp={rows[0][args.score_key]:.3f}, src={rows[0]['source_id']}):")
            best_cr = comp_by_sid[rows[0]['source_id']]['compressed_reasoning']
            print(f"  ({best_cr[:400]}...)")
            print(f"\nWorst (lp={rows[-1][args.score_key]:.3f}, src={rows[-1]['source_id']}):")
            worst_cr = comp_by_sid[rows[-1]['source_id']]['compressed_reasoning']
            print(f"  ({worst_cr[:400]}...)")


if __name__ == "__main__":
    main()
