"""Convert selected SoT teacher rows into the SFT JSONL format
(``messages`` + ``metadata``) consumed by ``sft.train``.

Format mirrors ``sft/data/base.py`` ``SFTSample.to_jsonl_dict`` with
``response_format="paper"``-ish — but for SoT we want the assistant
content to be exactly ``<think>\\n{reasoning}\\n</think>\\n\\boxed{ANSWER}``,
and a single fixed system prompt (the SoT chunked-symbolism system
prompt). The user content is just the raw question — no SoT exemplars,
since the model is being SFT'd to produce SoT format unconditionally.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sft.methods.sot.prompts import SOT_CHUNKED_SYMBOLISM_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+",
                        help="One or more selected.jsonl files (combined into one SFT pool)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    n_in = 0
    n_out = 0
    benchmarks = {}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fout:
        for path in args.input:
            with open(path) as fin:
                for line in fin:
                    r = json.loads(line)
                    n_in += 1
                    benchmarks[r["benchmark"]] = benchmarks.get(r["benchmark"], 0) + 1
                    reasoning = (r["reasoning"] or "").strip()
                    answer = r["answer"] if r["answer"] is not None else r["gold_answer"]
                    assistant = f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"
                    msg = {
                        "messages": [
                            {"role": "system", "content": SOT_CHUNKED_SYMBOLISM_SYSTEM_PROMPT},
                            {"role": "user", "content": r["question"]},
                            {"role": "assistant", "content": assistant},
                        ],
                        "metadata": {
                            "benchmark": r["benchmark"],
                            "method": "sot_distill",
                            "source_id": r["source_id"],
                            "question_source_id": r["question_source_id"],
                            "reasoning_token_count": r["reasoning_token_count"],
                            "response_token_count": r["response_token_count"],
                        },
                    }
                    fout.write(json.dumps(msg, ensure_ascii=False) + "\n")
                    n_out += 1
    logger.info("Wrote %d SFT rows (from %d selected; per-benchmark: %s) → %s",
                n_out, n_in, benchmarks, args.output)


if __name__ == "__main__":
    main()
