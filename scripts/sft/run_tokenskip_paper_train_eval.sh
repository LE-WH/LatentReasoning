#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON_BIN:-python}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-qwen2.5-3b-tokenskip-paper-repro-20260410-a100nv-80g-4gpu}"
MODEL_FAMILY="${MODEL_FAMILY:-qwen}"
DATA_PATH="${DATA_PATH:-data/sft/tokenskip/repro/${RUN_TAG}/gsm8k_tokenskip_rawtext.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/sft/gsm8k/${RUN_TAG}}"
EVAL_DIR="${EVAL_DIR:-${OUTPUT_DIR}/eval}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
EVAL_RATIOS="${EVAL_RATIOS:-1.0,0.9,0.8,0.7,0.6,0.5}"

mkdir -p "$EVAL_DIR"

cat <<EOF
============================================================
TokenSkip Paper Train+Eval Resume
  Run tag: ${RUN_TAG}
  Data path: ${DATA_PATH}
  Output dir: ${OUTPUT_DIR}
  Model family: ${MODEL_FAMILY}
  GPU: ${GPU}
  Eval ratios: ${EVAL_RATIOS}
============================================================
EOF

if [ ! -f "$DATA_PATH" ]; then
    echo "Missing dataset: $DATA_PATH" >&2
    exit 1
fi

echo ""
echo "== Step 4: Train =="
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON" -m sft.train \
    --config-name gsm8k_tokenskip_rawtext \
    data.path="$DATA_PATH" \
    training.output_dir="$OUTPUT_DIR" \
    training.run_name="$RUN_TAG"

echo ""
echo "== Step 5: Evaluate standard TokenSkip ratios =="
for ratio in ${EVAL_RATIOS//,/ }; do
    echo "-- ratio=${ratio}"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON" -m sft.eval_raw \
        --model "${OUTPUT_DIR}/merged" \
        --tokenskip-prompt \
        --compression-ratio "$ratio" \
        --model-family "$MODEL_FAMILY" \
        --output "${EVAL_DIR}/gsm8k_eval_ratio_${ratio}.jsonl"
done

echo ""
echo "== Step 6: Summarize metrics =="
PYTHONPATH=. "$PYTHON" - <<'PY' "$EVAL_DIR" "$EVAL_RATIOS"
import json
import sys
from pathlib import Path

eval_dir = Path(sys.argv[1])
ratios = [part.strip() for part in sys.argv[2].split(",") if part.strip()]
rows = []
for ratio in ratios:
    path = eval_dir / f"gsm8k_eval_ratio_{ratio}.jsonl"
    with open(path) as f:
        summary = json.loads(next(f))
    rows.append(
        {
            "ratio": ratio,
            "accuracy": summary["accuracy"],
            "correct": summary["correct"],
            "num_samples": summary["num_samples"],
            "avg_tokens": summary["avg_tokens"],
            "avg_correct_tokens": summary["avg_correct_tokens"],
        }
    )

summary_json = eval_dir / "summary.json"
summary_tsv = eval_dir / "summary.tsv"
summary_json.write_text(json.dumps(rows, indent=2) + "\n")
with open(summary_tsv, "w") as f:
    f.write("ratio\taccuracy\tcorrect\tnum_samples\tavg_tokens\tavg_correct_tokens\n")
    for row in rows:
        f.write(
            f"{row['ratio']}\t{row['accuracy']:.6f}\t{row['correct']}\t"
            f"{row['num_samples']}\t{row['avg_tokens']:.2f}\t"
            f"{row['avg_correct_tokens']:.2f}\n"
        )
print(summary_tsv)
PY

echo ""
echo "============================================================"
echo "TokenSkip paper train+eval resume complete"
echo "Summary: ${EVAL_DIR}/summary.tsv"
echo "============================================================"
