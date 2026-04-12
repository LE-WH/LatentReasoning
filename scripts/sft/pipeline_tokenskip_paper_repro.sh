#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON_BIN:-python}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
MODEL_FAMILY="${MODEL_FAMILY:-qwen}"
LLMLINGUA_PATH="${LLMLINGUA_PATH:-microsoft/llmlingua-2-xlm-roberta-large-meetingbank}"
RUN_TAG="${RUN_TAG:-qwen2.5-3b-tokenskip-paper-repro}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
EVAL_RATIOS="${EVAL_RATIOS:-1.0,0.9,0.8,0.7,0.6,0.5}"
MAX_TRAIN_QUESTIONS="${MAX_TRAIN_QUESTIONS:--1}"

IFS=',' read -r -a GPU_ARR <<< "$GPUS"
NUM_SHARDS="${NUM_SHARDS:-${#GPU_ARR[@]}}"
TRAIN_NPROC="${TRAIN_NPROC:-1}"
GPU0="${GPU_ARR[0]}"

if [ "${#GPU_ARR[@]}" -eq 0 ] || [ -z "${GPU_ARR[0]}" ]; then
    echo "No GPUs were resolved from GPUS='${GPUS}'." >&2
    exit 1
fi

if [ "$NUM_SHARDS" -gt "${#GPU_ARR[@]}" ]; then
    echo "NUM_SHARDS=${NUM_SHARDS} exceeds available GPU slots (${#GPU_ARR[@]}) from GPUS='${GPUS}'." >&2
    echo "Pass a longer GPUS list or lower NUM_SHARDS." >&2
    exit 1
fi

WORK_DIR="${WORK_DIR:-data/sft/tokenskip/repro/${RUN_TAG}}"
ORIGINAL_DIR="${ORIGINAL_DIR:-${WORK_DIR}/original}"
COMPRESSED_DIR="${COMPRESSED_DIR:-${WORK_DIR}/compressed}"
DATA_PATH="${DATA_PATH:-${WORK_DIR}/gsm8k_tokenskip_rawtext.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-results/sft/gsm8k/${RUN_TAG}}"
EVAL_DIR="${EVAL_DIR:-${OUTPUT_DIR}/eval}"

mkdir -p "$ORIGINAL_DIR" "$COMPRESSED_DIR" "$EVAL_DIR"

cat <<EOF
============================================================
TokenSkip Paper Reproduction
  Model: ${MODEL}
  Model family: ${MODEL_FAMILY}
  GPUs: ${GPUS}
  Num shards: ${NUM_SHARDS}
  Train nproc: ${TRAIN_NPROC}
  Eval ratios: ${EVAL_RATIOS}
  Run tag: ${RUN_TAG}
  Work dir: ${WORK_DIR}
  Output dir: ${OUTPUT_DIR}
============================================================
EOF

{
  echo "run_tag=${RUN_TAG}"
  echo "model=${MODEL}"
  echo "model_family=${MODEL_FAMILY}"
  echo "gpus=${GPUS}"
  echo "num_shards=${NUM_SHARDS}"
  echo "train_nproc=${TRAIN_NPROC}"
  echo "eval_ratios=${EVAL_RATIOS}"
  echo "llmlingua_path=${LLMLINGUA_PATH}"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_branch=$(git rev-parse --abbrev-ref HEAD)"
  date -Iseconds | sed 's/^/started_at=/'
} > "${OUTPUT_DIR}/run_metadata.txt"

rm -f "${ORIGINAL_DIR}"/gsm8k_raw_original_shard*.jsonl

echo ""
echo "== Step 1: Collect original raw CoTs =="
for i in $(seq 0 $((NUM_SHARDS-1))); do
    GPU="${GPU_ARR[$i]}"
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON" -m sft.methods.tokenskip.collect \
        --benchmark gsm8k \
        --model "$MODEL" \
        --gpu "$GPU" \
        --prompt-format paper \
        --num-questions "$MAX_TRAIN_QUESTIONS" \
        --num-shards "$NUM_SHARDS" \
        --shard-id "$i" \
        --max-tokens 512 \
        --temperature 0.0 \
        --output "${ORIGINAL_DIR}/gsm8k_raw_original_shard$(printf '%02d' "$i")of$(printf '%02d' "$NUM_SHARDS").jsonl" &
done
wait

cat "${ORIGINAL_DIR}"/gsm8k_raw_original_shard*.jsonl > "${ORIGINAL_DIR}/gsm8k_raw_original_merged.jsonl"

echo ""
echo "== Step 2: Compress correct CoTs with LLMLingua =="
PYTHONPATH=. "$PYTHON" -m sft.methods.tokenskip.compress \
    --input "${ORIGINAL_DIR}/gsm8k_raw_original_merged.jsonl" \
    --output-dir "${COMPRESSED_DIR}/gsm8k_raw" \
    --model-family "$MODEL_FAMILY" \
    --max-cot-tokens 500 \
    --llmlingua-path "$LLMLINGUA_PATH"

echo ""
echo "== Step 3: Build raw-text SFT dataset =="
PYTHONPATH=. "$PYTHON" -m sft.data.prepare \
    --method tokenskip \
    --benchmark gsm8k \
    --original-cot-path "${ORIGINAL_DIR}/gsm8k_raw_original_merged.jsonl" \
    --compressed-cot-dir "${COMPRESSED_DIR}/gsm8k_raw" \
    --response-format tokenskip_paper \
    --model-family "$MODEL_FAMILY" \
    --output "$DATA_PATH"

echo ""
echo "== Step 4: Train =="
if [ "$TRAIN_NPROC" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH=. "$PYTHON" -m torch.distributed.run \
        --nproc_per_node="$TRAIN_NPROC" \
        -m sft.train \
        --config-name gsm8k_tokenskip_rawtext \
        data.path="$DATA_PATH" \
        training.output_dir="$OUTPUT_DIR" \
        training.run_name="$RUN_TAG"
else
    CUDA_VISIBLE_DEVICES="$GPU0" PYTHONPATH=. "$PYTHON" -m sft.train \
        --config-name gsm8k_tokenskip_rawtext \
        data.path="$DATA_PATH" \
        training.output_dir="$OUTPUT_DIR" \
        training.run_name="$RUN_TAG"
fi

echo ""
echo "== Step 5: Evaluate standard TokenSkip ratios =="
for ratio in ${EVAL_RATIOS//,/ }; do
    echo "-- ratio=${ratio}"
    CUDA_VISIBLE_DEVICES="$GPU0" PYTHONPATH=. "$PYTHON" -m sft.eval_raw \
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
echo "TokenSkip paper reproduction complete"
echo "Summary: ${EVAL_DIR}/summary.tsv"
echo "============================================================"
