#!/bin/bash
# Eval driver for SoT-distilled SFT comparison.
#
# Usage:
#   bash scripts/sft/run_sot_distill_eval.sh <gpu_id> <model_path> <run_name> <benchmark> <output_dir>
#
# benchmark: math | gsm8k
# Always uses --sot-prompt --sot-system-only (SoT system prompt, no exemplars,
# matches the SFT training prompt format).
set -euo pipefail

GPU=${1:?"GPU id"}
MODEL=${2:?"model path"}
RUN_NAME=${3:?"run name"}
BENCH=${4:?"benchmark"}
OUT_DIR=${5:?"output dir"}

mkdir -p "$OUT_DIR"

cd /workspace/LatentReasoning

CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. /workspace/LatentReasoning/venv/bin/python -u -m sft.eval_raw \
  --model "$MODEL" \
  --benchmark "$BENCH" \
  --sot-prompt --sot-system-only \
  --max-tokens 16384 --max-model-len 17408 \
  --output "$OUT_DIR/${RUN_NAME}_${BENCH}.jsonl" \
  > "$OUT_DIR/${RUN_NAME}_${BENCH}.log" 2>&1
