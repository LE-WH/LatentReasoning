#!/usr/bin/env bash
# Smoke test: TokenSkip MATH pipeline (20 questions, 1 epoch)
# Verifies the full pipeline runs end-to-end.
#
# Usage:
#   # Standard model
#   LLMLINGUA_PATH=microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
#       bash scripts/sft/smoke_test_tokenskip_math.sh
#
#   # Dual-vocab model
#   LLMLINGUA_PATH=microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
#   DUAL_MODEL=checkpoints/dual_qwen_4b_thinking \
#       bash scripts/sft/smoke_test_tokenskip_math.sh

set -euo pipefail

PYTHON="${PYTHON_BIN:-python}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

GPU="${GPU:-0}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
DUAL_MODEL="${DUAL_MODEL:-}"
NUM_QUESTIONS=20
TRAIN_EPOCHS=1
COMPRESS_RATIOS="${COMPRESS_RATIOS:-0.5}"
TRAIN_RATIOS="${TRAIN_RATIOS:-1.0,0.5}"
EVAL_RATIO="${EVAL_RATIO:-0.5}"
: "${LLMLINGUA_PATH:?Set LLMLINGUA_PATH (e.g. microsoft/llmlingua-2-xlm-roberta-large-meetingbank)}"

BENCHMARK="math"

if [ -n "$DUAL_MODEL" ]; then
    TRAIN_MODEL="$DUAL_MODEL"
    TAG="dual"
else
    TRAIN_MODEL="$MODEL"
    TAG="standard"
fi

DATA_DIR="data/sft/tokenskip_smoke_${TAG}"
ORIGINAL="${DATA_DIR}/original.jsonl"
COMPRESSED_DIR="${DATA_DIR}/compressed"
SFT_DATA="${DATA_DIR}/sft_train.jsonl"
OUTPUT_DIR="results/sft_smoke/math_tokenskip_${TAG}"

echo "============================================================"
echo "TokenSkip MATH Smoke Test (${TAG}, ${NUM_QUESTIONS} questions)"
echo "  GPU: ${GPU}   Model: ${TRAIN_MODEL}"
echo "  Compress: ${COMPRESS_RATIOS}  Train: ${TRAIN_RATIOS}"
echo "============================================================"

mkdir -p "${DATA_DIR}" "${COMPRESSED_DIR}"

# Step 1: Collect
echo -e "\n== Step 1: Collect original CoTs =="
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON" -m sft.methods.tokenskip.collect \
    --benchmark "$BENCHMARK" \
    --model "$MODEL" \
    --gpu "$GPU" \
    --prompt-format paper \
    --max-tokens 4096 \
    --max-model-len 8192 \
    --num-questions "$NUM_QUESTIONS" \
    --output "$ORIGINAL"

echo "Collected $(wc -l < "$ORIGINAL") records"

# Step 2: Compress
echo -e "\n== Step 2: Compress with LLMLingua =="
PYTHONPATH=. "$PYTHON" -m sft.methods.tokenskip.compress \
    --input "$ORIGINAL" \
    --output-dir "$COMPRESSED_DIR" \
    --max-cot-tokens 2000 \
    --ratio-pool "$COMPRESS_RATIOS" \
    --llmlingua-path "$LLMLINGUA_PATH"

# Step 3: Prepare SFT data
echo -e "\n== Step 3: Build SFT dataset =="
PYTHONPATH=. "$PYTHON" -m sft.data.prepare \
    --method tokenskip \
    --benchmark "$BENCHMARK" \
    --original-cot-path "$ORIGINAL" \
    --compressed-cot-dir "$COMPRESSED_DIR" \
    --response-format tokenskip_paper \
    --model-family qwen \
    --ratio-pool "$TRAIN_RATIOS" \
    --output "$SFT_DATA"

echo "SFT samples: $(wc -l < "$SFT_DATA")"

# Step 4: Train
echo -e "\n== Step 4: Train (${TAG}, 1 epoch) =="
TRAIN_OVERRIDES=(
    training.num_epochs="$TRAIN_EPOCHS"
    training.save_strategy=no
    training.load_best_model_at_end=false
    training.gradient_accumulation_steps=1
    training.per_device_eval_batch_size=1
    training.eval_accumulation_steps=4
    training.report_to=none
    training.output_dir="$OUTPUT_DIR"
    data.path="$SFT_DATA"
    model.path="$TRAIN_MODEL"
)
if [ -n "$DUAL_MODEL" ]; then
    TRAIN_OVERRIDES+=(dual_vocab.enabled=true)
fi
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON" -m sft.train \
    --config-name math_tokenskip_rawtext \
    "${TRAIN_OVERRIDES[@]}"

# Step 5: Evaluate
echo -e "\n== Step 5: Evaluate (10 test samples) =="
WANDB_ARGS=""
if [ -n "${WANDB_PROJECT:-}" ]; then
    WANDB_ARGS="--wandb-project $WANDB_PROJECT --wandb-run-name smoke-${TAG}-math-ratio${EVAL_RATIO}"
fi
CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON" -m sft.eval_raw \
    --model "${OUTPUT_DIR}/merged" \
    --benchmark "$BENCHMARK" \
    --tokenskip-prompt \
    --compression-ratio "$EVAL_RATIO" \
    --num-samples 10 \
    --output "${OUTPUT_DIR}/eval_smoke.jsonl" \
    $WANDB_ARGS

echo -e "\n============================================================"
echo "Smoke test complete (${TAG})."
echo "Results: ${OUTPUT_DIR}/eval_smoke.jsonl"
echo "============================================================"
