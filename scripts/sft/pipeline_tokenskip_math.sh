#!/usr/bin/env bash
# TokenSkip pipeline on MATH with Qwen3-4B-Thinking
#
# Unified script for standard and dual-vocab models with configurable
# compression ratios. Runs all 5 steps (collect, compress, prepare, train, eval)
# or skips steps whose outputs already exist.
#
# Usage:
#   # Standard model, default ratios (0.3, 0.5, 0.7)
#   LLMLINGUA_PATH=microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
#       bash scripts/sft/pipeline_tokenskip_math.sh
#
#   # Dual-vocab model
#   LLMLINGUA_PATH=microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
#   DUAL_MODEL=checkpoints/dual_qwen_4b_thinking \
#       bash scripts/sft/pipeline_tokenskip_math.sh
#
#   # Custom ratios
#   COMPRESS_RATIOS=0.5 TRAIN_RATIOS=1.0,0.5 EVAL_RATIOS=1.0,0.5 \
#       bash scripts/sft/pipeline_tokenskip_math.sh
#
# Environment variables:
#   LLMLINGUA_PATH   (required) HF id or local path for LLMLingua-2
#   GPUS             GPU ids, comma-separated           (default: 0,1,2,3)
#   NUM_SHARDS       Shards for parallel collection     (default: 4)
#   MODEL            Base model for CoT collection      (default: Qwen/Qwen3-4B-Thinking-2507)
#   DUAL_MODEL       Path to dual-vocab model           (default: unset = standard)
#   TRAIN_NPROC      GPUs for training                  (default: 4)
#   COMPRESS_RATIOS  LLMLingua compression ratios       (default: 0.1,0.3,0.5,0.7)
#   TRAIN_RATIOS     Ratios in SFT data (includes 1.0)  (default: 1.0,0.1,0.3,0.5,0.7)
#   EVAL_RATIOS      Ratios to evaluate at              (default: 1.0,0.7,0.5,0.3,0.1)
#   MAX_COT_TOKENS   Max reasoning tokens to keep       (default: 2000)
#   NUM_SAMPLES      Responses per question at collect  (default: 8)
#   SAMPLE_TEMP      Sampling temperature at collect    (default: 0.7)
#   COMPRESS_SHARDS  Shards for parallel compression    (default: NUM_SHARDS)
#   WANDB_PROJECT    W&B project name                   (default: unset)
#   SKIP_COLLECT     Set to 1 to skip step 1            (default: unset)
#   SKIP_COMPRESS    Set to 1 to skip step 2            (default: unset)
#   SKIP_PREPARE     Set to 1 to skip step 3            (default: unset)
#   SKIP_TRAIN       Set to 1 to skip step 4            (default: unset)
#   SKIP_QUALITY     Set to 1 to skip the quality-filter step (2.5)
#   QUALITY_FILTER_TOP   Fraction in (0,1] for top-X%-per-question filter (default: unset = no filter)
#   QUALITY_SCORE_KEY    Score column to rank by: logprob_post_cot (default), logprob_boxed, logprob_answer

set -euo pipefail

PYTHON="${PYTHON_BIN:-python}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

# --- Configuration ---
GPUS="${GPUS:-0,1,2,3}"
NUM_SHARDS="${NUM_SHARDS:-4}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
DUAL_MODEL="${DUAL_MODEL:-}"
TRAIN_NPROC="${TRAIN_NPROC:-4}"
MAX_COT_TOKENS="${MAX_COT_TOKENS:-2000}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
SAMPLE_TEMP="${SAMPLE_TEMP:-0.7}"
COMPRESS_RATIOS="${COMPRESS_RATIOS:-0.1,0.3,0.5,0.7}"
COMPRESS_SHARDS="${COMPRESS_SHARDS:-$NUM_SHARDS}"
TRAIN_RATIOS="${TRAIN_RATIOS:-1.0,0.1,0.3,0.5,0.7}"
EVAL_RATIOS="${EVAL_RATIOS:-1.0,0.7,0.5,0.3,0.1}"
BENCHMARK="math"

# --- Derived paths ---
# Build TAG: multirate when >1 train ratio, then -dual suffix if dual model
NUM_TRAIN_RATIOS=$(echo "$TRAIN_RATIOS" | tr ',' '\n' | wc -l)
if [ "$NUM_TRAIN_RATIOS" -gt 1 ]; then
    TAG="multirate"
else
    TAG="rawtext"
fi
if [ -n "$DUAL_MODEL" ]; then
    TRAIN_MODEL="$DUAL_MODEL"
    TAG="${TAG}-dual"
else
    TRAIN_MODEL="$MODEL"
fi

DATA_DIR="data/sft/tokenskip"
ORIGINAL_DIR="${DATA_DIR}/original/${BENCHMARK}"
COMPRESSED_DIR="${DATA_DIR}/compressed/${BENCHMARK}_raw"
SCORES_DIR="${DATA_DIR}/scores/${BENCHMARK}_raw"
MERGED_JSONL="${ORIGINAL_DIR}/${BENCHMARK}_raw_original_merged.jsonl"

# Optional per-question quality filter. Set QUALITY_FILTER_TOP to a fraction in
# (0, 1] (e.g. 0.25 = top 25%) to enable. The score key is by default
# ``logprob_post_cot`` (sum log-prob of the response continuation after the
# compressed CoT, computed under the SFT base model).
QUALITY_FILTER_TOP="${QUALITY_FILTER_TOP:-}"
QUALITY_SCORE_KEY="${QUALITY_SCORE_KEY:-logprob_post_cot}"

if [ -n "$QUALITY_FILTER_TOP" ]; then
    QF_TAG="qf$(printf '%g' "$QUALITY_FILTER_TOP")"
    case "$QUALITY_SCORE_KEY" in
        logprob_answer)   QF_KEY_SHORT="ans" ;;
        logprob_boxed)    QF_KEY_SHORT="boxed" ;;
        logprob_post_cot) QF_KEY_SHORT="pc" ;;
        *) QF_KEY_SHORT="$QUALITY_SCORE_KEY" ;;
    esac
    QF_FULL_TAG="${QF_TAG}-${QF_KEY_SHORT}"
    FILTERED_DIR="${DATA_DIR}/compressed/${BENCHMARK}_raw_${QF_TAG}_${QUALITY_SCORE_KEY}"
    PREP_COMPRESSED_DIR="$FILTERED_DIR"
    TAG="${TAG}-${QF_FULL_TAG}"
    SFT_DATA="data/sft/${BENCHMARK}_tokenskip_rawtext_${QF_FULL_TAG}.jsonl"
else
    PREP_COMPRESSED_DIR="$COMPRESSED_DIR"
    SFT_DATA="data/sft/${BENCHMARK}_tokenskip_rawtext.jsonl"
fi

OUTPUT_DIR="results/sft/${BENCHMARK}/qwen3-4b-thinking-tokenskip-${TAG}"

echo "============================================================"
echo "TokenSkip MATH Pipeline"
echo "  Mode:            ${TAG}"
echo "  GPUs:            ${GPUS}"
echo "  Base model:      ${MODEL}"
if [ -n "$DUAL_MODEL" ]; then
echo "  Dual model:      ${DUAL_MODEL}"
fi
echo "  Collect n/T:     ${NUM_SAMPLES} @ T=${SAMPLE_TEMP}"
echo "  Compress shards: ${COMPRESS_SHARDS}"
echo "  Compress ratios: ${COMPRESS_RATIOS}"
echo "  Train ratios:    ${TRAIN_RATIOS}"
echo "  Eval ratios:     ${EVAL_RATIOS}"
if [ -n "$QUALITY_FILTER_TOP" ]; then
echo "  Quality filter:  top ${QUALITY_FILTER_TOP} per question (key=${QUALITY_SCORE_KEY})"
fi
echo "  Output:          ${OUTPUT_DIR}"
echo "============================================================"

mkdir -p "${ORIGINAL_DIR}" "${COMPRESSED_DIR}"
if [ -n "$QUALITY_FILTER_TOP" ]; then
    mkdir -p "${SCORES_DIR}" "${FILTERED_DIR}"
fi

# ------------------------------------------------------------------
# Step 1: Collect original CoTs
# ------------------------------------------------------------------
if [ "${SKIP_COLLECT:-}" = "1" ] || [ -f "$MERGED_JSONL" ]; then
    echo -e "\n== Step 1: SKIP (${MERGED_JSONL} exists) =="
else
    : "${LLMLINGUA_PATH:?Set LLMLINGUA_PATH (e.g. microsoft/llmlingua-2-xlm-roberta-large-meetingbank)}"
    echo -e "\n== Step 1: Collect original raw CoTs on MATH =="
    for i in $(seq 0 $((NUM_SHARDS-1))); do
        GPU=$(echo "$GPUS" | cut -d',' -f$((i+1)))
        CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON" -m sft.methods.tokenskip.collect \
            --benchmark "$BENCHMARK" \
            --model "$MODEL" \
            --gpu "$GPU" \
            --prompt-format paper \
            --max-tokens 4096 \
            --max-model-len 8192 \
            --num-samples "$NUM_SAMPLES" \
            --temperature "$SAMPLE_TEMP" \
            --num-shards "$NUM_SHARDS" \
            --shard-id "$i" \
            --output "${ORIGINAL_DIR}/${BENCHMARK}_raw_original_shard$(printf '%02d' "$i")of$(printf '%02d' "$NUM_SHARDS").jsonl" &
    done
    wait
    cat "${ORIGINAL_DIR}/${BENCHMARK}_raw_original_shard"*.jsonl > "$MERGED_JSONL"
fi
echo "Collected CoTs: $(wc -l < "$MERGED_JSONL") records"

# ------------------------------------------------------------------
# Step 2: Compress with LLMLingua-2
# ------------------------------------------------------------------
if [ "${SKIP_COMPRESS:-}" = "1" ]; then
    echo -e "\n== Step 2: SKIP =="
else
    : "${LLMLINGUA_PATH:?Set LLMLINGUA_PATH (e.g. microsoft/llmlingua-2-xlm-roberta-large-meetingbank)}"
    if [ "$COMPRESS_SHARDS" -gt 1 ]; then
        echo -e "\n== Step 2: Compress CoTs (ratios: ${COMPRESS_RATIOS}, sharded×${COMPRESS_SHARDS}) =="
        for i in $(seq 0 $((COMPRESS_SHARDS-1))); do
            GPU=$(echo "$GPUS" | cut -d',' -f$((i+1)))
            CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON" -m sft.methods.tokenskip.compress \
                --input "$MERGED_JSONL" \
                --output-dir "$COMPRESSED_DIR" \
                --max-cot-tokens "$MAX_COT_TOKENS" \
                --ratio-pool "$COMPRESS_RATIOS" \
                --llmlingua-path "$LLMLINGUA_PATH" \
                --num-shards "$COMPRESS_SHARDS" \
                --shard-id "$i" &
        done
        wait
        # Merge per-ratio shard files into the canonical filename select.py expects.
        for ratio in $(echo "$COMPRESS_RATIOS" | tr ',' ' '); do
            cat "${COMPRESSED_DIR}/compressed_ratio_${ratio}_shard"*.jsonl \
                > "${COMPRESSED_DIR}/compressed_ratio_${ratio}.jsonl"
            rm "${COMPRESSED_DIR}/compressed_ratio_${ratio}_shard"*.jsonl
        done
    else
        echo -e "\n== Step 2: Compress CoTs (ratios: ${COMPRESS_RATIOS}) =="
        PYTHONPATH=. "$PYTHON" -m sft.methods.tokenskip.compress \
            --input "$MERGED_JSONL" \
            --output-dir "$COMPRESSED_DIR" \
            --max-cot-tokens "$MAX_COT_TOKENS" \
            --ratio-pool "$COMPRESS_RATIOS" \
            --llmlingua-path "$LLMLINGUA_PATH"
    fi
fi

# ------------------------------------------------------------------
# Step 2.5: (optional) Quality-score and filter per-question top-X%
# ------------------------------------------------------------------
if [ -n "$QUALITY_FILTER_TOP" ] && [ "${SKIP_QUALITY:-}" != "1" ]; then
    echo -e "\n== Step 2.5: Score CoT quality (ratios from compressed_dir) =="
    SCORE_RATIOS=$(ls "$COMPRESSED_DIR" \
        | sed -nE 's/^compressed_ratio_([0-9.]+)\.jsonl$/\1/p' | sort -u | tr '\n' ' ')
    for ratio in $SCORE_RATIOS; do
        SCORE_OUT="${SCORES_DIR}/scores_ratio_${ratio}.jsonl"
        if [ -f "$SCORE_OUT" ]; then
            echo "  ratio=${ratio}: skip (${SCORE_OUT} exists)"
            continue
        fi
        echo "  ratio=${ratio}: scoring (sharded×${NUM_SHARDS})"
        for i in $(seq 0 $((NUM_SHARDS-1))); do
            GPU=$(echo "$GPUS" | cut -d',' -f$((i+1)))
            CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON" \
                -m sft.methods.tokenskip.quality_score \
                --input "${COMPRESSED_DIR}/compressed_ratio_${ratio}.jsonl" \
                --output "${SCORES_DIR}/scores_ratio_${ratio}_shard$(printf '%02d' "$i")of$(printf '%02d' "$NUM_SHARDS").jsonl" \
                --model "$MODEL" \
                --gpu "$GPU" \
                --num-shards "$NUM_SHARDS" --shard-id "$i" \
                --batch-size 8 &
        done
        wait
        cat "${SCORES_DIR}/scores_ratio_${ratio}_shard"*.jsonl > "$SCORE_OUT"
        rm "${SCORES_DIR}/scores_ratio_${ratio}_shard"*.jsonl
    done

    echo -e "\n== Step 2.6: Filter top ${QUALITY_FILTER_TOP} per question =="
    for ratio in $SCORE_RATIOS; do
        FILTERED_OUT="${FILTERED_DIR}/compressed_ratio_${ratio}.jsonl"
        if [ -f "$FILTERED_OUT" ]; then
            echo "  ratio=${ratio}: skip (${FILTERED_OUT} exists)"
            continue
        fi
        PYTHONPATH=. "$PYTHON" -m sft.methods.tokenskip.quality_filter \
            --scores "${SCORES_DIR}/scores_ratio_${ratio}.jsonl" \
            --compressed "${COMPRESSED_DIR}/compressed_ratio_${ratio}.jsonl" \
            --output "$FILTERED_OUT" \
            --top "$QUALITY_FILTER_TOP" \
            --score-key "$QUALITY_SCORE_KEY"
    done
fi

# ------------------------------------------------------------------
# Step 3: Build SFT dataset
# ------------------------------------------------------------------
if [ "${SKIP_PREPARE:-}" = "1" ]; then
    echo -e "\n== Step 3: SKIP =="
else
    echo -e "\n== Step 3: Build SFT dataset (ratios: ${TRAIN_RATIOS}) =="
    PYTHONPATH=. "$PYTHON" -m sft.data.prepare \
        --method tokenskip \
        --benchmark "$BENCHMARK" \
        --original-cot-path "$MERGED_JSONL" \
        --compressed-cot-dir "$PREP_COMPRESSED_DIR" \
        --response-format tokenskip_paper \
        --model-family qwen \
        --ratio-pool "$TRAIN_RATIOS" \
        --output "$SFT_DATA"
fi
echo "SFT samples: $(wc -l < "$SFT_DATA")"

# ------------------------------------------------------------------
# Step 4: Train
# ------------------------------------------------------------------
if [ "${SKIP_TRAIN:-}" = "1" ]; then
    echo -e "\n== Step 4: SKIP =="
else
    echo -e "\n== Step 4: Train (${TAG}) =="
    TRAIN_OVERRIDES=(
        data.path="$SFT_DATA"
        training.output_dir="$OUTPUT_DIR"
        training.run_name="sft-math-tokenskip-${TAG}"
        model.path="$TRAIN_MODEL"
    )
    if [ -n "$DUAL_MODEL" ]; then
        TRAIN_OVERRIDES+=(dual_vocab.enabled=true)
    fi
    CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH=. "$PYTHON" -m torch.distributed.run \
        --nproc_per_node="$TRAIN_NPROC" \
        -m sft.train \
        --config-name math_tokenskip_rawtext \
        "${TRAIN_OVERRIDES[@]}"
fi

# ------------------------------------------------------------------
# Step 5: Evaluate at all ratios
# ------------------------------------------------------------------
echo -e "\n== Step 5: Evaluate on MATH test (ratios: ${EVAL_RATIOS}) =="
GPU0=$(echo "$GPUS" | cut -d',' -f1)

for ratio in $(echo "$EVAL_RATIOS" | tr ',' ' '); do
    echo "--- ratio=${ratio} ---"
    WANDB_ARGS=""
    if [ -n "${WANDB_PROJECT:-}" ]; then
        WANDB_ARGS="--wandb-project $WANDB_PROJECT --wandb-run-name ${TAG}-math-eval-ratio${ratio}"
    fi
    CUDA_VISIBLE_DEVICES="$GPU0" PYTHONPATH=. "$PYTHON" -m sft.eval_raw \
        --model "${OUTPUT_DIR}/merged" \
        --benchmark "$BENCHMARK" \
        --tokenskip-prompt \
        --compression-ratio "$ratio" \
        --output "${OUTPUT_DIR}/eval/${BENCHMARK}_eval_ratio_${ratio}.jsonl" \
        $WANDB_ARGS
done

echo ""
echo "============================================================"
echo "TokenSkip MATH pipeline complete (${TAG})"
echo "  Model:  ${OUTPUT_DIR}/merged"
echo "  Eval:   ${OUTPUT_DIR}/eval/"
echo "============================================================"
