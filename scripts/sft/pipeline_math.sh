#!/usr/bin/env bash
set -euo pipefail

# SFT Pipeline for MATH benchmark (hendrycks-MATH)
# Zero-shot CoT sampling → exemplars → few-shot sampling → train → eval

PYTHON="${PYTHON_BIN:-python}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

GPUS="${GPUS:-0,1,2,3}"
NUM_SHARDS=4
MODEL="${MODEL:-/workspace/LatentReasoning/checkpoints/dual_qwen_4b_thinking}"
TRAIN_NPROC="${TRAIN_NPROC:-4}"
MAX_THINK_TOKENS="${MAX_THINK_TOKENS:-1023}"
NUM_QUESTIONS="${NUM_QUESTIONS:--1}"  # set to small number (e.g. 50) for testing

echo "============================================================"
echo "SFT Pipeline — MATH (XML format)"
echo "  GPUs: ${GPUS}"
echo "  Model: ${MODEL}"
echo "  Max think tokens: ${MAX_THINK_TOKENS}"
echo "============================================================"

# Step 1: Zero-shot sampling (4 shards in parallel)
echo ""
echo "== Step 1: Zero-shot sampling =="
for i in $(seq 0 $((NUM_SHARDS-1))); do
    GPU=$(echo $GPUS | cut -d',' -f$((i+1)))
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PYTHON -m sft.methods.self_training_concise.sample \
        --benchmark math \
        --model "$MODEL" \
        --gpu "$GPU" \
        --samples-per-question 16 \
        --max-tokens 1024 \
        --max-model-len 4096 \
        --num-shards $NUM_SHARDS \
        --shard-id $i \
        --num-questions $NUM_QUESTIONS \
        --output "data/sft/cot_samples/math_cot_xml_zero-shot_shard$(printf '%02d' $i)of$(printf '%02d' $NUM_SHARDS).jsonl" &
done
wait
echo "Zero-shot sampling done."

# Step 2: Build few-shot exemplars
echo ""
echo "== Step 2: Build few-shot exemplars =="
cat data/sft/cot_samples/math_cot_xml_zero-shot_shard*.jsonl > data/sft/cot_samples/math_cot_xml_zero-shot_merged.jsonl

PYTHONPATH=. $PYTHON -m sft.methods.self_training_concise.build_exemplars \
    --cot-samples-path data/sft/cot_samples/math_cot_xml_zero-shot_merged.jsonl \
    --output data/sft/cot_samples/math_xml_few_shot_exemplars.json \
    --num-exemplars 128 \
    --max-think-tokens $MAX_THINK_TOKENS
echo "Exemplars built."

# Step 3: Few-shot sampling (4 shards in parallel)
echo ""
echo "== Step 3: Few-shot sampling =="
for i in $(seq 0 $((NUM_SHARDS-1))); do
    GPU=$(echo $GPUS | cut -d',' -f$((i+1)))
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PYTHON -m sft.methods.self_training_concise.sample \
        --benchmark math \
        --model "$MODEL" \
        --gpu "$GPU" \
        --samples-per-question 16 \
        --max-tokens 1024 \
        --max-model-len 8192 \
        --few-shot-path data/sft/cot_samples/math_xml_few_shot_exemplars.json \
        --num-shards $NUM_SHARDS \
        --shard-id $i \
        --num-questions $NUM_QUESTIONS \
        --output "data/sft/cot_samples/math_cot_xml_few-shot_shard$(printf '%02d' $i)of$(printf '%02d' $NUM_SHARDS).jsonl" &
done
wait
echo "Few-shot sampling done."

# Step 4: Prepare training data
echo ""
echo "== Step 4: Prepare training data =="
ALL_PATHS=""
for mode in zero-shot few-shot; do
    for i in $(seq 0 $((NUM_SHARDS-1))); do
        [ -n "$ALL_PATHS" ] && ALL_PATHS+=","
        ALL_PATHS+="data/sft/cot_samples/math_cot_xml_${mode}_shard$(printf '%02d' $i)of$(printf '%02d' $NUM_SHARDS).jsonl"
    done
done

PYTHONPATH=. $PYTHON -m sft.data.prepare \
    --method concise \
    --benchmark math \
    --cot-samples-path "$ALL_PATHS" \
    --max-think-tokens $MAX_THINK_TOKENS \
    --output data/sft/math_concise_xml.jsonl
echo "Training data prepared."

# Step 5: Train
echo ""
echo "== Step 5: Train =="
CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH=. $PYTHON -m torch.distributed.run \
    --nproc_per_node="$TRAIN_NPROC" \
    -m sft.train \
    --config-name math_self_training_concise
echo "Training done."

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "  Model saved to: results/sft/math/dual-qwen-4b-thinking-concise"
echo "============================================================"
