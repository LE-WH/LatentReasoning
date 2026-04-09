#!/usr/bin/env bash
set -euo pipefail

# Raw text SFT pipeline: paper-style sampling → training → raw text eval
#
# Self-Training Elicits Concise Reasoning (arxiv 2502.20122)
# Uses the paper's original raw text format (no XML tags).
#
# Pipeline matches the paper exactly:
#   1. Zero-shot sampling
#   2. Build model-generated exemplars (128 shortest correct)
#   3. Few-shot sampling (with model-generated exemplars)
#   4. Prepare augmented training data
#   5. Train
#   6. Evaluate

PYTHON="${PYTHON_BIN:-python}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

ZERO_GPUS="${ZERO_GPUS:-0,1}"
FEW_GPUS="${FEW_GPUS:-2,3}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
NUM_SHARDS="${NUM_SHARDS:-2}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
TRAIN_NPROC="${TRAIN_NPROC:-4}"

ZERO_SHARD0="data/sft/cot_samples/gsm8k_cot_zero-shot_paper_shard00of02.jsonl"
ZERO_SHARD1="data/sft/cot_samples/gsm8k_cot_zero-shot_paper_shard01of02.jsonl"
ZERO_MERGED="data/sft/cot_samples/gsm8k_cot_zero-shot_paper_merged.jsonl"
EXEMPLARS="data/sft/cot_samples/gsm8k_paper_few_shot_exemplars.json"
FEW_SHARD0="data/sft/cot_samples/gsm8k_cot_few-shot_paper_shard00of02.jsonl"
FEW_SHARD1="data/sft/cot_samples/gsm8k_cot_few-shot_paper_shard01of02.jsonl"
AUG_DATA="data/sft/gsm8k_concise_paper_augmented.jsonl"
OUTPUT_DIR="results/sft/gsm8k/qwen2.5-3b-self-training-concise-rawtext"

echo "============================================================"
echo "Raw Text SFT Pipeline (paper-aligned)"
echo "============================================================"

# Step 1: Zero-shot sampling (2 shards in parallel)
echo ""
echo "== Step 1: Zero-shot sampling =="
CUDA_VISIBLE_DEVICES="${ZERO_GPUS%%,*}" PYTHONPATH=. "$PYTHON" -m sft.methods.self_training_concise.sample \
  --benchmark gsm8k \
  --model "$MODEL" \
  --gpu "${ZERO_GPUS%%,*}" \
  --samples-per-question 16 \
  --prompt-format paper \
  --num-shards "$NUM_SHARDS" \
  --shard-id 0 \
  --output "$ZERO_SHARD0" &
ZERO_PID0=$!

CUDA_VISIBLE_DEVICES="${ZERO_GPUS##*,}" PYTHONPATH=. "$PYTHON" -m sft.methods.self_training_concise.sample \
  --benchmark gsm8k \
  --model "$MODEL" \
  --gpu "${ZERO_GPUS##*,}" \
  --samples-per-question 16 \
  --prompt-format paper \
  --num-shards "$NUM_SHARDS" \
  --shard-id 1 \
  --output "$ZERO_SHARD1" &
ZERO_PID1=$!

wait "$ZERO_PID0" "$ZERO_PID1"
echo "Zero-shot sampling done."

# Step 2: Build model-generated exemplars (paper: 128 shortest correct)
echo ""
echo "== Step 2: Build few-shot exemplars (128 shortest correct) =="
cat "$ZERO_SHARD0" "$ZERO_SHARD1" > "$ZERO_MERGED"
PYTHONPATH=. "$PYTHON" -m sft.methods.self_training_concise.build_exemplars \
  --cot-samples-path "$ZERO_MERGED" \
  --output "$EXEMPLARS" \
  --num-exemplars 128 \
  --max-think-tokens 511
echo "Exemplars built."

# Step 3: Few-shot sampling with model-generated exemplars (2 shards in parallel)
echo ""
echo "== Step 3: Few-shot sampling (with model-generated exemplars) =="
CUDA_VISIBLE_DEVICES="${FEW_GPUS%%,*}" PYTHONPATH=. "$PYTHON" -m sft.methods.self_training_concise.sample \
  --benchmark gsm8k \
  --model "$MODEL" \
  --gpu "${FEW_GPUS%%,*}" \
  --samples-per-question 16 \
  --prompt-format paper \
  --few-shot-path "$EXEMPLARS" \
  --num-shards "$NUM_SHARDS" \
  --shard-id 0 \
  --output "$FEW_SHARD0" &
FEW_PID0=$!

CUDA_VISIBLE_DEVICES="${FEW_GPUS##*,}" PYTHONPATH=. "$PYTHON" -m sft.methods.self_training_concise.sample \
  --benchmark gsm8k \
  --model "$MODEL" \
  --gpu "${FEW_GPUS##*,}" \
  --samples-per-question 16 \
  --prompt-format paper \
  --few-shot-path "$EXEMPLARS" \
  --num-shards "$NUM_SHARDS" \
  --shard-id 1 \
  --output "$FEW_SHARD1" &
FEW_PID1=$!

wait "$FEW_PID0" "$FEW_PID1"
echo "Few-shot sampling done."

# Step 4: Prepare augmented training data
echo ""
echo "== Step 4: Prepare augmented training data =="
PYTHONPATH=. "$PYTHON" -m sft.data.prepare \
  --method concise \
  --benchmark gsm8k \
  --cot-samples-path "${ZERO_SHARD0},${ZERO_SHARD1},${FEW_SHARD0},${FEW_SHARD1}" \
  --max-think-tokens 511 \
  --fmt paper \
  --output "$AUG_DATA"

# Step 5: Train
echo ""
echo "== Step 5: Train =="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" PYTHONPATH=. "$PYTHON" -m torch.distributed.run \
  --nproc_per_node="$TRAIN_NPROC" \
  -m sft.train \
  --config-name gsm8k_self_training_concise_rawtext

# Step 6: Evaluate with raw text eval
echo ""
echo "== Step 6: Evaluate (raw text) =="
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS%%,*}" PYTHONPATH=. "$PYTHON" -m sft.eval_raw \
  --model "$OUTPUT_DIR" \
  --output "$OUTPUT_DIR/eval/gsm8k_rawtext_eval.jsonl"

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
