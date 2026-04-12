#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON_BIN:-python}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

GPUS="${GPUS:-0,1,2,3}"
NUM_SHARDS="${NUM_SHARDS:-4}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
TRAIN_NPROC="${TRAIN_NPROC:-4}"
EVAL_RATIO="${EVAL_RATIO:-1.0}"
: "${LLMLINGUA_PATH:?Set LLMLINGUA_PATH to the LLMLingua-2 checkpoint/path}"

echo "============================================================"
echo "TokenSkip Raw-Text Pipeline"
echo "  GPUs: ${GPUS}"
echo "  Model: ${MODEL}"
echo "  Eval ratio: ${EVAL_RATIO}"
echo "============================================================"

mkdir -p data/sft/tokenskip/original data/sft/tokenskip/compressed

echo ""
echo "== Step 1: Collect original raw CoTs =="
for i in $(seq 0 $((NUM_SHARDS-1))); do
    GPU=$(echo "$GPUS" | cut -d',' -f$((i+1)))
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH=. "$PYTHON" -m sft.methods.tokenskip.collect \
        --benchmark gsm8k \
        --model "$MODEL" \
        --gpu "$GPU" \
        --prompt-format paper \
        --num-shards "$NUM_SHARDS" \
        --shard-id "$i" \
        --output "data/sft/tokenskip/original/gsm8k_raw_original_shard$(printf '%02d' "$i")of$(printf '%02d' "$NUM_SHARDS").jsonl" &
done
wait

cat data/sft/tokenskip/original/gsm8k_raw_original_shard*.jsonl > data/sft/tokenskip/original/gsm8k_raw_original_merged.jsonl

echo ""
echo "== Step 2: Compress correct CoTs with LLMLingua =="
PYTHONPATH=. "$PYTHON" -m sft.methods.tokenskip.compress \
    --input data/sft/tokenskip/original/gsm8k_raw_original_merged.jsonl \
    --output-dir data/sft/tokenskip/compressed/gsm8k_raw \
    --llmlingua-path "$LLMLINGUA_PATH"

echo ""
echo "== Step 3: Build raw-text SFT dataset =="
PYTHONPATH=. "$PYTHON" -m sft.data.prepare \
    --method tokenskip \
    --benchmark gsm8k \
    --original-cot-path data/sft/tokenskip/original/gsm8k_raw_original_merged.jsonl \
    --compressed-cot-dir data/sft/tokenskip/compressed/gsm8k_raw \
    --response-format tokenskip_paper \
    --model-family qwen \
    --output data/sft/gsm8k_tokenskip_rawtext.jsonl

echo ""
echo "== Step 4: Train =="
CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH=. "$PYTHON" -m torch.distributed.run \
    --nproc_per_node="$TRAIN_NPROC" \
    -m sft.train \
    --config-name gsm8k_tokenskip_rawtext

echo ""
echo "== Step 5: Evaluate raw text =="
GPU0=$(echo "$GPUS" | cut -d',' -f1)
CUDA_VISIBLE_DEVICES="$GPU0" PYTHONPATH=. "$PYTHON" -m sft.eval_raw \
    --model results/sft/gsm8k/qwen2.5-3b-tokenskip-rawtext/merged \
    --tokenskip-prompt \
    --compression-ratio "$EVAL_RATIO" \
    --output results/sft/gsm8k/qwen2.5-3b-tokenskip-rawtext/eval/gsm8k_eval_ratio_${EVAL_RATIO}.jsonl

echo ""
echo "============================================================"
echo "TokenSkip raw-text pipeline complete"
echo "============================================================"
