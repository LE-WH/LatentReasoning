#!/usr/bin/env bash
set -euo pipefail

# Full SFT pipeline: XML format sampling → exemplars → training → RAGEN eval
# Uses RAGEN-compatible <think>/<answer> XML format throughout.

PYTHON="${PYTHON_BIN:-python}"
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

GPUS="${GPUS:-0,1,2,3}"
NUM_SHARDS=4
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
TRAIN_NPROC="${TRAIN_NPROC:-4}"

echo "============================================================"
echo "SFT Pipeline (XML format)"
echo "  GPUs: ${GPUS}"
echo "  Model: ${MODEL}"
echo "============================================================"

# Step 1: Zero-shot sampling (4 shards in parallel)
echo ""
echo "== Step 1: Zero-shot sampling =="
for i in $(seq 0 $((NUM_SHARDS-1))); do
    GPU=$(echo $GPUS | cut -d',' -f$((i+1)))
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PYTHON -m sft.methods.self_training_concise.sample \
        --benchmark gsm8k \
        --model "$MODEL" \
        --gpu "$GPU" \
        --samples-per-question 16 \
        --num-shards $NUM_SHARDS \
        --shard-id $i \
        --output "data/sft/cot_samples/gsm8k_cot_xml_zero-shot_shard$(printf '%02d' $i)of$(printf '%02d' $NUM_SHARDS).jsonl" &
done
wait
echo "Zero-shot sampling done."

# Step 2: Build few-shot exemplars
echo ""
echo "== Step 2: Build few-shot exemplars =="
ZERO_PATHS=""
for i in $(seq 0 $((NUM_SHARDS-1))); do
    [ -n "$ZERO_PATHS" ] && ZERO_PATHS+=","
    ZERO_PATHS+="data/sft/cot_samples/gsm8k_cot_xml_zero-shot_shard$(printf '%02d' $i)of$(printf '%02d' $NUM_SHARDS).jsonl"
done

# Merge shards for exemplar building (build_exemplars takes single file)
cat data/sft/cot_samples/gsm8k_cot_xml_zero-shot_shard*.jsonl > data/sft/cot_samples/gsm8k_cot_xml_zero-shot_merged.jsonl

PYTHONPATH=. $PYTHON -m sft.methods.self_training_concise.build_exemplars \
    --cot-samples-path data/sft/cot_samples/gsm8k_cot_xml_zero-shot_merged.jsonl \
    --output data/sft/cot_samples/gsm8k_xml_few_shot_exemplars.json \
    --num-exemplars 128 \
    --max-think-tokens 511
echo "Exemplars built."

# Step 3: Few-shot sampling (4 shards in parallel)
echo ""
echo "== Step 3: Few-shot sampling =="
for i in $(seq 0 $((NUM_SHARDS-1))); do
    GPU=$(echo $GPUS | cut -d',' -f$((i+1)))
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. $PYTHON -m sft.methods.self_training_concise.sample \
        --benchmark gsm8k \
        --model "$MODEL" \
        --gpu "$GPU" \
        --samples-per-question 16 \
        --few-shot-path data/sft/cot_samples/gsm8k_xml_few_shot_exemplars.json \
        --num-shards $NUM_SHARDS \
        --shard-id $i \
        --output "data/sft/cot_samples/gsm8k_cot_xml_few-shot_shard$(printf '%02d' $i)of$(printf '%02d' $NUM_SHARDS).jsonl" &
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
        ALL_PATHS+="data/sft/cot_samples/gsm8k_cot_xml_${mode}_shard$(printf '%02d' $i)of$(printf '%02d' $NUM_SHARDS).jsonl"
    done
done

PYTHONPATH=. $PYTHON -m sft.data.prepare \
    --method concise \
    --benchmark gsm8k \
    --cot-samples-path "$ALL_PATHS" \
    --max-think-tokens 511 \
    --output data/sft/gsm8k_concise_xml.jsonl
echo "Training data prepared."

# Step 5: Train
echo ""
echo "== Step 5: Train =="
CUDA_VISIBLE_DEVICES="$GPUS" PYTHONPATH=. $PYTHON -m torch.distributed.run \
    --nproc_per_node="$TRAIN_NPROC" \
    -m sft.train \
    --config-name gsm8k_self_training_concise
echo "Training done."

# Step 6: Evaluate with RAGEN
echo ""
echo "== Step 6: Evaluate with RAGEN =="
GPU0=$(echo $GPUS | cut -d',' -f1)
CUDA_VISIBLE_DEVICES=$GPU0 PYTHONPATH=. $PYTHON -c "
import sys
sys.argv = ['agent_proxy', '--config-name', '_11_gsm8k', 'model_path=results/sft/gsm8k/qwen2.5-3b-self-training-concise', 'system.CUDA_VISIBLE_DEVICES=$GPU0']
import transformers.tokenization_utils_base as _tub
if not hasattr(_tub.PreTrainedTokenizerBase, 'all_special_tokens_extended'):
    _tub.PreTrainedTokenizerBase.all_special_tokens_extended = property(lambda self: list(set(self.all_special_tokens)))
import runpy; runpy.run_module('ragen.llm_agent.agent_proxy', run_name='__main__', alter_sys=True)
"

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
