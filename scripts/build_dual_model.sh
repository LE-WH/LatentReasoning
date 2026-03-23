#!/bin/bash
# build_dual_model.sh
# ====================
# Full pipeline to create a dual-vocab model from a base model.
#
# Step 1: Build the dual tokenizer (add latent vocab tokens, save meta)
# Step 2: Expand the model embeddings to dual vocab size
# Step 3: (Optional) Verify the result
#
# Usage:
#   bash scripts/build_dual_model.sh \
#       --base_model Qwen/Qwen2.5-3B-Instruct \
#       --out_dir ./checkpoints/dual_qwen_3b
#
# Optional flags:
#   --mode        full_copy | subset | custom   (default: full_copy)
#   --think_missing  add | error                (default: error; use "add" if model lacks <think>)
#   --no_verify   skip verification step

set -euo pipefail

BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
OUT_DIR="./checkpoints/dual_model"
MODE="full_copy"
THINK_MISSING="error"
SKIP_VERIFY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base_model)  BASE_MODEL="$2"; shift 2;;
        --out_dir)     OUT_DIR="$2";    shift 2;;
        --mode)        MODE="$2";       shift 2;;
        --think_missing) THINK_MISSING="$2"; shift 2;;
        --no_verify)   SKIP_VERIFY=true; shift;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

TOK_DIR="${OUT_DIR}_tokenizer"

echo "=============================================="
echo "  LatentReasoning Dual Model Build Pipeline"
echo "=============================================="
echo "  base_model   : $BASE_MODEL"
echo "  tok_dir      : $TOK_DIR"
echo "  model_dir    : $OUT_DIR"
echo "  mode         : $MODE"
echo "  think_missing: $THINK_MISSING"
echo ""

# Step 1: Build dual tokenizer
echo "[1/3] Building dual-vocab tokenizer …"
python scripts/build_dual_vocab.py \
    --base_model "$BASE_MODEL" \
    --out_dir "$TOK_DIR" \
    --mode "$MODE" \
    --think_missing "$THINK_MISSING"

echo ""
echo "[2/3] Expanding model to dual vocab …"
python scripts/expand_model_to_dual_vocab.py \
    --base_model "$BASE_MODEL" \
    --dual_tok_dir "$TOK_DIR" \
    --out_dir "$OUT_DIR"

if [ "$SKIP_VERIFY" = false ]; then
    echo ""
    echo "[3/3] Verifying dual-vocab model …"
    python scripts/verify_dual_model.py \
        --model_dir "$OUT_DIR" \
        --check_latent
fi

echo ""
echo "Done! Dual-vocab model saved to: $OUT_DIR"
echo ""
echo "To train with this model, set in your config:"
echo "  model_path: $OUT_DIR"
echo ""
echo "To enable latent-only training:"
echo "  dual_vocab.latent_only: true"
