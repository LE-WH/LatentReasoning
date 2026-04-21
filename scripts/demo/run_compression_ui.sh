#!/usr/bin/env bash
# Launch the LLMLingua-2 CoT compression demo.
#
# Usage:
#   bash scripts/demo/run_compression_ui.sh                 # local, port 7860
#   PORT=8000 bash scripts/demo/run_compression_ui.sh       # custom port
#   SHARE=1 bash scripts/demo/run_compression_ui.sh         # public share link
#   LLMLINGUA_PATH=/local/path bash scripts/demo/run_compression_ui.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

if [ -f "venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
fi

LLMLINGUA_PATH="${LLMLINGUA_PATH:-microsoft/llmlingua-2-xlm-roberta-large-meetingbank}"
LM_MODEL_PATH="${LM_MODEL_PATH:-Qwen/Qwen2.5-0.5B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"

ARGS=(
    --model "$LLMLINGUA_PATH"
    --lm-model "$LM_MODEL_PATH"
    --host "$HOST"
    --port "$PORT"
    --preload
)
if [ "${SHARE:-}" = "1" ]; then
    ARGS+=(--share)
fi

PYTHONPATH=. exec python -m tools.compression_demo.app "${ARGS[@]}"
