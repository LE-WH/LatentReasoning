#!/bin/bash
set -e

echo "============================================"
echo "  LatentReasoning Environment Setup"
echo "============================================"

# --- Configuration ---
PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-venv}
INSTALL_WEBSHOP=${INSTALL_WEBSHOP:-0}
INSTALL_LEAN=${INSTALL_LEAN:-0}
INSTALL_MEGATRON=${INSTALL_MEGATRON:-0}

# --- Create and activate virtual environment ---
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/6] Creating virtual environment in $VENV_DIR ..."
    $PYTHON -m venv "$VENV_DIR"
else
    echo "[1/6] Virtual environment already exists at $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "[2/6] Installing core packages (ragen + dependencies) ..."
pip install --upgrade pip
pip install -e .

# Optional ragen extras
if [ "$INSTALL_WEBSHOP" -eq 1 ]; then
    echo "       -> Installing webshop extras"
    pip install -e ".[webshop]"
fi
if [ "$INSTALL_LEAN" -eq 1 ]; then
    echo "       -> Installing lean extras"
    pip install -e ".[lean]"
fi

echo "[3/6] Installing verl with vllm support ..."
pip install -e "./verl[vllm]"

# echo "[4/6] Installing flash-attn (building from source for current torch) ..."
# pip install flash-attn --no-build-isolation

echo "[5/6] Pinning compatibility versions ..."
pip install "setuptools>=77.0.3,<80"
pip install "fsspec[http]<=2026.2.0,>=2023.1.0"

# # --- Optional: Megatron + TransformerEngine ---
# if [ "$INSTALL_MEGATRON" -eq 1 ]; then
#     echo "       -> Installing Megatron and TransformerEngine (this takes a while) ..."
#     pip install "onnxscript==0.3.1"
#     NVTE_FRAMEWORK=pytorch pip install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.6
#     pip install --no-deps git+https://github.com/NVIDIA/Megatron-LM.git@core_v0.13.1
#     pip install nvidia-cudnn-cu12==9.10.2.21
# fi

echo "[6/6] Verifying installation ..."
$PYTHON -c "
import torch; print(f'  torch:        {torch.__version__}')
import vllm; print(f'  vllm:         {vllm.__version__}')
import transformers; print(f'  transformers: {transformers.__version__}')
import flash_attn; print(f'  flash_attn:   {flash_attn.__version__}')
import tensordict; print(f'  tensordict:   {tensordict.__version__}')
import ray; print(f'  ray:          {ray.__version__}')
import verl; print(f'  verl:         OK')
print()
if torch.cuda.is_available():
    print(f'  CUDA:         {torch.version.cuda}')
    print(f'  GPU:          {torch.cuda.get_device_name(0)}')
else:
    print('  WARNING: CUDA not available')
"

BROKEN=$(pip check 2>&1)
if [ "$BROKEN" = "No broken requirements found." ]; then
    echo "  pip check:    OK"
else
    echo "  pip check:    WARNINGS"
    echo "$BROKEN"
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Activate with:  source $VENV_DIR/bin/activate"
echo ""
echo "  Options (set before running):"
echo "    INSTALL_WEBSHOP=1  - WebShop environment"
echo "    INSTALL_LEAN=1     - Lean environment"
echo "    INSTALL_MEGATRON=1 - Megatron + TransformerEngine"
echo "============================================"
