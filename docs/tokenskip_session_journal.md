# TokenSkip + Dual-Vocab Session Journal

Record of the TokenSkip implementation and dual-vocab integration work.

---

## What was done

### 1. TokenSkip pipeline for MATH (from scratch)

The codebase originally only supported GSM8K with Qwen2.5-3B-Instruct. We adapted it to support **MATH** (nlile/hendrycks-MATH-benchmark) with **Qwen3-4B-Thinking-2507**.

**Files created/modified:**

- `sft/methods/tokenskip/collect.py` — Added MATH benchmark support via `_BENCHMARK_REGISTRY`, added `parse_think_output()` to handle `<think>...</think>` blocks from Qwen3-Thinking, appends `<think>\n` to prompt to trigger thinking mode.
- `sft/methods/tokenskip/utils.py` — Shared helpers for ratio conditioning. Moved here from root-level `token_skip_utils.py`. Contains `append_inline_ratio_condition()`, `build_inline_ratio_suffix()`, `get_ratio_boundary_token()`.
- `sft/methods/tokenskip/prompts.py` — Uses `<|im_end|>` as boundary token for Qwen (not `<|eot_id|>` which was Llama-only and tokenized to garbage in Qwen3).
- `sft/data/base.py` — Added `tokenskip_paper` response format: includes `</think>` before `\boxed{}` so the model learns the full `<think>reasoning</think>\nfinal answer` structure.
- `sft/eval_raw.py` — Added MATH eval, `--benchmark` arg, wandb logging (accuracy, thinking length histogram, sample CoT table), dual-vocab detection and latent ID decoding, per-sample latent/visible token counts and raw token IDs in output.
- `sft/train.py` — Added `remap_think_to_latent()` for dual-vocab token remapping during training, `per_device_eval_batch_size` and `eval_accumulation_steps` support, dual-vocab remapping stats logging, copies `dual_vocab_meta.json` to merged checkpoint.
- `sft/compat.py` — vLLM compatibility patches.
- `ragen/env/static/utils.py` — Fixed `compute_score_math` scoring bug: numeric fallback now only activates when gold label is a plain number (was incorrectly matching `\frac{3}{56}` as `3`).
- `config/sft/math_tokenskip_rawtext.yaml` — Training config for MATH TokenSkip.
- `config/sft/base.yaml` — Added `dual_vocab.enabled: false` default.

### 2. Dual-vocab integration

Integrated the dual-vocabulary model (latent reasoning tokens) into TokenSkip.

**Key concept:** The dual model has V=151936 visible tokens [0, V) and L=151936 latent tokens [V, V+L). During `<think>`, tokens are shifted +V to latent space. A logits constraint enforces latent-only during think, visible-only after `</think>`.

**Training:** `remap_think_to_latent()` in `train.py` shifts token IDs inside `<think>...</think>` by +V after tokenization. The SFT data file itself stays as plain text — remapping happens on token IDs in memory.

**Eval:** `eval_raw.py` auto-detects dual models via `dual_vocab_meta.json`, maps latent IDs back to visible for decoding (`tid - V`), and injects a vLLM logits processor to enforce the constraint during generation.

**vLLM compatibility fix:** vLLM 0.10.2 V0 engine has a guard at `llm_engine.py:673` that rejects per-request `logits_processors`. We patch around it with a two-part monkey-patch:
1. `add_request`: stash processors in a dict keyed by `request_id`, set `params.logits_processors = None` to pass the guard, restore after the call.
2. `_create_sequence_group_with_sampling`: re-inject stashed processors onto `sampling_params` before `_build_logits_processors` (line 723) and `clone()` (line 728) run.

This is in `eval_raw.py` lines 155-188. It's fragile and tied to vLLM 0.10.2 internals.

### 3. Multi-rate training

Added support for training with multiple compression ratios (0.3, 0.5, 0.7) instead of just 0.5, giving the model stronger conditioning signal for length control.

### 4. Script consolidation

Merged 4 pipeline scripts and 2 smoke tests into 1 each:
- `scripts/sft/pipeline_tokenskip_math.sh` — unified for standard/dual, any ratios
- `scripts/sft/smoke_test_tokenskip_math.sh` — unified smoke test

Deleted: `pipeline_tokenskip_math_dual.sh`, `pipeline_tokenskip_math_multirate.sh`, `pipeline_tokenskip_math_multirate_dual.sh`, `smoke_test_tokenskip_math_dual.sh`, `config/sft/math_tokenskip_dual.yaml`.

---

## Results on MATH (500 test samples)

Multi-rate training (ratios 1.0, 0.7, 0.5, 0.3):

| Model | Ratio | Accuracy | Avg Think Tokens | Avg Tok (correct) |
|-------|-------|----------|------------------|-------------------|
| Standard | 1.0 | 42.8% | 1530 | 1092 |
| Standard | 0.7 | 52.6% | 1330 | 991 |
| Standard | 0.5 | 51.6% | 1231 | 827 |
| Standard | 0.3 | 44.0% | 1331 | 762 |
| Dual | 1.0 | 43.2% | 1536 | 1121 |
| Dual | 0.7 | 50.4% | 1291 | 889 |
| **Dual** | **0.5** | **53.6%** | 1261 | 894 |
| Dual | 0.3 | 51.2% | 1266 | 862 |

---

## Known issues and limitations

### Token shortening is modest
Training data at r=0.3 averages ~350 tokens, but eval produces ~1266 tokens. The ratio signal controls conciseness/style more than raw length. This is inherent to SFT — the model learns to *sound like* compressed reasoning, not to *stop at a target count*. RL with a length penalty would be needed for stronger length control.

### vLLM exit crash
vLLM 0.10.2 crashes on process exit (`c10::Error: Trying to free a pointer not allocated here`). This is a CUDA cleanup issue, not functional — results are written correctly before the crash. However, `set -euo pipefail` in bash scripts treats it as a failure, which can prevent eval loops from completing all ratios. Workaround: run evals for each ratio separately, or remove `set -e`.

### vLLM logits processor monkey-patch
The dual-vocab eval monkey-patch in `eval_raw.py` is tied to vLLM 0.10.2 internals (`LLMEngine.add_request` and `_create_sequence_group_with_sampling`). It will likely break on vLLM upgrades.

---

## File layout reference

```
sft/
  train.py                          # SFT training with LoRA, dual-vocab remapping
  eval_raw.py                       # Evaluation with vLLM, dual-vocab support
  compat.py                         # vLLM compatibility patches
  data/
    base.py                         # SFTSample, response format builders
    prepare.py                      # Dataset preparation CLI
  methods/tokenskip/
    __init__.py
    collect.py                      # CoT collection with vLLM
    compress.py                     # LLMLingua-2 compression
    prompts.py                      # Prompt builders, answer extraction
    select.py                       # Training sample selection
    utils.py                        # Ratio conditioning helpers (moved from root)

config/sft/
  base.yaml                        # Base config (includes dual_vocab.enabled default)
  math_tokenskip_rawtext.yaml       # MATH TokenSkip config

scripts/sft/
  pipeline_tokenskip_math.sh        # Unified pipeline (standard + dual, any ratios)
  smoke_test_tokenskip_math.sh      # Unified smoke test

ragen/dual_vocab/
  constraint.py                     # Logits processors (HF + vLLM)
  utils.py                          # load_meta, is_dual_model, etc.
```

---

## How to run (quick reference)

```bash
# Standard, default ratios
LLMLINGUA_PATH=microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
    bash scripts/sft/pipeline_tokenskip_math.sh

# Dual-vocab
DUAL_MODEL=checkpoints/dual_qwen_4b_thinking \
LLMLINGUA_PATH=microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
    bash scripts/sft/pipeline_tokenskip_math.sh

# Skip data prep, retrain dual only
SKIP_COLLECT=1 SKIP_COMPRESS=1 SKIP_PREPARE=1 \
DUAL_MODEL=checkpoints/dual_qwen_4b_thinking \
    bash scripts/sft/pipeline_tokenskip_math.sh

# Smoke test (dual)
DUAL_MODEL=checkpoints/dual_qwen_4b_thinking \
LLMLINGUA_PATH=microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
    bash scripts/sft/smoke_test_tokenskip_math.sh

# Standalone eval at specific ratio
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m sft.eval_raw \
    --model results/sft/math/.../merged \
    --benchmark math --tokenskip-prompt --compression-ratio 0.5
```
