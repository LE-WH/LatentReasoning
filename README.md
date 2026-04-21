# LatentReasoning

**Dual-vocabulary latent chain-of-thought for LLM agents.**

LatentReasoning extends the [RAGEN](https://github.com/mll-lab-nu/RAGEN) framework with a dual-vocabulary architecture: the model reasons in a *latent* token space during `<think>...</think>`, then answers in the standard *visible* vocabulary. This separates "how the model thinks" from "what it says," enabling experiments on latent-only training, compressed reasoning, and hidden chain-of-thought.

## Key ideas

- **Dual vocabulary.** The tokenizer is expanded so that visible tokens live in `[0, V)` and latent mirrors in `[V, V+L)`. A logits-processor constraint enforces that only latent tokens are generated during the think phase, and only visible tokens after `</think>`.
- **`clone_eos` initialisation.** Newly-added `<think>`/`</think>` tokens are initialised by cloning the model's end-of-turn token (e.g. `<|im_end|>` for Qwen, `<|eot_id|>` for Llama-3) with subtle noise, giving the model a meaningful starting point so it can learn to transition out of the think phase.
- **Latent-only training.** With `dual_vocab.latent_only=true`, loss gradients only flow through latent token positions, training the model's internal reasoning without affecting its visible output distribution.
- **Static dataset evaluation.** A generic `StaticEnv` wrapper supports evaluating on HuggingFace datasets (MATH, GSM8K, MMLU, MetaMathQA) with pluggable processors and scorers.
- **SFT pipeline.** Supervised fine-tuning with two methods: direct (answer-only) and self-training concise (shortest correct chain-of-thought). Can serve as a warm-start before RL training.

---

## Setup

```bash
git clone <this-repo>
cd LatentReasoning
conda create -n ragen python=3.12 -y && conda activate ragen
bash scripts/setup_ragen.sh
```

---

## Step 1 -- Build the dual model

One-time setup that builds the expanded tokenizer + model.

```bash
bash scripts/build_dual_model.sh \
    --base_model Qwen/Qwen2.5-3B-Instruct \
    --out_dir ./checkpoints/dual_qwen_3b \
    --think_missing clone_eos
```
 
```bash
bash scripts/build_dual_model.sh \
    --base_model Qwen/Qwen3-4B-Thinking-2507 \
    --out_dir ./checkpoints/dual_qwen_4b_thinking \
    --think_missing clone_eos
```


**`--think_missing` options:**

| Option | Behaviour |
|---|---|
| `clone_eos` (recommended) | Adds `<think>`/`</think>` and initialises their embeddings from the model's EOS-like token + noise. Works across model families (Qwen, Llama-3, Phi-3, etc.). |
| `add` | Adds tokens with random near-zero init. The model may never learn to emit `</think>`. |
| `error` (default) | Fails if think tokens are not already in the vocabulary. |

What happens internally:
1. `scripts/build_dual_vocab.py` -- creates the dual tokenizer + `dual_vocab_meta.json` (records clone source when using `clone_eos`)
2. `scripts/expand_model_to_dual_vocab.py` -- resizes the embedding table, initialises think-token rows from clone source (or random), copies latent rows from visible tokens
3. `scripts/verify_dual_model.py` -- sanity checks

The final model is saved to `./checkpoints/dual_qwen_3b`.

---

## Step 2 -- Train

Use any RAGEN task config; override `model_path` to point to the dual model.

**MetaMathQA (recommended for dual model):**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config-name _5_metamathqa \
    model_path=./checkpoints/dual_qwen_3b \
    trainer.experiment_name=dual_metamathqa \
    trainer.total_training_steps=400
```

**Latent-only training:**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config-name _5_metamathqa \
    model_path=./checkpoints/dual_qwen_3b \
    trainer.experiment_name=dual_metamathqa_latentonly \
    dual_vocab.latent_only=true \
    trainer.total_training_steps=400
```

**Other environments (Sokoban, etc.):**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config-name _2_sokoban \
    model_path=./checkpoints/dual_qwen_3b \
    trainer.experiment_name=dual_sokoban
```

---

## Step 3 -- Eval

The eval script auto-detects dual models and applies the vocabulary constraint.

**Default eval (MetaMathQA):**
```bash
CUDA_VISIBLE_DEVICES=0 python -m ragen.eval \
    --config-name eval \
    model_path=./checkpoints/dual_qwen_3b \
    system.CUDA_VISIBLE_DEVICES=0
```

**Eval on static datasets (MATH, GSM8K, MMLU):**
```bash
CUDA_VISIBLE_DEVICES=0 python -m ragen.eval \
    --config-name eval \
    model_path=./checkpoints/dual_qwen_3b \
    system.CUDA_VISIBLE_DEVICES=0 \
    es_manager.train.env_configs.tags='["MATH"]' \
    es_manager.train.env_configs.n_groups='[8]' \
    es_manager.val.env_configs.tags='["MATH"]' \
    es_manager.val.env_configs.n_groups='[32]'
```

**Available static env tags** (defined in `config/envs.yaml`):

| Tag | Dataset |
|---|---|
| `MATH` | hendrycks competition math (`nlile/hendrycks-MATH-benchmark`), binary reward |
| `MATH_LogProb` | same dataset, continuous log-prob reward (requires scorer server) |
| `GSM8K` | grade school math (`openai/gsm8k`) |
| `StaticMetaMathQA` | MetaMathQA via the generic StaticEnv |
| `MetamathQA` | MetaMathQA via the dedicated env |

**Eval a trained checkpoint:**
```bash
CUDA_VISIBLE_DEVICES=0 python -m ragen.eval \
    --config-name eval \
    model_path=./checkpoints/dual_metamathqa/global_step_400 \
    system.CUDA_VISIBLE_DEVICES=0
```

---

## SFT (Supervised Fine-Tuning)

An SFT pipeline for training reasoning models before or independently of RL. Two methods are available:

- **Direct**: answer-only supervision (no reasoning trace)
- **Self-training concise**: samples multiple chain-of-thought traces, selects the shortest correct one per question

### Quick start

**Direct SFT on GSM8K:**
```bash
python -m sft.train --config-path ../config/sft --config-name gsm8k_direct
```

**Self-training concise (XML format, RAGEN-compatible):**
```bash
python -m sft.train --config-path ../config/sft --config-name gsm8k_self_training_concise
```

**Self-training concise (raw text, paper-aligned):**
```bash
python -m sft.train --config-path ../config/sft --config-name gsm8k_self_training_concise_rawtext
```

### Full self-training pipeline

The self-training concise method has a multi-step pipeline. Two end-to-end scripts are provided:

```bash
# XML format (RAGEN-compatible, uses <think>/<answer> tags)
bash scripts/sft/pipeline_xml.sh

# Raw text format (paper-aligned, no XML tags)
bash scripts/sft/pipeline_rawtext.sh
```

Both scripts run these steps:
1. **Zero-shot sampling** -- generate multiple CoT responses per question using vLLM
2. **Build exemplars** -- select the shortest correct traces as few-shot exemplars
3. **Few-shot sampling** -- resample with the exemplars as demonstrations
4. **Select training data** -- per-question, pick the shortest correct trace (merging zero-shot + few-shot)
5. **Train** -- fine-tune the model on the selected traces
6. **Evaluate** -- test on GSM8K

### SFT evaluation

**Raw text eval:**
```bash
CUDA_VISIBLE_DEVICES=0 python -m sft.eval_raw \
    --model results/sft/gsm8k/<model_dir> \
    --output results/sft/gsm8k/<model_dir>/eval/gsm8k_eval.jsonl
```

**RAGEN eval (uses the full environment framework):**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sft/eval_ragen.py \
    --config-name _11_gsm8k \
    model_path=results/sft/gsm8k/<model_dir>
```

### SFT configs

Configs live in `config/sft/`. Key settings in `base.yaml`:

| Setting | Default |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| LoRA | rank=64, alpha=128 (disabled in concise configs for full fine-tuning) |
| Learning rate | 2e-5 (base), 1e-5 (concise) |
| Epochs | 3 (base), 1 (concise) |
| Max length | 2048 |
| Precision | bf16 |

---

## TokenSkip: Compressed Reasoning with Ratio Conditioning

Replicates [TokenSkip](https://arxiv.org/abs/2502.12067) on MATH using Qwen3-4B-Thinking, with an extension to the dual-vocabulary latent reasoning model. The model learns to produce variable-length chain-of-thought conditioned on an inline compression ratio.

### Pipeline overview

| Step | What | Script |
|------|------|--------|
| 1. Collect | Sample `NUM_SAMPLES` CoTs per question on MATH train | `sft.methods.tokenskip.collect` |
| 2. Compress | Shorten CoTs with LLMLingua-2 at target ratios | `sft.methods.tokenskip.compress` |
| 3. Prepare | Build SFT data with ratio conditioning tokens | `sft.data.prepare` |
| 4. Train | LoRA fine-tune (standard or dual-vocab) | `sft.train` |
| 5. Eval | Generate on MATH test, score accuracy | `sft.eval_raw` |

The ratio signal is injected inline as `<|im_end|>{ratio}<|im_end|>` after the user message, training the model to control reasoning length.

**Pipeline env vars:**

| Variable | Default | Effect |
|---|---|---|
| `LLMLINGUA_PATH` | (required) | HF id / local path for LLMLingua-2 weights |
| `NUM_SAMPLES` | `8` | Responses generated per question at collect time |
| `SAMPLE_TEMP` | `0.7` | vLLM sampling temperature at collect time |
| `COMPRESS_SHARDS` | `NUM_SHARDS` | Parallel LLMLingua-2 workers at step 2 (one GPU each) |
| `COMPRESS_RATIOS` | `0.1,0.3,0.5,0.7` | Ratios passed to LLMLingua-2 at step 2 |
| `TRAIN_RATIOS` | `1.0,0.1,0.3,0.5,0.7` | Ratio pool randomly sampled per training row |
| `EVAL_RATIOS` | `1.0,0.7,0.5,0.3,0.1` | Ratios to evaluate at (one W&B run each) |
| `MAX_COT_TOKENS` | `2000` | Drop CoTs longer than this before compression |
| `GPUS` / `TRAIN_NPROC` / `NUM_SHARDS` | `0,1,2,3` / `4` / `4` | Parallelism knobs |
| `DUAL_MODEL` | unset | Path to dual-vocab checkpoint (enables dual mode) |
| `SKIP_COLLECT` / `SKIP_COMPRESS` / `SKIP_PREPARE` / `SKIP_TRAIN` | unset | Set to `1` to skip a step |

### Running the pipeline

A single unified script handles both standard and dual-vocab models with configurable ratios.

**Prerequisites:**
```bash
export LLMLINGUA_PATH=microsoft/llmlingua-2-xlm-roberta-large-meetingbank
```

**Standard model:**
```bash
LLMLINGUA_PATH=microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
WANDB_PROJECT=tokenskip-math \
    bash scripts/sft/pipeline_tokenskip_math.sh
```

**Dual-vocab model:**
```bash
LLMLINGUA_PATH=microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
DUAL_MODEL=checkpoints/dual_qwen_4b_thinking \
WANDB_PROJECT=tokenskip-math \
    bash scripts/sft/pipeline_tokenskip_math.sh
```

**Custom ratios and GPUs:**
```bash
COMPRESS_RATIOS=0.2,0.4,0.6,0.8 \
TRAIN_RATIOS=1.0,0.2,0.4,0.6,0.8 \
EVAL_RATIOS=1.0,0.8,0.6,0.4,0.2 \
GPUS=0,1 TRAIN_NPROC=2 \
    bash scripts/sft/pipeline_tokenskip_math.sh
```

**Multi-sample collection (default) vs. paper-strict greedy.** The pipeline collects `NUM_SAMPLES=8` responses per question at `SAMPLE_TEMP=0.7` by default, giving ~8× more training data than the greedy setup in the original paper. Each sampled response gets a unique `source_id` (`..._s0`, `..._s1`, …) and independently flows through compression and ratio sampling.

```bash
# Cheaper / faster: 4 diverse samples
NUM_SAMPLES=4 SAMPLE_TEMP=0.7 \
LLMLINGUA_PATH=microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
    bash scripts/sft/pipeline_tokenskip_math.sh

# Paper-strict (1 greedy response per question)
NUM_SAMPLES=1 SAMPLE_TEMP=0.0 \
LLMLINGUA_PATH=microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
    bash scripts/sft/pipeline_tokenskip_math.sh
```

Note: if you have pre-existing data collected before this change (source_ids without the `_s{idx}` suffix), run `rm -rf data/sft/tokenskip/` before re-running — the old format is incompatible with the new select/compress pipeline.

**Skip completed steps** (e.g. reuse collected CoTs, retrain only):
```bash
SKIP_COLLECT=1 SKIP_COMPRESS=1 SKIP_PREPARE=1 \
DUAL_MODEL=checkpoints/dual_qwen_4b_thinking \
    bash scripts/sft/pipeline_tokenskip_math.sh
```

**Smoke test (20 questions, 1 epoch):**
```bash
# Standard
LLMLINGUA_PATH=microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
    bash scripts/sft/smoke_test_tokenskip_math.sh

# Dual
LLMLINGUA_PATH=microsoft/llmlingua-2-xlm-roberta-large-meetingbank \
DUAL_MODEL=checkpoints/dual_qwen_4b_thinking \
    bash scripts/sft/smoke_test_tokenskip_math.sh
```

**Standalone eval:**
```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m sft.eval_raw \
    --model results/sft/math/<model_dir>/merged \
    --benchmark math \
    --tokenskip-prompt \
    --compression-ratio 0.5 \
    --output results/eval.jsonl \
    --wandb-project tokenskip-math
```

### Output locations

| Step | Path |
|------|------|
| 1. Collected CoTs | `data/sft/tokenskip/original/math/` |
| 2. Compressed CoTs | `data/sft/tokenskip/compressed/math_raw/` |
| 3. SFT data | `data/sft/math_tokenskip_rawtext.jsonl` |
| 4. Model (standard) | `results/sft/math/qwen3-4b-thinking-tokenskip-standard/merged/` |
| 4. Model (dual) | `results/sft/math/qwen3-4b-thinking-tokenskip-dual/merged/` |
| 5. Eval results | `results/sft/math/.../eval/math_eval_ratio_{r}.jsonl` |

### Results on MATH (500 test samples, multi-rate training with ratios 1.0, 0.7, 0.5, 0.3)

| Model | Ratio | Accuracy | Avg Think Tokens | Avg Tok (correct) |
|-------|-------|----------|------------------|-------------------|
| Standard | 1.0 | 42.8% | 1530 | 1092 |
| Standard | 0.7 | 52.6% | 1330 | 991 |
| Standard | 0.5 | 51.6% | 1231 | 827 |
| Standard | 0.3 | 44.0% | 1331 | 762 |
| **Dual** | 1.0 | 43.2% | 1536 | 1121 |
| **Dual** | 0.7 | 50.4% | 1291 | 889 |
| **Dual** | **0.5** | **53.6%** | 1261 | 894 |
| **Dual** | 0.3 | 51.2% | 1266 | 862 |

**Observations:**
- Dual-vocab matches standard at r=1.0 — latent tokens don't degrade quality.
- Ratio conditioning works: lower ratios produce shorter, more concise CoTs with higher accuracy (compressed reasoning removes filler).
- Dual r=0.5 achieves the best accuracy (53.6%), 10+ points above the r=1.0 baselines.
- All models are trained on the same dataset; only the ratio in the eval prompt differs. The ratio signal shifts the model's reasoning *style* (verbose vs. concise) rather than setting a hard token budget.

**Known limitation — modest length reduction.** Training data at r=0.3 averages ~350 tokens, but at eval the model produces ~1266 tokens. The ratio conditioning controls conciseness more than raw length. This is inherent to SFT: the model learns to *sound like* compressed reasoning, not to *stop at a target count*. Stronger length control would require RL with an explicit length penalty.

### Compression demo (Gradio UI)

An interactive tool for inspecting Step 2 of the TokenSkip pipeline — the LLMLingua-2 compression itself. Useful for getting an intuition for what a given rate actually produces before kicking off a full run.

Two modes:
- **Samples tab** — pick from preset math-style CoTs, slide the rate, see the compressed output, metrics, and a diff view with dropped tokens struck-through.
- **Custom tab** — paste any text and compress it at an arbitrary rate.

**Run locally:**
```bash
bash scripts/demo/run_compression_ui.sh               # http://localhost:7860
PORT=8000 bash scripts/demo/run_compression_ui.sh     # custom port
SHARE=1  bash scripts/demo/run_compression_ui.sh      # public *.gradio.live URL
LLMLINGUA_PATH=/local/path bash scripts/demo/run_compression_ui.sh
```

**On a remote server (recommended):** use SSH port forwarding from your laptop:
```bash
ssh -N -L 7860:localhost:7860 user@your-server
# then open http://localhost:7860 in your local browser
```

The UI includes a collapsible *"Why is the actual rate not exactly my target?"* explainer that covers the four real causes (threshold-not-top-k, chunking, tokenizer mismatch, asymmetric bias) and clarifies that the `force_tokens` / `force_reserve_digit` / `drop_consecutive` flags are only active in the pipeline's `llama3` branch, not here.

---

## Continuous reward (log-prob scorer)

Instead of binary 0/1 correctness, the log-prob scorer computes `mean log P(gold_answer | prompt + CoT)` -- a continuous reward that measures how well the reasoning trajectory supports the correct answer. This provides smoother gradients for RL training.

Migrated from [scalable-latent-reasoning](https://github.com/...).

### How it works

1. The model generates a response with `<think>latent tokens...</think>answer`
2. The scorer extracts the latent prefix (up to `</think>`)
3. It feeds `prompt + latent_prefix` into the model and computes the mean token log-probability of the gold answer
4. Higher score (closer to 0) = the reasoning makes the correct answer more predictable

Reward range: typically `[-10, 0]`, with `-100.0` for invalid inputs and a `-2.0` penalty if `</think>` is missing.

### Usage

**Terminal 1 -- start the scorer server:**
```bash
CUDA_VISIBLE_DEVICES=1 python -m ragen.reward.scorer_server \
    --model_dir ./checkpoints/dual_qwen_4B_thinking \
    --port 8009
```

**Terminal 2 -- run RL training with log-prob reward:**
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config-name _5_metamathqa \
    model_path=./checkpoints/dual_qwen_4B_thinking \
    trainer.experiment_name=dual_4b_math_logprob \
    es_manager.train.env_configs.tags='["MATH_LogProb"]' \
    es_manager.train.env_configs.n_groups='[8]' \
    es_manager.val.env_configs.tags='["MATH"]' \
    es_manager.val.env_configs.n_groups='[512]' \
    agent_proxy.max_turn=1 \
    actor_rollout_ref.rollout.response_length=800 \
    system.CUDA_VISIBLE_DEVICES=0
```

Note: use `MATH_LogProb` for train (continuous reward) and `MATH` for val (binary accuracy for clean metrics).

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `SCORER_URL` | `http://127.0.0.1:8009` | Scorer server endpoint |
| `SCORER_TIMEOUT` | `120` | HTTP request timeout (seconds) |

---

## Config reference

| Override | Effect |
|---|---|
| `model_path=./checkpoints/dual_qwen_3b` | Use the dual model |
| `dual_vocab.latent_only=true` | Train loss only on latent (think-phase) tokens |
| `actor_rollout_ref.rollout.response_length=800` | Longer responses (useful since latent tokens are hidden) |
| `agent_proxy.enable_think=true` | Keep `<think>` tags enabled (default) |

---

## How dual-vocab works at runtime

Once `model_path` points to a directory containing `dual_vocab_meta.json`:

- **Rollout** (`VllmWrapperWg`): a per-request vLLM logits processor is injected. During `<think>`, only latent tokens `[V, V+L)` + `</think>` + EOS are allowed. After `</think>`, only visible tokens `[0, V)` + EOS are allowed.
- **Training** (`ContextManager`): the loss mask accounts for latent tokens. With `latent_only=true`, gradients only flow through latent token positions.

---

## Built on RAGEN

This project is built on top of [RAGEN](https://github.com/mll-lab-nu/RAGEN) (Reasoning Agent), a flexible RL framework for training reasoning agents. RAGEN provides:

- **StarPO** (State-Thinking-Actions-Reward Policy Optimization) for multi-turn trajectory-level RL training
- **10+ built-in environments**: Sokoban, FrozenLake, WebShop, DeepCoder, SearchQA, Lean, Bandit, Countdown, MetaMathQA, Sudoku
- **V2 diagnostics**: SNR-Adaptive Filtering and mutual-information-based reasoning collapse detection

For the full RAGEN documentation, see:
- [RAGEN V2 paper](https://ragen-ai.github.io/v2)
- [RAGEN V1 paper](https://arxiv.org/abs/2504.20073)
- [RAGEN documentation](https://ragen-doc.readthedocs.io/)

### RAGEN citation

```bibtex
@misc{ragen-v2,
      title={RAGEN-V2: Understanding Reasoning Collapse in LLM Agent Reinforcement Learning},
      author={Zihan Wang and Chi Gui and Xing Jin and Qineng Wang and Licheng Liu and Kangrui Wang and Shiqi Chen and Linjie Li and Zhengyuan Yang and Pingyue Zhang and Yiping Lu and Jiajun Wu and Li Fei-Fei and Lijuan Wang and Yejin Choi and Manling Li},
      year={2026},
      url={https://ragen-ai.github.io/v2},
}
```

```bibtex
@misc{ragen,
      title={RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning},
      author={Zihan Wang and Kangrui Wang and Qineng Wang and Pingyue Zhang and Linjie Li and Zhengyuan Yang and Xing Jin and Kefan Yu and Minh Nhat Nguyen and Licheng Liu and Eli Gottlieb and Yiping Lu and Kyunghyun Cho and Jiajun Wu and Li Fei-Fei and Lijuan Wang and Yejin Choi and Manling Li},
      year={2025},
      eprint={2504.20073},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.20073},
}
```
