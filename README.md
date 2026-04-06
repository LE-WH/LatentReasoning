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
