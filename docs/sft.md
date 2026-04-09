# SFT Module Guide

> Paper: [Self-Training Elicits Concise Reasoning](https://arxiv.org/abs/2502.20122)
> Code: [TergelMunkhbat/concise-reasoning](https://github.com/TergelMunkhbat/concise-reasoning)

## Overview

SFT module implements the Self-Training Concise Reasoning paper on the RAGEN framework. It provides two pipelines:

- **XML pipeline**: RAGEN-compatible format (`<think>/<answer>` tags + chat template). SFT model can be directly used for RL training and evaluated alongside RL models.
- **Raw text pipeline**: Paper's original format (`"The answer is"` + raw text). Matches the paper's exact training setup for reproduction.

## Quick Start

```bash
# XML pipeline (RAGEN-compatible)
bash scripts/sft/pipeline_xml.sh

# Raw text pipeline (paper-aligned)
bash scripts/sft/pipeline_rawtext.sh
```

## Pipeline Steps

Both pipelines follow the same 5-step process:

```
Step 1: Zero-shot sampling
  → Each question gets 16 model-generated responses

Step 2: Few-shot sampling
  → Build 128 shortest-correct exemplars from Step 1
  → Re-sample with 8 exemplars per prompt

Step 3: Prepare training data
  → Merge Step 1 + Step 2 responses
  → For each question: dedup → hard cutoff (<511 tokens) → select shortest correct

Step 4: Train
  → Full fine-tuning, 1 epoch, lr=1e-5, constant scheduler
  → Hyperparameters match the paper

Step 5: Evaluate
  → XML: RAGEN framework (agent_proxy + StaticEnv)
  → Raw text: eval_raw.py (vLLM + direct prompt)
```

## Results

### Raw text evaluation (paper-aligned)

| Model | Accuracy | Avg Tokens | Token Reduction |
|-------|----------|------------|-----------------|
| Base (Qwen2.5-3B-Instruct) | 83.9% | 296.8 | — |
| SFT | **84.2%** | **250.5** | **-15.6%** |

### RAGEN evaluation (XML format)

| Model | pass@1 | Avg Tokens | Token Reduction |
|-------|--------|------------|-----------------|
| Base (Qwen2.5-3B-Instruct) | 36.7% | 172.2 | — |
| SFT | **58.8%** | **148.9** | **-13.5%** |

Note: RAGEN base model accuracy is low because the base model is not familiar with `<think>/<answer>` XML tags. SFT teaches the model this format.

## Format Comparison

| | Raw Text | RAGEN XML |
|---|----------|-----------|
| System prompt | `"Your task is... use 'The answer is'"` | `"Write reasoning in <think>... answer in <answer>"` |
| Model output | `reasoning... The answer is 42` | `<think>reasoning</think><answer>42</answer>` |
| Training format | raw text (`use_chat_template: false`) | chat template (`use_chat_template: true`) |
| Evaluation | `eval_raw.py` (direct prompt) | RAGEN agent_proxy (environment interaction) |
| Compatible with RL | No | **Yes** |

## Project Structure

```
sft/
├── methods/
│   ├── base.py                          Method base class
│   ├── direct.py                        Direct SFT (answer-only baseline)
│   └── self_training_concise/           Self-Training paper implementation
│       ├── __init__.py                  Paper attribution
│       ├── prompts.py                   Prompts + answer extraction
│       ├── sample.py                    CoT sampling (Step 1 & 2)
│       ├── build_exemplars.py           Few-shot exemplar selection
│       └── select.py                    Training data selection (reward_func)
├── data/
│   ├── base.py                          SFTSample dataclass
│   └── prepare.py                       Data preparation CLI
├── train.py                             HuggingFace Trainer
├── eval_raw.py                          Raw text evaluation
└── compat.py                            vLLM compatibility patches

ragen/env/
├── reward_funcs.py                      Shared reward function (SFT + RL)
└── wrappers.py                          LengthRewardEnvWrapper (for RL)

config/sft/
├── base.yaml                            Default SFT config
├── gsm8k_self_training_concise.yaml     XML version
└── gsm8k_self_training_concise_rawtext.yaml  Raw text version

scripts/sft/
├── pipeline_xml.sh                      XML end-to-end pipeline
├── pipeline_rawtext.sh                  Raw text end-to-end pipeline
├── eval_ragen.py                        RAGEN evaluation entry point
└── verify_envs.py                       Environment verification tests
```

## RAGEN Modifications

| File | Change | Why |
|------|--------|-----|
| `config/envs.yaml` | Added GSM8K environment definition | RAGEN didn't have GSM8K registered |
| `ragen/env/__init__.py` | Registered StaticEnv | StaticEnv wasn't in REGISTERED_ENVS |
| `ragen/env/static/env.py` | Added `render()` method | RAGEN eval calls render(), StaticEnv didn't have it |
| `scripts/eval_selected_envs.sh` | Added gsm8k option | Enable GSM8K evaluation |

## Shared Reward Function

`ragen/env/reward_funcs.py` provides `init_reward_func()` used by both:
- **SFT**: scoring candidates during data selection (`select.py`, `build_exemplars.py`)
- **RL** (future): via `LengthRewardEnvWrapper` to shape environment rewards

This ensures SFT and RL use the same definition of "good response" (correct + concise).


