"""SFT training script with LoRA.

Loads chat-format jsonl data, applies response masking (loss only on assistant
tokens), trains with LoRA via peft, then merges and saves a full checkpoint.

Supports dual-vocabulary models: when ``dual_vocab.enabled`` is set in the
config, the reasoning tokens inside the ``<think>...</think>`` block are
remapped to their latent counterparts (visible_id + V).

Usage:
    python -m sft.train --config-path ../config/sft --config-name gsm8k_direct
"""

import json
import logging
import os
import shutil
from pathlib import Path

import hydra
import torch
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading & tokenization
# ---------------------------------------------------------------------------

def load_jsonl(path: str, max_samples: int | None = None) -> list[dict]:
    """Load messages from a jsonl file."""
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def _looks_like_xml_response(text: str) -> bool:
    return "<think>" in text or "<answer>" in text


def is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def maybe_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def validate_samples(samples: list[dict], use_chat_template: bool) -> None:
    """Catch stale XML-tagged data before paper-style training starts."""
    if use_chat_template:
        return

    for idx, sample in enumerate(samples):
        messages = sample.get("messages", [])
        assistant = next(
            (m.get("content", "") for m in messages if m.get("role") == "assistant"),
            "",
        )
        if _looks_like_xml_response(assistant):
            raise ValueError(
                "Raw-text training received XML-tagged assistant responses. "
                "Regenerate the dataset with `prepare.py --response-format paper` "
                f"before training. First offending sample index: {idx}."
            )


def tokenize_with_response_mask(
    samples: list[dict],
    tokenizer: AutoTokenizer,
    max_length: int,
    use_chat_template: bool = True,
) -> Dataset:
    """Tokenize samples and create labels with response masking.

    Labels for all tokens before the response are set to -100 so
    the loss is computed only on the response portion.

    When use_chat_template=False, uses raw text format aligned with the
    original paper (sft_trainer.py): ``input_text + ' ' + label + EOS``.
    """
    all_input_ids = []
    all_labels = []
    all_attention_mask = []

    for sample in samples:
        messages = sample["messages"]

        if use_chat_template:
            # ---- Chat template mode (system + user + assistant) ----
            non_assistant = [m for m in messages if m["role"] != "assistant"]
            prefix_text = tokenizer.apply_chat_template(
                non_assistant, tokenize=False, add_generation_prompt=True
            )
            prefix_ids = tokenizer(
                prefix_text, add_special_tokens=False, truncation=False
            )["input_ids"]

            full_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            # ---- Raw text mode (aligned with paper) ----
            # Paper: input_text = question; full_text = question + ' ' + rationale + EOS
            question = next(m["content"] for m in messages if m["role"] == "user")
            response = next(m["content"] for m in messages if m["role"] == "assistant")

            input_text = question
            prefix_ids = tokenizer(
                input_text, add_special_tokens=False, truncation=False
            )["input_ids"]

            full_text = input_text + " " + response + tokenizer.eos_token

        full_encoding = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        input_ids = full_encoding["input_ids"]
        attention_mask = full_encoding["attention_mask"]

        # Labels: -100 for prefix, real token ids for response
        labels = [-100] * len(input_ids)
        prefix_len = min(len(prefix_ids), len(input_ids))
        for i in range(prefix_len, len(input_ids)):
            labels[i] = input_ids[i]

        all_input_ids.append(input_ids)
        all_labels.append(labels)
        all_attention_mask.append(attention_mask)

    return Dataset.from_dict({
        "input_ids": all_input_ids,
        "labels": all_labels,
        "attention_mask": all_attention_mask,
    })


def remap_think_to_latent(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    meta: dict,
) -> Dataset:
    """Remap tokens inside the <think>...</think> region to latent IDs.

    For each sample, find the think region in the response and shift those
    token IDs by V (the visible vocab size) so they become latent tokens.
    Both input_ids and labels are updated.
    """
    V = meta["V"]
    think_end_id = meta["think_end_token_id"]

    new_input_ids = []
    new_labels = []

    for i in range(len(dataset)):
        input_ids = list(dataset[i]["input_ids"])
        labels = list(dataset[i]["labels"])

        # Find the response start (first non-masked label)
        resp_start = 0
        for j, lbl in enumerate(labels):
            if lbl != -100:
                resp_start = j
                break

        # Find </think> in the response region
        think_end_pos = None
        for j in range(resp_start, len(input_ids)):
            if input_ids[j] == think_end_id:
                think_end_pos = j
                break

        if think_end_pos is not None:
            # Remap tokens in [resp_start, think_end_pos) to latent
            for j in range(resp_start, think_end_pos):
                vis_id = input_ids[j]
                if 0 <= vis_id < V:
                    input_ids[j] = vis_id + V
                    if labels[j] != -100:
                        labels[j] = vis_id + V

        new_input_ids.append(input_ids)
        new_labels.append(labels)

    return Dataset.from_dict({
        "input_ids": new_input_ids,
        "labels": new_labels,
        "attention_mask": dataset["attention_mask"],
    })


def log_masking_sanity_check(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    num_examples: int = 3,
) -> None:
    """Log the first few examples to verify response masking boundaries."""
    for idx in range(min(num_examples, len(dataset))):
        input_ids = dataset[idx]["input_ids"]
        labels = dataset[idx]["labels"]
        # Find the boundary where labels switch from -100 to real ids
        boundary = 0
        for i, lbl in enumerate(labels):
            if lbl != -100:
                boundary = i
                break

        masked_text = tokenizer.decode(input_ids[:boundary], skip_special_tokens=False)
        response_text = tokenizer.decode(input_ids[boundary:], skip_special_tokens=False)
        logger.info(
            f"Sample {idx}: total_tokens={len(input_ids)}, "
            f"masked_prefix={boundary}, response_tokens={len(input_ids) - boundary}"
        )
        logger.info(f"  Prefix (masked):  {masked_text[:200]}...")
        logger.info(f"  Response (train):  {response_text[:200]}")
        logger.info("")


# ---------------------------------------------------------------------------
# Data collator with padding
# ---------------------------------------------------------------------------

class SFTDataCollator:
    """Pads input_ids, attention_mask, and labels to the same length within a batch."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        labels = []
        attention_mask = []

        for f in features:
            pad_len = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad_len)
            labels.append(f["labels"] + [-100] * pad_len)
            attention_mask.append(f["attention_mask"] + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../config/sft", config_name="gsm8k_direct")
def main(cfg: DictConfig) -> None:
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # --- Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.path, cache_dir=cfg.model.cache_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Data ---
    raw_samples = load_jsonl(cfg.data.path, max_samples=cfg.data.max_samples)
    logger.info(f"Loaded {len(raw_samples)} samples from {cfg.data.path}")

    use_chat_template = cfg.training.get("use_chat_template", True)
    logger.info(f"Training format: {'chat template' if use_chat_template else 'raw text (paper-aligned)'}")
    validate_samples(raw_samples, use_chat_template)

    full_dataset = tokenize_with_response_mask(
        raw_samples, tokenizer, cfg.model.max_length, use_chat_template=use_chat_template
    )
    logger.info(f"Tokenized dataset: {len(full_dataset)} samples")

    # --- Dual-vocab remapping ---
    dual_meta = None
    dual_enabled = cfg.get("dual_vocab", {}).get("enabled", False)
    if dual_enabled:
        from ragen.dual_vocab.utils import load_meta
        dual_meta = load_meta(cfg.model.path)
        logger.info(
            f"Dual-vocab enabled: V={dual_meta['V']}, L={dual_meta['L']}. "
            f"Remapping think tokens to latent IDs."
        )
        full_dataset = remap_think_to_latent(full_dataset, tokenizer, dual_meta)

        # Log latent remapping stats
        V = dual_meta["V"]
        total_latent = 0
        total_response = 0
        for i in range(len(full_dataset)):
            ids = full_dataset[i]["input_ids"]
            labels = full_dataset[i]["labels"]
            n_latent = sum(1 for tid in ids if V <= tid < V + dual_meta["L"])
            n_resp = sum(1 for lbl in labels if lbl != -100)
            total_latent += n_latent
            total_response += n_resp
        logger.info(
            f"Dual-vocab remapping stats: {total_latent} latent tokens "
            f"out of {total_response} response tokens "
            f"({total_latent/total_response*100:.1f}%) across {len(full_dataset)} samples"
        )

    # Split into train/eval if load_best_model_at_end is enabled
    eval_dataset = None
    eval_ratio = cfg.training.get("eval_ratio", 0.05)
    if cfg.training.get("load_best_model_at_end", False):
        split = full_dataset.train_test_split(test_size=eval_ratio, seed=cfg.training.get("seed", 42))
        dataset = split["train"]
        eval_dataset = split["test"]
        logger.info(f"Split: {len(dataset)} train, {len(eval_dataset)} eval")
    else:
        dataset = full_dataset

    # Sanity check: print first 3 examples
    if is_main_process():
        log_masking_sanity_check(dataset, tokenizer, num_examples=3)

    # --- Model ---
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.path,
        cache_dir=cfg.model.cache_dir,
        torch_dtype=torch.bfloat16 if cfg.training.bf16 else torch.float32,
        trust_remote_code=True,
    )

    if cfg.lora.enabled:
        target_modules_cfg = cfg.lora.target_modules
        if isinstance(target_modules_cfg, str):
            target_modules = target_modules_cfg
        else:
            target_modules = OmegaConf.to_container(target_modules_cfg, resolve=True)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora.rank,
            lora_alpha=cfg.lora.alpha,
            target_modules=target_modules,
            lora_dropout=0.0,
        )
        model = get_peft_model(model, lora_config)
        if is_main_process():
            model.print_trainable_parameters()

    # --- Training ---
    output_dir = cfg.training.output_dir
    save_steps = cfg.training.get("save_steps", None)
    training_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.per_device_batch_size,
        per_device_eval_batch_size=cfg.training.get("per_device_eval_batch_size", cfg.training.per_device_batch_size),
        eval_accumulation_steps=cfg.training.get("eval_accumulation_steps", None),
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.get("lr_scheduler_type", "linear"),
        warmup_ratio=cfg.training.warmup_ratio,
        weight_decay=cfg.training.weight_decay,
        bf16=cfg.training.bf16,
        logging_steps=cfg.training.logging_steps,
        save_strategy=cfg.training.save_strategy,
        seed=cfg.training.get("seed", 42),
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        report_to=cfg.training.get("report_to", "none"),
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        load_best_model_at_end=cfg.training.get("load_best_model_at_end", False),
        metric_for_best_model=cfg.training.get("metric_for_best_model", "loss"),
        save_total_limit=cfg.training.get("save_total_limit", None),
        optim=cfg.training.get("optim", "adamw_torch"),
    )
    if "greater_is_better" in cfg.training:
        training_kwargs["greater_is_better"] = cfg.training.greater_is_better
    if save_steps is not None:
        training_kwargs["save_steps"] = save_steps
    max_steps = cfg.training.get("max_steps", None)
    if max_steps is not None and int(max_steps) > 0:
        training_kwargs["max_steps"] = int(max_steps)
    if cfg.training.get("load_best_model_at_end", False):
        training_kwargs["eval_strategy"] = cfg.training.save_strategy
        if save_steps is not None:
            training_kwargs["eval_steps"] = save_steps
    run_name = cfg.training.get("run_name", None)
    if run_name:
        training_kwargs["run_name"] = run_name
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=SFTDataCollator(tokenizer),
    )

    logger.info("Starting training...")
    trainer.train()

    # --- Save model ---
    if is_main_process():
        if cfg.lora.enabled:
            adapter_dir = str(Path(output_dir) / "adapter")
            logger.info(f"Saving LoRA adapter to {adapter_dir}")
            model.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)

            merged_dir = str(Path(output_dir) / "merged")
            logger.info(f"Merging adapter and saving to {merged_dir}")
            merged_model = model.merge_and_unload()
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir)
            save_dir = merged_dir
        else:
            logger.info(f"Saving full model to {output_dir}")
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            save_dir = output_dir

        # Copy dual_vocab_meta.json so the saved model is self-contained
        if dual_meta is not None:
            src_meta = Path(cfg.model.path) / "dual_vocab_meta.json"
            if src_meta.exists():
                dst_meta = Path(save_dir) / "dual_vocab_meta.json"
                shutil.copy2(str(src_meta), str(dst_meta))
                logger.info(f"Copied dual_vocab_meta.json to {dst_meta}")

    maybe_barrier()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
