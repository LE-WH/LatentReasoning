"""Collect original CoT trajectories for TokenSkip SFT."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset

from sft.compat import apply_patches

apply_patches()

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ragen.env.static.utils import compute_score_numeric, process_gsm8k
from sft.methods.tokenskip.prompts import (
    build_tokenskip_raw_chat_messages,
    extract_raw_reasoning_and_answer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_questions(cache_dir: str, num_questions: int) -> list[dict]:
    """Load GSM8K train questions."""
    ds = load_dataset("openai/gsm8k", name="main", split="train", cache_dir=cache_dir)
    raw_data = []
    for idx, item in enumerate(ds):
        if num_questions > 0 and idx >= num_questions:
            break
        question, answer = process_gsm8k(item)
        raw_data.append(
            {
                "question": question,
                "answer": answer,
                "source_id": f"gsm8k_train_{idx}",
            }
        )
    return raw_data


def build_messages(question: str, benchmark: str, fmt: str) -> list[dict]:
    """Build generation messages for original CoT collection."""
    return build_tokenskip_raw_chat_messages(question, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect original CoTs for TokenSkip")
    parser.add_argument("--benchmark", type=str, default="gsm8k", choices=["gsm8k"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--cache-dir", type=str, default="./data")
    parser.add_argument("--prompt-format", type=str, default="paper", choices=["paper"])
    parser.add_argument("--num-questions", type=int, default=-1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    args = parser.parse_args()

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    raw_data = load_questions(args.cache_dir, args.num_questions)
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")
    if args.num_shards > 1:
        raw_data = raw_data[args.shard_id::args.num_shards]
    logger.info("Loaded %d training questions", len(raw_data))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )

    llm = LLM(
        args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel,
        enable_sleep_mode=True,
        enforce_eager=True,
        disable_custom_all_reduce=args.tensor_parallel == 1,
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        n=1,
    )

    prompts = []
    for item in raw_data:
        messages = build_messages(item["question"], args.benchmark, args.prompt_format)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    outputs = llm.generate(prompts, sampling_params)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    extract_fn = extract_raw_reasoning_and_answer

    with open(args.output, "w") as f:
        for item, output in zip(raw_data, outputs):
            response_text = output.outputs[0].text
            reasoning, answer = extract_fn(response_text)

            is_correct = False
            if answer is not None:
                score = compute_score_numeric(answer, item["answer"])
                is_correct = score["is_correct"]

            reasoning_token_count = len(
                tokenizer.encode(reasoning or "", add_special_tokens=False)
            )

            record = {
                "benchmark": args.benchmark,
                "source_id": item["source_id"],
                "question": item["question"],
                "gold_answer": item["answer"],
                "prompt_format": args.prompt_format,
                "response_text": response_text.strip(),
                "reasoning": reasoning,
                "answer": answer,
                "is_correct": is_correct,
                "reasoning_token_count": reasoning_token_count,
                "response_token_count": len(output.outputs[0].token_ids),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Saved original CoTs to %s", args.output)


if __name__ == "__main__":
    main()
