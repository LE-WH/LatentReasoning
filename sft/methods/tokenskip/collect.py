"""Collect original CoT trajectories for TokenSkip SFT."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from datasets import load_dataset

from sft.compat import apply_patches

apply_patches()

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ragen.env.static.utils import (
    compute_score_math,
    compute_score_numeric,
    process_gsm8k,
    process_math,
)
from sft.methods.tokenskip.prompts import (
    build_tokenskip_raw_chat_messages,
    extract_raw_reasoning_and_answer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_BENCHMARK_REGISTRY: dict[str, dict] = {
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "train",
        "processor": process_gsm8k,
        "scorer": compute_score_numeric,
    },
    "math": {
        "path": "nlile/hendrycks-MATH-benchmark",
        "name": None,
        "split": "train",
        "processor": process_math,
        "scorer": compute_score_math,
    },
}


def load_questions(benchmark: str, cache_dir: str, num_questions: int) -> list[dict]:
    """Load train questions for the given benchmark."""
    reg = _BENCHMARK_REGISTRY[benchmark]
    load_kwargs: dict = {"path": reg["path"], "split": reg["split"], "cache_dir": cache_dir}
    if reg["name"]:
        load_kwargs["name"] = reg["name"]
    ds = load_dataset(**load_kwargs)
    processor = reg["processor"]
    raw_data = []
    for idx, item in enumerate(ds):
        if num_questions > 0 and idx >= num_questions:
            break
        question, answer = processor(item)
        raw_data.append(
            {
                "question": question,
                "answer": answer,
                "source_id": f"{benchmark}_train_{idx}",
            }
        )
    return raw_data


def parse_think_output(text: str) -> tuple[str | None, str]:
    """Parse model output generated after a ``<think>`` prompt prefix.

    The generated text looks like:
        ``reasoning...</think>\nvisible answer with \\boxed{}``

    If ``</think>`` is present, returns (reasoning, answer_part).
    If ``<think>...</think>`` is fully present (model echoed the open tag),
    extracts the inner block.  Otherwise returns (None, original_text).
    """
    # Case 1: full <think>...</think> block in output (model echoed open tag)
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        reasoning = m.group(1).strip()
        after = text[m.end():].strip()
        return reasoning, after

    # Case 2: we prefixed <think> in the prompt, output starts with reasoning
    idx = text.find("</think>")
    if idx != -1:
        reasoning = text[:idx].strip()
        after = text[idx + len("</think>"):].strip()
        return reasoning, after

    # No think block found
    return None, text


def build_messages(question: str, benchmark: str, fmt: str) -> list[dict]:
    """Build generation messages for original CoT collection."""
    return build_tokenskip_raw_chat_messages(question, None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect original CoTs for TokenSkip")
    parser.add_argument("--benchmark", type=str, default="gsm8k", choices=list(_BENCHMARK_REGISTRY.keys()))
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
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Responses per question. Use with temperature > 0 for diversity.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if args.num_samples > 1 and args.temperature <= 0.0:
        logger.warning(
            "--num-samples=%d with temperature=%.3f will produce identical "
            "responses. Set --temperature > 0 for diversity.",
            args.num_samples, args.temperature,
        )

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    raw_data = load_questions(args.benchmark, args.cache_dir, args.num_questions)
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
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.num_samples,
    )

    prompts = []
    for item in raw_data:
        messages = build_messages(item["question"], args.benchmark, args.prompt_format)
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        # Start the thinking block so the model enters reasoning mode
        prompt += "<think>\n"
        prompts.append(prompt)

    outputs = llm.generate(prompts, sampling_params)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    extract_fn = extract_raw_reasoning_and_answer
    scorer = _BENCHMARK_REGISTRY[args.benchmark]["scorer"]

    with open(args.output, "w") as f:
        for item, output in zip(raw_data, outputs):
            for sample_idx, sample in enumerate(output.outputs):
                response_text = sample.text
                # Split at </think>: reasoning is inside the think block,
                # the visible part after </think> contains \boxed{answer}.
                thinking, visible = parse_think_output(response_text)
                if thinking is not None:
                    reasoning = thinking
                    _, answer = extract_fn(visible)
                else:
                    reasoning, answer = extract_fn(visible)

                is_correct = False
                if answer is not None:
                    score = scorer(answer, item["answer"])
                    is_correct = score["is_correct"]

                reasoning_token_count = len(
                    tokenizer.encode(reasoning or "", add_special_tokens=False)
                )

                # Suffix source_id so each sample is independently addressable
                # downstream (compress.py / select.py key by source_id).
                source_id = f"{item['source_id']}_s{sample_idx}"

                record = {
                    "benchmark": args.benchmark,
                    "source_id": source_id,
                    "question_source_id": item["source_id"],
                    "sample_index": sample_idx,
                    "question": item["question"],
                    "gold_answer": item["answer"],
                    "prompt_format": args.prompt_format,
                    "response_text": response_text.strip(),
                    "reasoning": reasoning,
                    "answer": answer,
                    "is_correct": is_correct,
                    "reasoning_token_count": reasoning_token_count,
                    "response_token_count": len(sample.token_ids),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info("Saved original CoTs to %s", args.output)


if __name__ == "__main__":
    main()
