"""Collect SoT (Sketch-of-Thought) teacher trajectories.

Mirrors ``sft/methods/tokenskip/collect.py`` but uses the multi-turn SoT
prompt (see ``sft.methods.sot.prompts.build_sot_multiturn_prompt``) to
produce the teacher's `<think>...</think>` plus `\\boxed{...}` answer.

For each train question, generate ``--num-samples`` responses at
``--temperature``. Each sample is graded with the benchmark scorer; the
selection step downstream (``sft/methods/sot/select.py``) keeps the top-K
shortest correct samples per question.

Output JSONL row:
    {benchmark, source_id (`{bench}_train_{i}_s{j}`),
     question_source_id, sample_index, question, gold_answer,
     response_text, reasoning, answer, is_correct,
     reasoning_token_count, response_token_count}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

from sft.compat import apply_patches

apply_patches()

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ragen.env.static.utils import (
    compute_score_math,
    compute_score_numeric,
    process_gsm8k,
    process_math,
)
from sft.methods.sot.prompts import build_sot_multiturn_prompt
from sft.methods.tokenskip.collect import parse_think_output
from sft.methods.tokenskip.prompts import extract_raw_reasoning_and_answer

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
    reg = _BENCHMARK_REGISTRY[benchmark]
    load_kwargs: dict = {"path": reg["path"], "split": reg["split"], "cache_dir": cache_dir}
    if reg["name"]:
        load_kwargs["name"] = reg["name"]
    ds = load_dataset(**load_kwargs)
    processor = reg["processor"]
    out = []
    for idx, item in enumerate(ds):
        if num_questions > 0 and idx >= num_questions:
            break
        q, a = processor(item)
        out.append({"question": q, "answer": a, "source_id": f"{benchmark}_train_{idx}"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect SoT teacher samples")
    parser.add_argument("--benchmark", required=True, choices=list(_BENCHMARK_REGISTRY.keys()))
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--cache-dir", default="./data")
    parser.add_argument("--paradigm", default="chunked_symbolism", choices=["chunked_symbolism"])
    parser.add_argument("--num-questions", type=int, default=-1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=6144)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    args = parser.parse_args()

    # Use the shell's CUDA_VISIBLE_DEVICES; do not override here.
    raw = load_questions(args.benchmark, args.cache_dir, args.num_questions)
    if args.num_shards > 1:
        raw = raw[args.shard_id::args.num_shards]
    logger.info("shard %d/%d: %d questions", args.shard_id, args.num_shards, len(raw))

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, trust_remote_code=True)

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

    prompts = [
        build_sot_multiturn_prompt(item["question"], paradigm=args.paradigm)
        for item in raw
    ]
    logger.info("Generating %d prompts × %d samples ...", len(prompts), args.num_samples)
    outputs = llm.generate(prompts, sampling_params)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    scorer = _BENCHMARK_REGISTRY[args.benchmark]["scorer"]
    n_correct = 0
    n_total = 0
    with open(args.output, "w") as f:
        for item, output in zip(raw, outputs):
            for sample_idx, sample in enumerate(output.outputs):
                response_text = sample.text
                thinking, visible = parse_think_output(response_text)
                if thinking is not None:
                    reasoning = thinking
                    _, answer = extract_raw_reasoning_and_answer(visible)
                else:
                    reasoning, answer = extract_raw_reasoning_and_answer(visible)

                is_correct = False
                if answer is not None:
                    score = scorer(answer, item["answer"])
                    is_correct = score["is_correct"]
                if is_correct:
                    n_correct += 1
                n_total += 1

                reasoning_tok = len(tokenizer.encode(reasoning or "", add_special_tokens=False))

                rec = {
                    "benchmark": args.benchmark,
                    "source_id": f"{item['source_id']}_s{sample_idx}",
                    "question_source_id": item["source_id"],
                    "sample_index": sample_idx,
                    "question": item["question"],
                    "gold_answer": item["answer"],
                    "response_text": response_text.strip(),
                    "reasoning": reasoning,
                    "answer": answer,
                    "is_correct": is_correct,
                    "reasoning_token_count": reasoning_tok,
                    "response_token_count": len(sample.token_ids),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("shard %d done: %d/%d correct (%.1f%%) → %s",
                args.shard_id, n_correct, n_total,
                100*n_correct/n_total if n_total else 0.0, args.output)


if __name__ == "__main__":
    main()
