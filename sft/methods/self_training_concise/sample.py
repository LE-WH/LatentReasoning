"""CoT sampling for Concise SFT.

Generates multiple reasoning responses per question using vLLM, scores them,
and saves all results to an intermediate file.

Supports two modes:
  - Zero-shot (default)
  - Few-shot (--few-shot-path)
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

os.environ.setdefault("VLLM_USE_V1", "0")  # V0 engine needed for per-request logits processors

from sft.compat import apply_patches
apply_patches()

from vllm import LLM, SamplingParams

from datasets import load_dataset
from ragen.env.static.utils import (
    process_gsm8k, compute_score_numeric,
    process_math, compute_score_math,
)
from ragen.dual_vocab.utils import is_dual_model, load_meta
from ragen.dual_vocab.constraint import make_vllm_logits_processor
from .prompts import (
    DEFAULT_NUM_FEW_SHOT,
    _LATENT_RE,
    extract_think_answer,
    get_system_prompt,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_zero_shot_messages(
    question: str,
    system_prompt: str,
) -> list[dict]:
    """Build zero-shot prompt messages."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def build_few_shot_messages(
    question: str,
    system_prompt: str,
    exemplars: list[dict],
    num_shots: int = DEFAULT_NUM_FEW_SHOT,
) -> list[dict]:
    """Build few-shot messages with XML-format exemplars."""
    messages = [{"role": "system", "content": system_prompt}]

    selected = random.sample(exemplars, min(num_shots, len(exemplars)))

    for ex in selected:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({
            "role": "assistant",
            "content": f"<think>\n{ex['reasoning']}\n</think>\n<answer>{ex['answer']}</answer>",
        })

    messages.append({"role": "user", "content": question})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample CoT responses for Concise SFT")
    parser.add_argument("--benchmark", type=str, required=True,
                        choices=["gsm8k", "math"])
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--num-questions", type=int, default=-1,
                        help="Max questions to sample (-1 = all)")
    parser.add_argument("--samples-per-question", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max new tokens (512 for GSM8k, 1024 for MATH)")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default="./data")
    parser.add_argument("--few-shot-path", type=str, default=None,
                        help="Path to few-shot exemplars JSON (enables FS-BoN mode)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="Tensor parallel size for vLLM (number of GPUs)")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="0-based shard index for data-parallel sampling")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of data shards for sampling")
    parser.add_argument("--num-few-shot", type=int, default=DEFAULT_NUM_FEW_SHOT,
                        help=f"Number of exemplars per prompt (default: {DEFAULT_NUM_FEW_SHOT})")
    args = parser.parse_args()

    random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")
    if args.few_shot_path and args.max_model_len < 4096:
        logger.info(
            "Increasing max_model_len from %s to 4096 for few-shot prompts",
            args.max_model_len,
        )
        args.max_model_len = 4096

    mode = "few-shot" if args.few_shot_path else "zero-shot"
    shard_suffix = (
        ""
        if args.num_shards == 1
        else f"_shard{args.shard_id:02d}of{args.num_shards:02d}"
    )
    output_path = args.output or (
        f"data/sft/cot_samples/{args.benchmark}_cot_{mode}{shard_suffix}.jsonl"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # --- Load few-shot exemplars if provided ---
    exemplars = None
    if args.few_shot_path:
        with open(args.few_shot_path) as f:
            exemplars = json.load(f)
        exemplar_benchmarks = {ex.get("benchmark") for ex in exemplars if ex.get("benchmark")}
        if exemplar_benchmarks and exemplar_benchmarks != {args.benchmark}:
            raise ValueError(
                f"Few-shot exemplars benchmark mismatch: expected {args.benchmark}, got {sorted(exemplar_benchmarks)}"
            )
        logger.info(f"Loaded {len(exemplars)} few-shot exemplars from {args.few_shot_path}")

    # --- Load benchmark data ---
    BENCHMARK_DATASETS = {
        "gsm8k": {"path": "openai/gsm8k", "name": "main", "processor": process_gsm8k, "scorer": compute_score_numeric},
        "math": {"path": "nlile/hendrycks-MATH-benchmark", "name": None, "processor": process_math, "scorer": compute_score_math},
    }
    bench_cfg = BENCHMARK_DATASETS[args.benchmark]
    ds_kwargs = {"path": bench_cfg["path"], "split": "train", "cache_dir": args.cache_dir}
    if bench_cfg["name"]:
        ds_kwargs["name"] = bench_cfg["name"]
    ds = load_dataset(**ds_kwargs)
    scorer = bench_cfg["scorer"]
    raw_data = []
    for idx, item in enumerate(ds):
        if args.num_questions > 0 and idx >= args.num_questions:
            break
        question, answer = bench_cfg["processor"](item)
        raw_data.append({
            "question": question,
            "answer": answer,
            "source_id": f"{args.benchmark}_train_{idx}",
        })
    logger.info(f"Loaded {len(raw_data)} questions from {args.benchmark}")
    if args.num_shards > 1:
        raw_data = raw_data[args.shard_id::args.num_shards]
        logger.info(
            f"Sampling shard {args.shard_id + 1}/{args.num_shards}: {len(raw_data)} questions"
        )

    # --- Load model ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.cache_dir, trust_remote_code=True
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

    # --- Dual-vocab logits constraint (if applicable) ---
    logits_processors = None
    if is_dual_model(args.model):
        dual_meta = load_meta(args.model)
        logits_processors = [make_vllm_logits_processor(
            V=dual_meta["V"],
            think_end_id=dual_meta["think_end_token_id"],
            eos_id=tokenizer.eos_token_id,
        )]
        logger.info(
            "Dual-vocab model detected — logits constraint enabled "
            "(V=%d, think_end_id=%d)", dual_meta["V"], dual_meta["think_end_token_id"],
        )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        n=args.samples_per_question,
        logits_processors=logits_processors,
    )

    # --- Build prompts ---
    system_prompt = get_system_prompt(args.benchmark)
    prompts = []
    for item in raw_data:
        if exemplars is not None:
            # Few-shot mode: inject exemplars as conversation turns
            messages = build_few_shot_messages(
                item["question"], system_prompt, exemplars,
                num_shots=args.num_few_shot,
            )
        else:
            # Zero-shot mode
            messages = build_zero_shot_messages(
                item["question"], system_prompt
            )
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    logger.info(
        f"[{mode}] Generating {len(prompts)} x {args.samples_per_question} = "
        f"{len(prompts) * args.samples_per_question} responses..."
    )

    # --- Generate ---
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - start_time
    logger.info(f"Generation done in {gen_time:.1f}s")

    # --- Score and save ---
    total_samples = 0
    total_correct = 0
    questions_with_correct = 0

    with open(output_path, "w") as f:
        for item, output in zip(raw_data, outputs):
            question_correct = 0
            sample_results = []
            seen_thinks = set()  # For deduplication

            for k, completion in enumerate(output.outputs):
                text = completion.text
                think, answer = extract_think_answer(text)

                # Deduplication: skip if we've seen this exact think text
                think_key = think.strip() if think else None
                if think_key is not None and think_key in seen_thinks:
                    continue
                if think_key is not None:
                    seen_thinks.add(think_key)

                # Score: compare extracted answer with gold answer
                if answer is not None:
                    score = scorer(answer, item["answer"])
                    is_correct = score["is_correct"]
                else:
                    is_correct = False

                if is_correct:
                    question_correct += 1
                    total_correct += 1

                # Token count used for shortest-response selection.
                # For dual-vocab models, count latent tokens in raw text
                # (those ARE the thinking); otherwise count visible think text.
                latent_count = len(_LATENT_RE.findall(text))
                if latent_count > 0:
                    think_token_count = latent_count
                else:
                    think_token_count = (
                        len(tokenizer.encode(think, add_special_tokens=False))
                        if think else 0
                    )

                sample_results.append({
                    "sample_idx": k,
                    "text": text,
                    "think": think,
                    "answer": answer,
                    "is_correct": is_correct,
                    "think_length": len(think) if think else 0,
                    "think_token_count": think_token_count,
                    "response_text": text.strip(),
                })
                total_samples += 1

            if question_correct > 0:
                questions_with_correct += 1

            record = {
                "benchmark": args.benchmark,
                "source_id": item["source_id"],
                "question": item["question"],
                "gold_answer": item["answer"],
                "num_correct": question_correct,
                "num_samples": len(sample_results),
                "samples": sample_results,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # --- Summary ---
    logger.info(f"Results saved to {output_path}")
    if len(raw_data) == 0 or total_samples == 0:
        logger.warning("No data was processed -- nothing to summarize.")
        return
    logger.info(
        f"Questions: {len(raw_data)}, "
        f"with >=1 correct: {questions_with_correct} "
        f"({questions_with_correct/len(raw_data)*100:.1f}%)"
    )
    logger.info(
        f"Total samples (after dedup): {total_samples}, "
        f"correct: {total_correct} "
        f"({total_correct/total_samples*100:.1f}%)"
    )
    logger.info(f"Generation time: {gen_time:.1f}s")


if __name__ == "__main__":
    main()
