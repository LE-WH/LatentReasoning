"""SFT model evaluation with raw text prompts.

Default mode uses the same prompt format as the Self-Training paper:
  system: "Your task is to answer the question below..."
  user: "Question: {question}\nSolution:"

When ``--tokenskip-prompt`` is enabled, the script switches to the
paper-faithful TokenSkip chat prompt used by the upstream Qwen evaluation.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m sft.eval_raw \
        --model Qwen/Qwen2.5-3B-Instruct
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python -m sft.eval_raw \
        --model results/sft/gsm8k/qwen2.5-3b-self-training-concise-rawtext
"""

import argparse
import json
import logging
import time
from pathlib import Path

from sft.compat import apply_patches
apply_patches()

from vllm import LLM, SamplingParams
from datasets import load_dataset
from ragen.env.static.utils import process_gsm8k, compute_score_numeric
from sft.methods.tokenskip.prompts import (
    build_tokenskip_raw_chat_messages,
    build_tokenskip_raw_question,
    extract_boxed_or_numeric_answer,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def build_tokenskip_eval_prompt(
    tokenizer,
    question: str,
    compression_ratio: float,
    *,
    model_family: str,
) -> str:
    """Build the upstream-style TokenSkip evaluation prompt."""
    family = (model_family or "qwen").lower()
    if family == "qwen":
        messages = build_tokenskip_raw_chat_messages(
            question,
            compression_ratio,
            model_family=family,
        )
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    if family == "llama3":
        user_content = build_tokenskip_raw_question(question, None, model_family=family)
        eos_token = tokenizer.eos_token or "<|eot_id|>"
        bos_token = tokenizer.bos_token or ""
        if compression_ratio < 1.0:
            return (
                f"{bos_token}<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_content}\n"
                f"{eos_token}{compression_ratio:g}{eos_token}{eos_token}"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        return (
            f"{bos_token}<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_content}\n"
            f"{eos_token}<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    raise ValueError(f"Unsupported TokenSkip model family: {model_family}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT model with raw text prompt")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--cache-dir", type=str, default="./data")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--tokenskip-prompt", action="store_true", default=False,
                        help="Use the TokenSkip raw prompt template")
    parser.add_argument("--compression-ratio", type=float, default=1.0,
                        help="Inline TokenSkip ratio conditioning for raw eval")
    parser.add_argument("--model-family", type=str, default="qwen", choices=["qwen", "llama3"])
    args = parser.parse_args()

    # Load test data
    ds = load_dataset("openai/gsm8k", name="main", split="test", cache_dir=args.cache_dir)
    test_samples = []
    for idx, item in enumerate(ds):
        if args.num_samples > 0 and idx >= args.num_samples:
            break
        question, answer = process_gsm8k(item)
        test_samples.append({"question": question, "answer": answer, "source_id": f"gsm8k_test_{idx}"})
    logger.info(f"Loaded {len(test_samples)} test samples")

    # Load model
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, trust_remote_code=True)

    llm = LLM(
        args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
        enable_sleep_mode=True,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    is_qwen = "qwen" in type(tokenizer).__name__.lower()
    prompts = []
    for sample in test_samples:
        if args.tokenskip_prompt:
            prompt = build_tokenskip_eval_prompt(
                tokenizer,
                sample["question"],
                args.compression_ratio,
                model_family=args.model_family,
            )
        else:
            prompt_body = sample["question"]
            if is_qwen:
                prompt = prompt_body
            else:
                prompt = tokenizer.bos_token + prompt_body
        prompts.append(prompt)

    logger.info(f"Generating {len(prompts)} responses...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    gen_time = time.time() - start_time
    logger.info(f"Generation done in {gen_time:.1f}s")

    # Score
    correct = 0
    total_tokens = 0
    correct_tokens = 0
    correct_count = 0
    results = []

    for sample, output in zip(test_samples, outputs):
        prediction = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)

        answer = extract_boxed_or_numeric_answer(prediction)
        if answer is not None:
            score = compute_score_numeric(answer, sample["answer"])
            is_correct = score["is_correct"]
        else:
            is_correct = False

        if is_correct:
            correct += 1
            correct_tokens += num_tokens
            correct_count += 1
        total_tokens += num_tokens

        results.append({
            "source_id": sample["source_id"],
            "prediction": prediction[:500],
            "num_tokens": num_tokens,
            "is_correct": is_correct,
            "extracted": answer,
        })

    accuracy = correct / len(test_samples) if test_samples else 0.0
    avg_tokens = total_tokens / len(test_samples) if test_samples else 0.0
    avg_correct_tokens = correct_tokens / correct_count if correct_count else 0.0

    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    prompt_format = (
        f"tokenskip_chat_template:{args.model_family}"
        if args.tokenskip_prompt
        else "raw_text"
    )
    print(f"Prompt: {prompt_format}")
    print(f"Samples: {len(test_samples)}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"Avg tokens (all): {avg_tokens:.1f}")
    print(f"Avg tokens (correct): {avg_correct_tokens:.1f}")
    print(f"Generation time: {gen_time:.1f}s")
    print(f"{'='*60}\n")

    # Save
    output_path = args.output or f"results/eval_rawtext/{Path(args.model).name}_eval.jsonl"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        summary = {
            "type": "summary",
            "model_path": args.model,
            "prompt_format": prompt_format,
            "tokenskip_prompt": args.tokenskip_prompt,
            "compression_ratio": args.compression_ratio,
            "num_samples": len(test_samples),
            "correct": correct,
            "accuracy": accuracy,
            "avg_tokens": avg_tokens,
            "avg_correct_tokens": avg_correct_tokens,
        }
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
