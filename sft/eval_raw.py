"""SFT model evaluation with raw text prompt (paper-aligned).

Uses the same prompt format as the Self-Training paper:
  system: "Your task is to answer the question below..."
  user: "Question: {question}\nSolution:"

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
from sft.methods.self_training_concise.prompts import get_system_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Your task is to answer the question below. "
    "Give step by step reasoning before you answer, and when you're ready to answer, "
    "please use the format 'The answer is'"
)


def extract_numeric_answer(text: str) -> str | None:
    """Extract final numeric answer from 'The answer is ...' style text."""
    import re
    parts = text.lower().split("answer is")
    answer_flag = len(parts) > 1
    candidate_text = parts[-1] if answer_flag else text.lower()
    candidate_text = candidate_text.replace(",", "")
    matches = re.findall(r"-?\d+\.?\d*", candidate_text)
    if not matches:
        return None
    answer = matches[0] if answer_flag else matches[-1]
    if re.match(r"^-?\d+\.\d+$", answer):
        answer = answer.rstrip("0").rstrip(".")
    return answer


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT model with raw text prompt")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--cache-dir", type=str, default="./data")
    parser.add_argument("--output", type=str, default=None)
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

    # Paper's evaluation format (--prompt direct):
    # Raw text, no chat template, no system prompt.
    # For Qwen: just the raw question text (no BOS needed).
    # For other models: BOS + raw question text.
    is_qwen = "qwen" in type(tokenizer).__name__.lower()
    prompts = []
    for sample in test_samples:
        if is_qwen:
            prompt = sample["question"]
        else:
            prompt = tokenizer.bos_token + sample["question"]
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

        answer = extract_numeric_answer(prediction)
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
    print(f"Prompt: raw text (paper-aligned)")
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
            "prompt_format": "raw_text",
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
