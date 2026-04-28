"""Re-run a small list of MATH-500 source_ids through Phase-1 modes with full output."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from sft.compat import apply_patches

apply_patches()

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from ragen.env.static.utils import process_math
from sft.methods.sot.prompts import build_sot_eval_prompt
from sft.methods.tokenskip.prompts import build_tokenskip_raw_chat_messages


SRC_IDS = ["math_test_363", "math_test_56", "math_test_297"]


def build_vanilla_prompt(tokenizer, question: str) -> str:
    msgs = build_tokenskip_raw_chat_messages(question, 1.0, model_family="qwen")
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--cache-dir", default="./data")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    ds = load_dataset(
        path="nlile/hendrycks-MATH-benchmark",
        split="test",
        cache_dir=args.cache_dir,
    )

    pick: dict[str, dict] = {}
    for idx, item in enumerate(ds):
        sid = f"math_test_{idx}"
        if sid in SRC_IDS:
            q, a = process_math(item)
            pick[sid] = {"source_id": sid, "question": q, "gold_answer": a}
        if len(pick) == len(SRC_IDS):
            break
    if len(pick) != len(SRC_IDS):
        missing = set(SRC_IDS) - set(pick)
        raise RuntimeError(f"Could not locate source_ids: {missing}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, trust_remote_code=True)
    llm = LLM(
        args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        enable_prefix_caching=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    runs = []
    # Build all prompts up front so vLLM batches them.
    prompts: list[tuple[str, str, str]] = []  # (mode, source_id, prompt)
    for sid in SRC_IDS:
        q = pick[sid]["question"]
        prompts.append(("vanilla", sid, build_vanilla_prompt(tokenizer, q)))
        prompts.append(("sot_on", sid, build_sot_eval_prompt(tokenizer, q, suppress_thinking=False)))
        prompts.append(("sot_suppress", sid, build_sot_eval_prompt(tokenizer, q, suppress_thinking=True)))

    outputs = llm.generate([p[2] for p in prompts], sp)
    for (mode, sid, prompt), out in zip(prompts, outputs):
        gen_text = out.outputs[0].text
        gen_ids = list(out.outputs[0].token_ids)
        runs.append({
            "source_id": sid,
            "mode": mode,
            "question": pick[sid]["question"],
            "gold_answer": pick[sid]["gold_answer"],
            "prompt": prompt,
            "generation": gen_text,
            "num_generated_tokens": len(gen_ids),
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(runs, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(runs)} captures to {args.output}")


if __name__ == "__main__":
    main()
