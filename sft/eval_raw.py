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
from ragen.env.static.utils import (
    compute_score_math,
    compute_score_numeric,
    process_gsm8k,
    process_math,
)
from sft.methods.tokenskip.prompts import (
    build_tokenskip_raw_chat_messages,
    build_tokenskip_raw_question,
    extract_boxed_or_numeric_answer,
)

_EVAL_BENCHMARKS: dict[str, dict] = {
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "test",
        "processor": process_gsm8k,
        "scorer": compute_score_numeric,
    },
    "math": {
        "path": "nlile/hendrycks-MATH-benchmark",
        "name": None,
        "split": "test",
        "processor": process_math,
        "scorer": compute_score_math,
    },
}

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
    parser.add_argument("--benchmark", type=str, default="gsm8k",
                        choices=list(_EVAL_BENCHMARKS.keys()),
                        help="Benchmark to evaluate on")
    parser.add_argument("--num-samples", type=int, default=-1)
    parser.add_argument("--cache-dir", type=str, default="./data")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--tokenskip-prompt", action="store_true", default=False,
                        help="Use the TokenSkip raw prompt template")
    parser.add_argument("--compression-ratio", type=float, default=1.0,
                        help="Inline TokenSkip ratio conditioning for raw eval")
    parser.add_argument("--model-family", type=str, default="qwen", choices=["qwen", "llama3"])
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="W&B project name. If set, logs metrics and sample CoTs to wandb.")
    parser.add_argument("--wandb-run-name", type=str, default=None,
                        help="W&B run name (auto-generated if omitted)")
    parser.add_argument("--num-cot-samples", type=int, default=20,
                        help="Number of sample CoTs to log to wandb table")
    args = parser.parse_args()

    # Load test data
    bench = _EVAL_BENCHMARKS[args.benchmark]
    load_kwargs: dict = {"path": bench["path"], "split": bench["split"], "cache_dir": args.cache_dir}
    if bench["name"]:
        load_kwargs["name"] = bench["name"]
    ds = load_dataset(**load_kwargs)
    processor = bench["processor"]
    scorer = bench["scorer"]
    test_samples = []
    for idx, item in enumerate(ds):
        if args.num_samples > 0 and idx >= args.num_samples:
            break
        question, answer = processor(item)
        test_samples.append({"question": question, "answer": answer, "source_id": f"{args.benchmark}_test_{idx}"})
    logger.info(f"Loaded {len(test_samples)} test samples from {args.benchmark}")

    # Load model
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir, trust_remote_code=True)

    # Detect dual-vocab model
    import os
    from ragen.dual_vocab.utils import is_dual_model, load_meta
    dual_meta = None
    if is_dual_model(args.model):
        dual_meta = load_meta(args.model)
        logger.info(f"Dual-vocab model detected: V={dual_meta['V']}, L={dual_meta['L']}")
        # vLLM V1 does not support per-request logits processors; fall back to V0.
        # V0 has a blanket guard that rejects logits_processors (line 673 of
        # llm_engine.py).  The guard is overly broad — the processors work fine
        # when multi-step is effectively 1.  Patch the check away so our
        # dual-vocab constraint can be passed through SamplingParams.
        os.environ["VLLM_USE_V1"] = "0"
        # Patch out the overly-broad guard (line 673-676 of llm_engine.py)
        # that rejects logits_processors in multi-step decoding.  Multi-step
        # is effectively 1 here, so the processors work fine.
        #
        # The guard sits in add_request (L673) *before*
        # _create_sequence_group_with_sampling which calls
        # _build_logits_processors (L723) and clone() (L728).
        # We hide the processors from the guard, then re-inject them in
        # _create_sequence_group_with_sampling before _build_logits_processors
        # merges them in.
        import vllm.engine.llm_engine as _eng
        from vllm import SamplingParams as _SP
        _stashed_lp = {}  # request_id → logits_processors

        _orig_add_request = _eng.LLMEngine.add_request

        def _add_request_allow_lp(self, request_id, prompt, params, *args, **kwargs):
            if isinstance(params, _SP) and params.logits_processors:
                lp = list(params.logits_processors)
                _stashed_lp[request_id] = lp
                params.logits_processors = None  # hide from guard
                result = _orig_add_request(self, request_id, prompt, params, *args, **kwargs)
                params.logits_processors = lp  # restore for next request
                return result
            return _orig_add_request(self, request_id, prompt, params, *args, **kwargs)
        _eng.LLMEngine.add_request = _add_request_allow_lp

        _orig_create_sg = _eng.LLMEngine._create_sequence_group_with_sampling
        def _create_sg_with_lp(self, request_id, seq, sampling_params, **kwargs):
            if request_id in _stashed_lp:
                sampling_params.logits_processors = _stashed_lp.pop(request_id)
            return _orig_create_sg(self, request_id, seq, sampling_params, **kwargs)
        _eng.LLMEngine._create_sequence_group_with_sampling = _create_sg_with_lp

    llm = LLM(
        args.model,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=8192,
        enable_sleep_mode=True,
        enforce_eager=True,
        disable_custom_all_reduce=True,
        enable_prefix_caching=True,
    )
    max_tokens = 4096 if args.benchmark == "math" else 512
    logits_processors = []
    if dual_meta is not None:
        from ragen.dual_vocab.constraint import make_vllm_logits_processor
        logits_processors.append(
            make_vllm_logits_processor(
                V=dual_meta["V"],
                think_end_id=dual_meta["think_end_token_id"],
                eos_id=tokenizer.eos_token_id,
            )
        )
        logger.info("Using dual-vocab logits constraint for generation")
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=max_tokens, logits_processors=logits_processors or None,
    )

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
    total_think_tokens = 0
    correct_tokens = 0
    correct_count = 0
    results = []

    for sample, output in zip(test_samples, outputs):
        output_ids = list(output.outputs[0].token_ids)
        num_tokens = len(output_ids)

        # For dual-vocab models, map latent IDs back to visible for decoding
        num_latent_ids = 0
        num_visible_ids = 0
        if dual_meta is not None:
            V = dual_meta["V"]
            think_end_id = dual_meta["think_end_token_id"]
            readable_ids = []
            for tid in output_ids:
                if V <= tid < V + dual_meta["L"]:
                    readable_ids.append(tid - V)  # latent → visible
                    num_latent_ids += 1
                else:
                    readable_ids.append(tid)
                    num_visible_ids += 1
            prediction = tokenizer.decode(readable_ids, skip_special_tokens=False)
        else:
            prediction = output.outputs[0].text

        # Split at </think>: reasoning tokens before, visible answer after.
        think_tokens = num_tokens
        visible_part = prediction
        think_idx = prediction.find("</think>")
        if think_idx != -1:
            thinking_text = prediction[:think_idx]
            visible_part = prediction[think_idx + len("</think>"):].strip()
            think_tokens = len(tokenizer.encode(thinking_text, add_special_tokens=False))

        answer = extract_boxed_or_numeric_answer(visible_part)
        if answer is not None:
            score = scorer(answer, sample["answer"])
            is_correct = score["is_correct"]
        else:
            is_correct = False

        if is_correct:
            correct += 1
            correct_tokens += num_tokens
            correct_count += 1
        total_tokens += num_tokens
        total_think_tokens += think_tokens

        result_entry = {
            "source_id": sample["source_id"],
            "prediction": visible_part[:500],
            "thinking": prediction[:think_idx][:500] if think_idx != -1 else "",
            "num_tokens": num_tokens,
            "think_tokens": think_tokens,
            "is_correct": is_correct,
            "extracted": answer,
        }
        if dual_meta is not None:
            result_entry["num_latent_ids"] = num_latent_ids
            result_entry["num_visible_ids"] = num_visible_ids
            # Store first 50 raw token IDs so latent range is visible
            result_entry["raw_token_ids_head"] = output_ids[:50]
            # Store the boundary region around </think> if present
            if think_end_id in output_ids:
                te_pos = output_ids.index(think_end_id)
                start = max(0, te_pos - 5)
                end = min(len(output_ids), te_pos + 10)
                result_entry["raw_ids_around_think_end"] = {
                    "offset": start,
                    "ids": output_ids[start:end],
                    "think_end_pos": te_pos,
                }
            logger.info(
                f"  [{sample['source_id']}] tokens={num_tokens} "
                f"latent={num_latent_ids} visible={num_visible_ids} "
                f"correct={is_correct}"
            )
        results.append(result_entry)

    accuracy = correct / len(test_samples) if test_samples else 0.0
    avg_tokens = total_tokens / len(test_samples) if test_samples else 0.0
    avg_think_tokens = total_think_tokens / len(test_samples) if test_samples else 0.0
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
    print(f"Avg thinking tokens: {avg_think_tokens:.1f}")
    print(f"Avg tokens (correct): {avg_correct_tokens:.1f}")
    print(f"Generation time: {gen_time:.1f}s")
    if dual_meta is not None:
        total_latent = sum(r.get("num_latent_ids", 0) for r in results)
        total_visible = sum(r.get("num_visible_ids", 0) for r in results)
        pct = total_latent / (total_latent + total_visible) * 100 if (total_latent + total_visible) else 0
        print(f"Latent tokens: {total_latent} ({pct:.1f}% of all generated)")
        print(f"Visible tokens: {total_visible}")
    print(f"{'='*60}\n")

    # --- wandb logging ---
    if args.wandb_project:
        import wandb

        run_name = args.wandb_run_name or (
            f"{Path(args.model).name}_{args.benchmark}_ratio{args.compression_ratio:g}"
        )
        wandb.init(project=args.wandb_project, name=run_name, config={
            "model": args.model,
            "benchmark": args.benchmark,
            "compression_ratio": args.compression_ratio,
            "num_samples": len(test_samples),
            "tokenskip_prompt": args.tokenskip_prompt,
            "model_family": args.model_family,
        })

        # Summary metrics
        wandb.log({
            "eval/accuracy": accuracy,
            "eval/num_correct": correct,
            "eval/num_samples": len(test_samples),
            "eval/avg_tokens": avg_tokens,
            "eval/avg_thinking_tokens": avg_think_tokens,
            "eval/avg_tokens_correct": avg_correct_tokens,
            "eval/generation_time_s": gen_time,
        })

        # Thinking-length histogram
        think_token_counts = [r["think_tokens"] for r in results]
        wandb.log({
            "eval/thinking_length_distribution": wandb.Histogram(think_token_counts),
        })

        # Sample CoT table
        num_to_log = min(args.num_cot_samples, len(results))
        columns = ["source_id", "question", "gold_answer", "thinking", "answer_part",
                    "extracted_answer", "is_correct", "think_tokens", "total_tokens"]
        table = wandb.Table(columns=columns)
        for i in range(num_to_log):
            r = results[i]
            s = test_samples[i]
            table.add_data(
                r["source_id"],
                s["question"][:500],
                s["answer"],
                r.get("thinking", ""),
                r["prediction"],
                r.get("extracted"),
                r["is_correct"],
                r["think_tokens"],
                r["num_tokens"],
            )
        wandb.log({"eval/sample_cots": table})
        wandb.finish()
        logger.info("Logged results to wandb project=%s run=%s", args.wandb_project, run_name)

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
            "avg_thinking_tokens": avg_think_tokens,
            "avg_correct_tokens": avg_correct_tokens,
        }
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
