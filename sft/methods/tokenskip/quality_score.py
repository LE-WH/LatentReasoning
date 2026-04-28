"""Score compressed CoTs by P(real_answer | prompt + compressed_reasoning).

For each compressed record we build the same chat the SFT trainer/evaluator
builds, append the gold answer in the boxed form used at training time, and
compute the sum log-probability of the answer tokens via a single forward
pass with HuggingFace transformers (memory-efficient: realized-token
log-probs only, never materializing the full vocab dimension as a softmax).

The output is one JSONL row per scored record, keyed by ``source_id``. A
separate filter step (``quality_filter.py``) consumes the score file and emits
top-K-per-question subsets without re-running the LM.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _tokens_overlapping(
    offsets: list[tuple[int, int]], char_start: int, char_end: int
) -> list[int]:
    """Return token indices whose char span overlaps [char_start, char_end).

    Tokens with a (0, 0) span (special tokens added by the chat template) are
    skipped.
    """
    out = []
    for i, (s, e) in enumerate(offsets):
        if s == 0 and e == 0:
            continue
        if s < char_end and e > char_start:
            out.append(i)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score compressed CoTs by P(answer | prompt+cot)")
    parser.add_argument("--input", type=str, required=True,
                        help="compressed_ratio_R.jsonl produced by compress.py")
    parser.add_argument("--output", type=str, required=True,
                        help="JSONL output path for per-record scores")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Thinking-2507",
                        help="Scoring model (defaults to the SFT base model)")
    parser.add_argument("--cache-dir", type=str, default="./data")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--max-seq-len", type=int, default=4096,
                        help="Skip records whose tokenized length exceeds this")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Records per forward pass")
    parser.add_argument("--limit", type=int, default=-1,
                        help="If >0, only score the first N records (for smoke tests)")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--model-family", type=str, default="qwen", choices=["qwen", "llama3"])
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sft.methods.tokenskip.prompts import (
        build_tokenskip_raw_question,
        get_raw_chat_system_prompt,
    )

    records = _load_jsonl(args.input)
    if args.num_shards > 1:
        records = records[args.shard_id::args.num_shards]
    if args.limit > 0:
        records = records[: args.limit]
    logger.info("Loaded %d compressed records (shard %d/%d, limit=%s)",
                len(records), args.shard_id, args.num_shards, args.limit)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.cache_dir, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        # Qwen tokenizer ships with eos but no pad — reuse eos for padding.
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model %s ...", args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda").eval()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build per-record (token_ids, answer-token-positions, boxed-token-positions).
    prepared: list[dict] = []
    skipped_long = 0
    skipped_no_box = 0
    for record in records:
        question = record["question"]
        cot = record["compressed_reasoning"]
        answer = str(record["answer"])
        ratio = record["compression_ratio"]

        user_content = build_tokenskip_raw_question(
            question, ratio, model_family=args.model_family
        )
        assistant_content = (
            f"{cot}\n\n</think>\n\nThe final answer is: $\\boxed{{{answer}}}$"
        )
        messages = [
            {"role": "system", "content": get_raw_chat_system_prompt()},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        boxed_open_marker = "\\boxed{"
        box_open_char = full_text.rfind(boxed_open_marker)
        if box_open_char < 0:
            skipped_no_box += 1
            continue
        answer_start_char = box_open_char + len(boxed_open_marker)
        answer_end_char = answer_start_char + len(answer)
        boxed_end_char = answer_end_char + 1  # include closing }

        # Post-CoT region = everything after the compressed reasoning ends in
        # the assistant turn. We anchor on the closing "</think>" tag itself
        # (the chat template can collapse the leading newlines around it).
        post_cot_marker = "</think>"
        post_cot_char = full_text.rfind(post_cot_marker)
        if post_cot_char < 0:
            skipped_no_box += 1
            continue
        # End-of-assistant-content = the position right after the closing $.
        # We use the boxed_end_char + 1 (to include the trailing $) as the
        # right bound. Any chat-template tokens after that are excluded so
        # the metric reflects only the response text the model would emit.
        post_cot_end_char = boxed_end_char + 1

        encoded = tokenizer(
            full_text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]

        if len(input_ids) > args.max_seq_len:
            skipped_long += 1
            continue

        answer_tok_idxs = _tokens_overlapping(offsets, answer_start_char, answer_end_char)
        boxed_tok_idxs = _tokens_overlapping(offsets, box_open_char, boxed_end_char)
        post_cot_tok_idxs = _tokens_overlapping(offsets, post_cot_char, post_cot_end_char)

        prepared.append({
            "record": record,
            "input_ids": input_ids,
            "answer_tok_idxs": answer_tok_idxs,
            "boxed_tok_idxs": boxed_tok_idxs,
            "post_cot_tok_idxs": post_cot_tok_idxs,
        })

    logger.info("Prepared %d records (skipped_long=%d, skipped_no_box=%d)",
                len(prepared), skipped_long, skipped_no_box)

    pad_id = tokenizer.pad_token_id
    written = 0
    with open(out_path, "w") as f, torch.inference_mode():
        for batch_start in range(0, len(prepared), args.batch_size):
            batch = prepared[batch_start : batch_start + args.batch_size]
            max_len = max(len(b["input_ids"]) for b in batch)
            input_ids = torch.full(
                (len(batch), max_len), pad_id, dtype=torch.long
            )
            attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
            for j, b in enumerate(batch):
                L = len(b["input_ids"])
                input_ids[j, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
                attention_mask[j, :L] = 1

            input_ids = input_ids.to("cuda")
            attention_mask = attention_mask.to("cuda")

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits  # (B, T, V) bf16

            # logits[:, t-1, :] predicts token at position t. Compute realized
            # token logprob via realized_logit - logsumexp(all_logits) without
            # materializing the full softmax.
            shifted_logits = logits[:, :-1, :]  # (B, T-1, V) bf16
            shifted_targets = input_ids[:, 1:]  # (B, T-1)
            # Gather realized-token logits first (tiny tensor) before any fp32
            # promotion. Then chunk the logsumexp along the time axis to avoid
            # materializing a fp32 copy of the full (B, T-1, V) tensor.
            realized = shifted_logits.gather(
                2, shifted_targets.unsqueeze(-1)
            ).squeeze(-1).float()  # (B, T-1) fp32
            B, Tm1, _V = shifted_logits.shape
            chunk = 256
            log_z = torch.empty((B, Tm1), device=shifted_logits.device, dtype=torch.float32)
            for s in range(0, Tm1, chunk):
                e = min(s + chunk, Tm1)
                log_z[:, s:e] = torch.logsumexp(
                    shifted_logits[:, s:e, :].float(), dim=-1
                )
            tok_lp = realized - log_z  # (B, T-1) fp32
            # Free large intermediates before the per-record gather loop.
            del shifted_logits, realized, log_z, logits

            for j, b in enumerate(batch):
                # answer/boxed token positions are in [0, T) of the original
                # token sequence. Logprob of token at position i is tok_lp[j, i-1]
                # (we drop position 0 since there is no preceding context).
                ans_idxs = [i for i in b["answer_tok_idxs"] if i >= 1]
                box_idxs = [i for i in b["boxed_tok_idxs"] if i >= 1]
                pc_idxs  = [i for i in b["post_cot_tok_idxs"] if i >= 1]
                ans_lp = tok_lp[j, [i - 1 for i in ans_idxs]].sum().item() if ans_idxs else 0.0
                box_lp = tok_lp[j, [i - 1 for i in box_idxs]].sum().item() if box_idxs else 0.0
                pc_lp  = tok_lp[j, [i - 1 for i in pc_idxs]].sum().item() if pc_idxs else 0.0

                rec = b["record"]
                source_id = rec.get("source_id", "")
                qsid = source_id.rsplit("_s", 1)[0] if "_s" in source_id else source_id
                f.write(json.dumps({
                    "source_id": source_id,
                    "question_source_id": qsid,
                    "compression_ratio": rec.get("compression_ratio"),
                    "answer": str(rec.get("answer", "")),
                    "logprob_answer": ans_lp,
                    "num_answer_tokens": len(ans_idxs),
                    "logprob_boxed": box_lp,
                    "num_boxed_tokens": len(box_idxs),
                    "logprob_post_cot": pc_lp,
                    "num_post_cot_tokens": len(pc_idxs),
                    "input_token_count": len(b["input_ids"]),
                    "skipped": False,
                }) + "\n")
                written += 1

            if (batch_start // args.batch_size) % 25 == 0:
                logger.info("Scored %d / %d", written, len(prepared))

    logger.info("Done: wrote %d scored rows to %s", written, out_path)


if __name__ == "__main__":
    main()
