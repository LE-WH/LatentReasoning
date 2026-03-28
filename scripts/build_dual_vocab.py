"""
scripts/build_dual_vocab.py
===========================
Build a dual-vocabulary tokenizer for LatentReasoning.

Creates:
  <out_dir>/tokenizer_config.json  + all tokenizer files
  <out_dir>/dual_vocab_meta.json   — metadata used by expand_model_to_dual_vocab.py

Vocab layout:
  [0,   V)   visible vocab V  (base model + optional think-boundary tokens)
  [V,   V+L) latent vocab  L  (for silent CoT reasoning)

Think-boundary token selection
-------------------------------
Searched in order; first existing single-token match wins.
  start: <think>  <thinking>  [THINK]  <|think|>
  end  : </think> </thinking> [/THINK] <|/think|>
Pass --think_missing add to auto-add the first candidate if none exist (random init).
Pass --think_missing clone_eos to auto-add and init from the model's EOS-like token
  (e.g. <|im_end|> for Qwen, <|eot_id|> for Llama-3) with subtle noise.

Latent vocab modes
------------------
  full_copy  (default): L = V, copies every visible token
  subset              : filtered slice; use --subset_regex or --subset_ids_file
  custom              : arbitrary tokens from --custom_tokens_file

Usage
-----
# Full copy (default)
python scripts/build_dual_vocab.py \\
    --base_model Qwen/Qwen2.5-3B-Instruct \\
    --out_dir ./checkpoints/dual_tokenizer

# Add think tokens if missing
python scripts/build_dual_vocab.py \\
    --base_model Qwen/Qwen2.5-3B-Instruct \\
    --out_dir ./checkpoints/dual_tokenizer \\
    --think_missing add

# Subset: only alpha tokens as latent
python scripts/build_dual_vocab.py \\
    --base_model Qwen/Qwen2.5-3B-Instruct \\
    --out_dir ./checkpoints/dual_tokenizer_subset \\
    --mode subset --subset_regex "^[a-zA-Z]"
"""

import sys
import os

# Allow running from repo root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re
import json
import argparse
import numpy as np
from typing import List, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_THINK_START_CANDIDATES = ["<think>", "<thinking>", "[THINK]", "<|think|>"]
DEFAULT_THINK_END_CANDIDATES = ["</think>", "</thinking>", "[/THINK]", "<|/think|>"]

# Ordered by preference: model-specific end-of-turn tokens, then generic EOS.
# The first one found in the tokenizer's vocab is used as the clone source for
# think-boundary tokens when think_missing="clone_eos".
EOS_CLONE_CANDIDATES = [
    "<|im_end|>",       # Qwen family
    "<|eot_id|>",       # Llama-3 family
    "<|end|>",          # Phi-3 family
    "</s>",             # LLaMA-2 / Mistral / general sentencepiece
]


def _is_single_token(tok, token_str: str) -> Optional[int]:
    vocab = tok.get_vocab()
    if token_str not in vocab:
        return None
    ids = tok.encode(token_str, add_special_tokens=False)
    if len(ids) == 1 and ids[0] == vocab[token_str]:
        return ids[0]
    return None


def _find_eos_clone_source(tok) -> Tuple[str, int]:
    """Find the best end-of-turn token to clone from for think-boundary init."""
    for candidate in EOS_CLONE_CANDIDATES:
        tid = _is_single_token(tok, candidate)
        if tid is not None:
            return candidate, tid
    # Last resort: use the tokenizer's own eos_token
    if tok.eos_token is not None:
        tid = _is_single_token(tok, tok.eos_token)
        if tid is not None:
            return tok.eos_token, tid
    raise ValueError(
        "Cannot find any EOS-like token to clone from.\n"
        f"Searched: {EOS_CLONE_CANDIDATES} + tokenizer.eos_token={tok.eos_token}"
    )


def resolve_think_token(
    tok,
    candidates: List[str],
    role: str,
    if_missing: str = "error",
) -> Tuple[str, int, bool]:
    for candidate in candidates:
        tid = _is_single_token(tok, candidate)
        if tid is not None:
            print(f"  Think-{role} token: '{candidate}' (id={tid}) [already in vocab]")
            return candidate, tid, False

    if if_missing == "error":
        raise ValueError(
            f"No think-{role} token found in the tokenizer.\n"
            f"Searched: {candidates}\n"
            f"Run with --think_missing add or clone_eos to automatically add '{candidates[0]}'."
        )

    token_to_add = candidates[0]
    n = tok.add_tokens([token_to_add], special_tokens=True)
    assert n == 1, f"Failed to add think-{role} token '{token_to_add}'"
    tid = tok.convert_tokens_to_ids(token_to_add)
    print(f"  Think-{role} token: '{token_to_add}' (id={tid}) [newly added]")
    return token_to_add, tid, True


def build_dual_vocab_tokenizer(
    base_model: str,
    out_dir: str,
    mode: str = "full_copy",
    subset_ids: Optional[List[int]] = None,
    subset_regex: Optional[str] = None,
    custom_tokens: Optional[List[str]] = None,
    custom_embeddings=None,
    think_start_candidates: Optional[List[str]] = None,
    think_end_candidates: Optional[List[str]] = None,
    think_missing: str = "error",
    latent_prefix: str = "<|latent_{}|>",
    filler_prefix: str = "<|filler_{}|>",
) -> dict:
    assert mode in ("full_copy", "subset", "custom")
    assert think_missing in ("add", "clone_eos", "error")

    if think_start_candidates is None:
        think_start_candidates = DEFAULT_THINK_START_CANDIDATES
    if think_end_candidates is None:
        think_end_candidates = DEFAULT_THINK_END_CANDIDATES

    print(f"Loading model '{base_model}' to read embedding vocab size …")
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
    V_model_orig: int = model.get_input_embeddings().weight.shape[0]
    del model
    print(f"  Original model vocab size: {V_model_orig}")

    tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    V_tok = len(tok)

    if V_tok < V_model_orig:
        need = V_model_orig - V_tok
        fillers = [filler_prefix.format(i) for i in range(need)]
        added = tok.add_tokens(fillers, special_tokens=False)
        assert added == need
        print(f"  Padded tokenizer by {need} filler tokens → size {V_model_orig}")
    elif V_tok > V_model_orig:
        raise ValueError(
            f"Tokenizer ({V_tok}) is larger than model embedding ({V_model_orig})."
        )

    print("\nResolving think-boundary tokens …")
    think_start_str, think_start_id, start_added = resolve_think_token(
        tok, think_start_candidates, role="start", if_missing=think_missing
    )
    think_end_str, think_end_id, end_added = resolve_think_token(
        tok, think_end_candidates, role="end", if_missing=think_missing
    )

    # If clone_eos, find the EOS-like token to use as initialisation source
    think_clone_source_id: Optional[int] = None
    think_clone_source_str: Optional[str] = None
    if think_missing == "clone_eos" and (start_added or end_added):
        think_clone_source_str, think_clone_source_id = _find_eos_clone_source(tok)
        print(f"  Clone source for think tokens: '{think_clone_source_str}' (id={think_clone_source_id})")

    V_final = len(tok)
    n_added = (1 if start_added else 0) + (1 if end_added else 0)
    print(f"  Visible vocab size V = {V_final}"
          + (f" (+{n_added} think token(s) added)" if n_added else ""))

    latent_token_strings: List[str] = []
    source_ids: Optional[List[int]] = None

    if mode == "full_copy":
        latent_token_strings = [latent_prefix.format(i) for i in range(V_final)]
        source_ids = list(range(V_final))
        print(f"\n  Mode=full_copy → {V_final} latent tokens (mirrors all of V)")

    elif mode == "subset":
        if subset_ids is not None and subset_regex is not None:
            raise ValueError("Provide subset_ids OR subset_regex, not both.")
        if subset_ids is None and subset_regex is None:
            raise ValueError("mode='subset' requires either --subset_ids_file or --subset_regex.")

        if subset_ids is not None:
            selected = sorted(set(int(i) for i in subset_ids))
        else:
            pattern = re.compile(subset_regex)
            vocab = tok.get_vocab()
            id2str = {v: k for k, v in vocab.items()}
            selected = sorted(
                tid for tid, tstr in id2str.items()
                if tid < V_final and pattern.search(tstr)
            )

        bad = [i for i in selected if not (0 <= i < V_final)]
        if bad:
            raise ValueError(f"subset contains ids outside [0, V={V_final}): {bad[:10]}")

        latent_token_strings = [latent_prefix.format(i) for i in selected]
        source_ids = selected
        print(f"\n  Mode=subset → {len(selected)} latent tokens")

    elif mode == "custom":
        if not custom_tokens:
            raise ValueError("mode='custom' requires a non-empty custom_tokens list.")
        latent_token_strings = list(custom_tokens)
        source_ids = None
        print(f"\n  Mode=custom → {len(custom_tokens)} custom latent tokens")

    L = len(latent_token_strings)
    num_added = tok.add_tokens(latent_token_strings, special_tokens=False)
    assert num_added == L, f"Expected to add {L} latent tokens, got {num_added}"
    total_vocab = len(tok)
    assert total_vocab == V_final + L

    print(f"  Total vocab: V={V_final} + L={L} = {total_vocab}")

    os.makedirs(out_dir, exist_ok=True)
    tok.save_pretrained(out_dir)

    emb_filename = None
    if mode == "custom" and custom_embeddings is not None:
        emb_filename = "latent_init_embeddings.npy"
        np.save(os.path.join(out_dir, emb_filename), np.asarray(custom_embeddings, dtype=np.float32))
        print(f"  Saved custom embeddings → {os.path.join(out_dir, emb_filename)}")

    meta = {
        "base_model": base_model,
        "V_model_orig": V_model_orig,
        "V": V_final,
        "L": L,
        "total_vocab": total_vocab,
        "latent_id_offset": V_final,
        "mode": mode,
        "latent_token_format": latent_prefix if mode != "custom" else None,
        "filler_token_format": filler_prefix,
        "source_ids": source_ids,
        "latent_init_embeddings_file": emb_filename,
        "think_start_token": think_start_str,
        "think_start_token_id": think_start_id,
        "think_end_token": think_end_str,
        "think_end_token_id": think_end_id,
        "think_init_mode": think_missing if (start_added or end_added) else "existing",
        "think_clone_source_id": think_clone_source_id,
        "think_clone_source_token": think_clone_source_str,
    }

    meta_path = os.path.join(out_dir, "dual_vocab_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nSaved dual-vocab tokenizer → {out_dir}")
    print(f"  V_orig={V_model_orig}, V={V_final}, L={L}, total={total_vocab}, mode={mode}")
    print(f"  think_start='{think_start_str}' (id={think_start_id}), "
          f"think_end='{think_end_str}' (id={think_end_id})")
    return meta


def parse_args():
    p = argparse.ArgumentParser(description="Build dual-vocabulary tokenizer for LatentReasoning")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--out_dir", default="./checkpoints/dual_tokenizer")
    p.add_argument("--mode", choices=["full_copy", "subset", "custom"], default="full_copy")
    p.add_argument("--latent_prefix", default="<|latent_{}|>")
    p.add_argument("--filler_prefix", default="<|filler_{}|>")
    p.add_argument("--think_start", nargs="+", default=None, metavar="TOKEN")
    p.add_argument("--think_end", nargs="+", default=None, metavar="TOKEN")
    p.add_argument("--think_missing", choices=["add", "clone_eos", "error"], default="error")
    p.add_argument("--subset_ids_file", default=None)
    p.add_argument("--subset_regex", default=None)
    p.add_argument("--custom_tokens_file", default=None)
    p.add_argument("--custom_embeddings_file", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    subset_ids = None
    if args.subset_ids_file:
        with open(args.subset_ids_file) as f:
            subset_ids = json.load(f)

    custom_tokens = None
    if args.custom_tokens_file:
        with open(args.custom_tokens_file) as f:
            custom_tokens = [line.strip() for line in f if line.strip()]

    custom_embeddings = None
    if args.custom_embeddings_file:
        custom_embeddings = np.load(args.custom_embeddings_file)

    build_dual_vocab_tokenizer(
        base_model=args.base_model,
        out_dir=args.out_dir,
        mode=args.mode,
        subset_ids=subset_ids,
        subset_regex=args.subset_regex,
        custom_tokens=custom_tokens,
        custom_embeddings=custom_embeddings,
        think_start_candidates=args.think_start,
        think_end_candidates=args.think_end,
        think_missing=args.think_missing,
        latent_prefix=args.latent_prefix,
        filler_prefix=args.filler_prefix,
    )


if __name__ == "__main__":
    main()
