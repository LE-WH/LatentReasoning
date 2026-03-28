"""
scripts/expand_model_to_dual_vocab.py
======================================
Resize the base model's embedding table and LM head to the dual-vocab size
produced by build_dual_vocab.py, then initialise the new latent rows.

Run AFTER build_dual_vocab.py.

Initialisation strategy (per mode in dual_vocab_meta.json):
  full_copy : emb[V+j] ← emb[source_ids[j]]  for j in [0, L)
  subset    : same copy logic, fewer rows
  custom    : load from .npy file, or random init matched to existing distribution

Usage
-----
python scripts/expand_model_to_dual_vocab.py \\
    --base_model Qwen/Qwen2.5-3B-Instruct \\
    --dual_tok_dir ./checkpoints/dual_tokenizer \\
    --out_dir ./checkpoints/dual_model
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import argparse
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def _emb_stats(weight: torch.Tensor):
    return weight.mean(dim=0), weight.std(dim=0)


def _is_tied(model) -> bool:
    if not hasattr(model, "lm_head"):
        return True
    emb_ptr = model.get_input_embeddings().weight.storage().data_ptr()
    head_ptr = model.lm_head.weight.storage().data_ptr()
    return emb_ptr == head_ptr


@torch.no_grad()
def resize_embeddings_and_head(model, new_vocab_size: int) -> int:
    old_vocab = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(new_vocab_size)

    if hasattr(model, "lm_head") and model.lm_head.weight.shape[0] != new_vocab_size:
        old_w = model.lm_head.weight.data
        d = old_w.shape[1]
        new_head = torch.nn.Linear(d, new_vocab_size, bias=False).to(old_w.device, old_w.dtype)
        new_head.weight.data[:old_w.shape[0]] = old_w
        model.lm_head = new_head
        print(f"  Manually resized lm_head: {old_w.shape[0]} → {new_vocab_size}")

    return old_vocab


@torch.no_grad()
def init_think_token_rows(model, V_model_orig: int, V_final: int,
                          clone_source_id: int = None, noise_scale: float = 0.01):
    """Initialise newly-added think-boundary token rows.

    If *clone_source_id* is given (think_init_mode == "clone_eos"), each new
    row is copied from the clone source (e.g. <|im_end|>) with small Gaussian
    noise so the model already "wants" to emit </think> in EOS-like contexts.
    Otherwise falls back to random initialisation (original "add" behaviour).
    """
    n = V_final - V_model_orig
    if n <= 0:
        return

    emb_w = model.get_input_embeddings().weight
    tied = _is_tied(model)

    if clone_source_id is not None:
        # Clone from the EOS-like token + subtle noise
        src_emb = emb_w[clone_source_id].clone()
        noise = torch.randn(n, emb_w.shape[1], device=emb_w.device, dtype=emb_w.dtype)
        emb_w[V_model_orig:V_final] = src_emb.unsqueeze(0) + noise * src_emb.abs().mean() * noise_scale

        if not tied:
            out_w = model.lm_head.weight
            src_out = out_w[clone_source_id].clone()
            noise2 = torch.randn(n, out_w.shape[1], device=out_w.device, dtype=out_w.dtype)
            out_w[V_model_orig:V_final] = src_out.unsqueeze(0) + noise2 * src_out.abs().mean() * noise_scale

        print(f"  Initialised {n} think-token row(s) in [{V_model_orig}, {V_final}) "
              f"cloned from token id {clone_source_id} (noise_scale={noise_scale})")
    else:
        # Original random init
        mean, std = _emb_stats(emb_w[:V_model_orig])
        noise = torch.randn(n, emb_w.shape[1], device=emb_w.device, dtype=emb_w.dtype)
        emb_w[V_model_orig:V_final] = mean.unsqueeze(0) + noise * std.unsqueeze(0) * 0.01

        if not tied:
            out_w = model.lm_head.weight
            mean2, std2 = _emb_stats(out_w[:V_model_orig])
            noise2 = torch.randn(n, out_w.shape[1], device=out_w.device, dtype=out_w.dtype)
            out_w[V_model_orig:V_final] = mean2.unsqueeze(0) + noise2 * std2.unsqueeze(0) * 0.01

        print(f"  Initialised {n} think-token row(s) in [{V_model_orig}, {V_final}) with random init")


@torch.no_grad()
def init_latent_rows(model, V_final: int, L: int, source_ids, custom_embeddings):
    emb_w = model.get_input_embeddings().weight
    tied = _is_tied(model)

    if source_ids is not None:
        assert len(source_ids) == L
        src = torch.tensor(source_ids, dtype=torch.long, device=emb_w.device)
        src_emb = emb_w[src].clone()
        emb_w[V_final:V_final + L] = src_emb

        if not tied:
            out_w = model.lm_head.weight
            out_w[V_final:V_final + L] = out_w[src].clone()

        print(f"  Latent rows [{V_final}, {V_final+L}) copied from {len(source_ids)} source_ids.")

    elif custom_embeddings is not None:
        arr = torch.tensor(custom_embeddings, dtype=emb_w.dtype, device=emb_w.device)
        assert arr.shape == (L, emb_w.shape[1])
        emb_w[V_final:V_final + L] = arr

        if not tied:
            out_w = model.lm_head.weight
            out_w[V_final:V_final + L] = arr.to(out_w.dtype)

        print(f"  Latent rows [{V_final}, {V_final+L}) loaded from custom embeddings.")

    else:
        mean, std = _emb_stats(emb_w[:V_final])
        noise = torch.randn(L, emb_w.shape[1], device=emb_w.device, dtype=emb_w.dtype)
        emb_w[V_final:V_final + L] = mean.unsqueeze(0) + noise * std.unsqueeze(0)

        if not tied:
            out_w = model.lm_head.weight
            mean2, std2 = _emb_stats(out_w[:V_final])
            noise2 = torch.randn(L, out_w.shape[1], device=out_w.device, dtype=out_w.dtype)
            out_w[V_final:V_final + L] = mean2.unsqueeze(0) + noise2 * std2.unsqueeze(0)

        print(f"  Latent rows [{V_final}, {V_final+L}) randomly initialised.")


def expand(base_model: str, dual_tok_dir: str, out_dir: str):
    tok = AutoTokenizer.from_pretrained(dual_tok_dir, use_fast=True)

    meta_path = os.path.join(dual_tok_dir, "dual_vocab_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    V_model_orig: int = meta["V_model_orig"]
    V_final: int = meta["V"]
    L: int = meta["L"]
    total_vocab: int = meta["total_vocab"]
    source_ids = meta.get("source_ids")
    emb_file = meta.get("latent_init_embeddings_file")

    assert total_vocab == V_final + L
    assert len(tok) == total_vocab, (
        f"Tokenizer has {len(tok)} tokens but metadata says {total_vocab}. "
        "Re-run build_dual_vocab.py."
    )

    print(f"Metadata: mode={meta['mode']}, V_orig={V_model_orig}, V={V_final}, "
          f"L={L}, total={total_vocab}")

    custom_embeddings = None
    if emb_file is not None:
        emb_path = os.path.join(dual_tok_dir, emb_file)
        custom_embeddings = np.load(emb_path).astype(np.float32)
        print(f"Loaded custom embeddings from {emb_path}, shape={custom_embeddings.shape}")

    print(f"Loading base model '{base_model}' …")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    actual_V_orig = model.get_input_embeddings().weight.shape[0]
    assert actual_V_orig == V_model_orig, (
        f"Model embedding size ({actual_V_orig}) != metadata V_model_orig ({V_model_orig})."
    )

    print(f"Resizing embeddings: {actual_V_orig} → {total_vocab} …")
    resize_embeddings_and_head(model, total_vocab)

    clone_source_id = meta.get("think_clone_source_id")
    init_think_token_rows(model, V_model_orig, V_final, clone_source_id=clone_source_id)
    init_latent_rows(model, V_final, L, source_ids, custom_embeddings)

    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving model → {out_dir} …")
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    with open(os.path.join(out_dir, "dual_vocab_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\nDone. Saved dual-vocab model → {out_dir}")
    print(f"  V_orig={V_model_orig}, V={V_final}, L={L}, total={total_vocab}, mode={meta['mode']}")


def parse_args():
    p = argparse.ArgumentParser(description="Expand base model to dual-vocab size")
    p.add_argument("--base_model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--dual_tok_dir", default="./checkpoints/dual_tokenizer")
    p.add_argument("--out_dir", default="./checkpoints/dual_model")
    return p.parse_args()


def main():
    args = parse_args()
    expand(base_model=args.base_model, dual_tok_dir=args.dual_tok_dir, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
