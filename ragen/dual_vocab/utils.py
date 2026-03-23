"""
utils.py
========
Utilities for working with dual-vocabulary models in LatentReasoning.

Public API
----------
# Loading
load_dual_model(model_dir, ...)          -> (model, tokenizer, meta)
load_meta(model_dir)                     -> dict
is_dual_model(model_dir)                 -> bool

# Token / id helpers
is_visible(meta, token_id)               -> bool
is_latent(meta, token_id)                -> bool
visible_id(meta, latent_id)              -> int
latent_id(meta, visible_id)              -> int

# Embedding inspection
check_latent_init(model, meta, ...)      -> dict
verify_dual_model(model, tok, meta)      -> list[str]

Example
-------
    from ragen.dual_vocab.utils import load_dual_model, verify_dual_model

    model, tok, meta = load_dual_model("./dual_model")
    warnings = verify_dual_model(model, tok, meta)
"""
from __future__ import annotations

import os
import json
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Meta helpers
# ---------------------------------------------------------------------------

def is_dual_model(model_dir: str) -> bool:
    """Return True if model_dir contains a dual_vocab_meta.json."""
    return os.path.exists(os.path.join(model_dir, "dual_vocab_meta.json"))


def load_meta(model_dir: str) -> dict:
    """Load dual_vocab_meta.json from model_dir."""
    path = os.path.join(model_dir, "dual_vocab_meta.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"dual_vocab_meta.json not found in '{model_dir}'. "
            "Is this a dual-vocab model directory?"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dual_model(
    model_dir: str,
    dtype: torch.dtype = torch.float16,
    device_map: str = "auto",
    trust_remote_code: bool = True,
) -> tuple:
    """
    Load a dual-vocab model, tokenizer, and metadata from model_dir.

    Returns
    -------
    model      : AutoModelForCausalLM (eval mode)
    tokenizer  : AutoTokenizer
    meta       : dict  (contents of dual_vocab_meta.json)
    """
    meta = load_meta(model_dir)

    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    expected = meta["total_vocab"]
    actual = len(tok)
    if actual != expected:
        print(f"[Warning] Tokenizer length {actual} != meta total_vocab {expected}.")

    print(f"Loaded dual-vocab model from '{model_dir}'")
    print(f"  mode={meta['mode']}, V={meta['V']}, L={meta['L']}, "
          f"total={meta['total_vocab']}")
    print(f"  think_start='{meta['think_start_token']}' (id={meta['think_start_token_id']}), "
          f"think_end='{meta['think_end_token']}' (id={meta['think_end_token_id']})")

    return model, tok, meta


# ---------------------------------------------------------------------------
# Token / id helpers
# ---------------------------------------------------------------------------

def is_visible(meta: dict, token_id: int) -> bool:
    """True if token_id is in the visible vocab [0, V)."""
    return 0 <= token_id < meta["V"]


def is_latent(meta: dict, token_id: int) -> bool:
    """True if token_id is in the latent vocab [V, V+L)."""
    V, L = meta["V"], meta["L"]
    return V <= token_id < V + L


def visible_id(meta: dict, lat_id: int) -> int:
    """
    Given a latent-space id, return its corresponding visible-space id.
    Only valid for full_copy / subset modes where source_ids is defined.
    """
    source_ids = meta.get("source_ids")
    if source_ids is None:
        raise ValueError(
            "visible_id() requires source_ids in meta (full_copy or subset mode). "
            "Custom mode has no guaranteed V-space correspondence."
        )
    V = meta["V"]
    j = lat_id - V
    if not (0 <= j < len(source_ids)):
        raise ValueError(f"lat_id={lat_id} is out of latent range [V={V}, V+L={V+meta['L']})")
    return source_ids[j]


def latent_id(meta: dict, vis_id: int) -> int:
    """
    Given a visible-space id, return the corresponding latent-space id.
    Only valid for full_copy / subset modes.
    """
    source_ids = meta.get("source_ids")
    if source_ids is None:
        raise ValueError(
            "latent_id() requires source_ids in meta (full_copy or subset mode)."
        )
    V = meta["V"]
    try:
        j = source_ids.index(vis_id)
    except ValueError:
        raise KeyError(
            f"visible id {vis_id} has no latent counterpart "
            f"(not in source_ids; mode={meta['mode']})."
        )
    return V + j


# ---------------------------------------------------------------------------
# Embedding access helpers
# ---------------------------------------------------------------------------

def _emb_matrix(model) -> torch.Tensor:
    return model.get_input_embeddings().weight.detach()


def _head_matrix(model) -> Optional[torch.Tensor]:
    if not hasattr(model, "lm_head"):
        return None
    emb = model.get_input_embeddings().weight
    head = model.lm_head.weight
    if head.storage().data_ptr() == emb.storage().data_ptr():
        return None  # tied
    return head.detach()


def _compare_embeddings(model, id_a: int, id_b: int) -> dict:
    emb = _emb_matrix(model)
    va = emb[id_a].float().cpu()
    vb = emb[id_b].float().cpu()
    cos = F.cosine_similarity(va.unsqueeze(0), vb.unsqueeze(0)).item()
    diff = va - vb
    return {
        "cosine_sim": round(cos, 8),
        "max_abs_diff": round(diff.abs().max().item(), 8),
        "are_identical": torch.equal(va, vb),
    }


# ---------------------------------------------------------------------------
# Latent init verification
# ---------------------------------------------------------------------------

def check_latent_init(
    model,
    meta: dict,
    sample_size: int = 50,
    tol: float = 1e-4,
) -> dict:
    """
    Spot-check that latent embeddings were correctly copied from V-space.

    For full_copy / subset modes: checks emb[V+j] ≈ emb[source_ids[j]].
    For custom mode: reports basic stats only.

    Returns a summary dict.
    """
    V = meta["V"]
    L = meta["L"]
    mode = meta["mode"]
    source_ids = meta.get("source_ids")

    result = {"mode": mode, "V": V, "L": L}

    if source_ids is None:
        emb = _emb_matrix(model)
        lat_emb = emb[V: V + L].float().cpu()
        result.update({
            "note": "Custom mode: no source_ids to verify against.",
            "latent_emb_mean_norm": round(lat_emb.norm(dim=1).mean().item(), 6),
            "latent_has_nan": bool(lat_emb.isnan().any().item()),
            "latent_has_inf": bool(lat_emb.isinf().any().item()),
        })
        return result

    indices = list(range(L))
    if sample_size != -1 and sample_size < L:
        step = max(1, L // sample_size)
        indices = indices[::step][:sample_size]

    cosines, max_abs_diffs, suspect_ids = [], [], []

    for j in indices:
        lat = V + j
        src = source_ids[j]
        stats = _compare_embeddings(model, src, lat)
        cosines.append(stats["cosine_sim"])
        max_abs_diffs.append(stats["max_abs_diff"])
        if stats["max_abs_diff"] > tol:
            suspect_ids.append({"latent_id": lat, "source_id": src, **stats})

    correct = len(indices) - len(suspect_ids)
    result.update({
        "sampled": len(indices),
        "correct": correct,
        "incorrect": len(suspect_ids),
        "tol": tol,
        "mean_cosine": round(sum(cosines) / len(cosines), 8),
        "min_cosine": round(min(cosines), 8),
        "mean_max_abs_diff": round(sum(max_abs_diffs) / len(max_abs_diffs), 8),
        "suspect_ids": suspect_ids[:10],
        "status": "OK" if not suspect_ids else f"FAIL ({len(suspect_ids)} suspect pairs)",
    })
    return result


# ---------------------------------------------------------------------------
# Full sanity check
# ---------------------------------------------------------------------------

def verify_dual_model(model, tok, meta: dict) -> list:
    """
    Run sanity checks on a loaded dual-vocab model.
    Returns a list of warning strings; empty list = all OK.
    """
    warnings = []
    V = meta["V"]
    L = meta["L"]
    total = meta["total_vocab"]
    emb = _emb_matrix(model)

    if len(tok) != total:
        warnings.append(f"Tokenizer has {len(tok)} tokens but meta.total_vocab={total}.")

    if emb.shape[0] != total:
        warnings.append(f"Embedding matrix has {emb.shape[0]} rows but meta.total_vocab={total}.")

    for role in ("start", "end"):
        tid = meta[f"think_{role}_token_id"]
        tstr = meta[f"think_{role}_token"]
        if not (0 <= tid < V):
            warnings.append(
                f"think_{role}_token '{tstr}' has id={tid} NOT in visible vocab [0, {V})."
            )
        ids = tok.encode(tstr, add_special_tokens=False)
        if len(ids) != 1:
            warnings.append(
                f"think_{role}_token '{tstr}' encodes as {len(ids)} tokens: {ids}. Expected 1."
            )

    emb_f = emb.float()
    if emb_f.isnan().any():
        warnings.append("NaN detected in embedding matrix.")
    if emb_f.isinf().any():
        warnings.append("Inf detected in embedding matrix.")

    head = _head_matrix(model)
    if head is not None:
        head_f = head.float()
        if head_f.isnan().any():
            warnings.append("NaN detected in lm_head weight matrix.")
        if head_f.isinf().any():
            warnings.append("Inf detected in lm_head weight matrix.")

    if meta.get("source_ids") is not None:
        report = check_latent_init(model, meta, sample_size=100, tol=1e-3)
        if report.get("incorrect", 0) > 0:
            warnings.append(
                f"Latent init check FAILED: {report['incorrect']}/{report['sampled']} "
                f"pairs have max_abs_diff > {report['tol']}. "
                f"min_cosine={report['min_cosine']}."
            )

    if warnings:
        print(f"verify_dual_model: {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  [!] {w}")
    else:
        print("verify_dual_model: all checks passed.")

    return warnings
