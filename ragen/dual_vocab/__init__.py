"""
ragen/dual_vocab
================
Dual-vocabulary model support for LatentReasoning.

Provides:
  - constraint.py  : logits processors to enforce visible/latent vocab switching
  - utils.py       : load/verify dual-vocab models, token helpers

Build a dual model from a base model:
  1. python scripts/build_dual_vocab.py --base_model <model> --out_dir <tok_dir>
  2. python scripts/expand_model_to_dual_vocab.py --base_model <model> \
         --dual_tok_dir <tok_dir> --out_dir <model_dir>
"""

from .constraint import (
    CoTVocabConstraint,
    make_hf_logits_processor,
    make_vllm_logits_processor,
)
from .utils import (
    load_dual_model,
    load_meta,
    verify_dual_model,
    check_latent_init,
    is_visible,
    is_latent,
    visible_id,
    latent_id,
)

__all__ = [
    "CoTVocabConstraint",
    "make_hf_logits_processor",
    "make_vllm_logits_processor",
    "load_dual_model",
    "load_meta",
    "verify_dual_model",
    "check_latent_init",
    "is_visible",
    "is_latent",
    "visible_id",
    "latent_id",
]
