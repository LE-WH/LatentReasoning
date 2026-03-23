"""
scripts/verify_dual_model.py
=============================
Quick sanity check for a dual-vocab model built with build_dual_vocab.py +
expand_model_to_dual_vocab.py.

Checks:
  - Tokenizer / embedding / lm_head sizes match dual_vocab_meta.json
  - Think-boundary tokens are single-token and in visible vocab [0, V)
  - No NaN/Inf in embeddings
  - Latent embeddings were correctly copied from source ids (full_copy/subset)

Usage
-----
python scripts/verify_dual_model.py --model_dir ./checkpoints/dual_model
python scripts/verify_dual_model.py --model_dir ./checkpoints/dual_model --check_latent
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
from ragen.dual_vocab.utils import load_dual_model, verify_dual_model, check_latent_init


def main():
    p = argparse.ArgumentParser(description="Verify a dual-vocab model")
    p.add_argument("--model_dir", required=True, help="Path to dual-vocab model directory")
    p.add_argument("--check_latent", action="store_true",
                   help="Run spot-check on latent embedding copy correctness")
    p.add_argument("--sample_size", type=int, default=200,
                   help="Number of latent tokens to spot-check (default: 200)")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="Max abs diff threshold for correct copy (default: 1e-4)")
    args = p.parse_args()

    print(f"Loading dual-vocab model from '{args.model_dir}' …\n")
    model, tok, meta = load_dual_model(args.model_dir)

    print("\n--- Meta ---")
    for k, v in meta.items():
        if k == "source_ids":
            print(f"  source_ids: [{len(v)} entries]" if v else "  source_ids: None")
        else:
            print(f"  {k}: {v}")

    print("\n--- Full Sanity Check ---")
    warnings = verify_dual_model(model, tok, meta)

    if args.check_latent:
        print("\n--- Latent Embedding Copy Check ---")
        report = check_latent_init(model, meta, sample_size=args.sample_size, tol=args.tol)
        for k, v in report.items():
            if k == "suspect_ids":
                if v:
                    print(f"  suspect_ids (first {len(v)}):")
                    for s in v:
                        print(f"    {s}")
            else:
                print(f"  {k}: {v}")

    sys.exit(0 if not warnings else 1)


if __name__ == "__main__":
    main()
