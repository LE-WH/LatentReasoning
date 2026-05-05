"""Upload SoT-distill artefacts to the Hugging Face Hub.

Three repos:
  1. Dataset repo  ``{user}/sot-distill-qwen3-4b-thinking-n16``  (~600 MB)
       collected/, selected/, sft_combined, eval/, figures/
  2. Model repo    ``{user}/sot-distill-vanilla-qwen3-4b-thinking-2507``  (~8 GB merged ckpt only)
  3. Model repo    ``{user}/sot-distill-latent-qwen3-4b-thinking-2507``   (~8.8 GB merged ckpt + dual_vocab_meta)

Skips upload of the things that are easily regenerated on the destination
machine:
  - ``checkpoints/dual_qwen_4b_thinking/`` (run ``scripts/build_dual_model.sh`` instead)
  - ``checkpoint-*`` directories inside model dirs (optimizer.pt etc.)
  - logs

Requires the env var ``HF_TOKEN`` (or HUGGINGFACE_USER_KEY in the shell)
and ``huggingface_hub`` installed.

Usage:
    python scripts/sft/hf_upload_sot_distill.py --user leapeto
    python scripts/sft/hf_upload_sot_distill.py --user leapeto --skip-models   # data only
    python scripts/sft/hf_upload_sot_distill.py --user leapeto --skip-data     # models only
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder

REPO_BASE = "/workspace/LatentReasoning"


def _stage_dataset(stage_dir: Path) -> None:
    """Materialise the dataset layout into ``stage_dir`` via cp/symlink."""
    base = Path(REPO_BASE)
    layout = {
        # SFT teacher samples (most expensive: ~2 hr on 4×A100 to regenerate)
        "collected/math":  base / "data/sft/sot/collected/math",
        "collected/gsm8k": base / "data/sft/sot/collected/gsm8k",
        # Selection + final SFT JSONL (cheap to regenerate but small)
        "selected":        base / "data/sft/sot/selected",
        # Final combined SFT input file
        "sft_combined.jsonl": base / "data/sft/sot/sft_combined.jsonl",
        # Eval JSONLs from both reports
        "eval/cod_vs_sot": base / "results/sft/cod_vs_sot/eval",
        "eval/sot_distill": base / "results/sft/sot_distill/eval",
        # Reused SoT-multiturn MATH-500 cell from the prior report
        "eval/phase1_sot/sot_thinking_on_multiturn_16k.jsonl":
            base / "results/sft/math/phase1_sot/eval/sot_thinking_on_multiturn_16k.jsonl",
        # Histograms used in the report (no PII / no code)
        "figures": Path("/workspace/LatentReports/figures"),
    }
    for dst_rel, src in layout.items():
        dst = stage_dir / dst_rel
        if not src.exists():
            print(f"  [skip] {src} (missing)")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        print(f"  staged {dst_rel}  <- {src}")
    # Drop a README
    (stage_dir / "README.md").write_text(_DATASET_README, encoding="utf-8")


def _stage_model(stage_dir: Path, model_dir: Path, *, include_dual_meta: bool) -> None:
    """Stage only the inference-ready merged files; skip checkpoint/optimizer."""
    keep_glob = [
        "model.safetensors",
        "model-*.safetensors",
        "model.safetensors.index.json",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "chat_template.jinja",
        "added_tokens.json",
    ]
    if include_dual_meta:
        keep_glob.append("dual_vocab_meta.json")

    matched: list[Path] = []
    for pat in keep_glob:
        matched.extend(model_dir.glob(pat))

    for f in matched:
        if not f.is_file():
            continue
        dst = stage_dir / f.name
        shutil.copy2(f, dst)
        print(f"  staged {f.name} ({f.stat().st_size/1e9:.2f} GB)")


_DATASET_README = """---
license: apache-2.0
language:
- en
size_categories:
- 100K<n<1M
task_categories:
- text-generation
tags:
- math
- chain-of-thought
- sketch-of-thought
- chain-of-draft
- qwen3
- sft
- distillation
- latent-reasoning
---

# SoT-distill artefacts — Qwen3-4B-Thinking-2507, N=16 samples / question

Collection + selection + SFT-input + eval JSONLs from the SoT-distillation
experiments described in the 20260503 report. Lets you skip the ~2 hours of
SoT teacher collection and the ~12 hours of training/eval on a fresh
machine.

## Layout

```
collected/math/shard_{0..3}.jsonl   # 16 SoT-multi-turn samples / question, T=0.7
collected/gsm8k/shard_{0..3}.jsonl  # idem
selected/math_top4.jsonl            # top-4 shortest correct per Q
selected/gsm8k_top4.jsonl
sft_combined.jsonl                  # final combined SFT data, 36,851 rows
eval/cod_vs_sot/                    # SoT-mt vs CoD-mt eval (Part A of the report)
eval/sot_distill/                   # base/vanilla/latent × math/gsm8k (Part B)
eval/phase1_sot/                    # the SoT-mt MATH-500 cell reused from prior work
figures/                            # histograms of token distributions
```

## How each file was made

- `collected/`: `python -m sft.methods.sot.collect --benchmark {math,gsm8k} --num-samples 16 --temperature 0.7 --max-tokens 4096 --num-shards 4 --shard-id $i`
- `selected/`: `python -m sft.methods.sot.select --top-k 4 --input-glob '...shard_*.jsonl'`
- `sft_combined.jsonl`: `python -m sft.methods.sot.build_sft_data --input math_top4.jsonl gsm8k_top4.jsonl`
- `eval/`: `python -m sft.eval_raw --sot-prompt --sot-{multiturn-exemplars,system-only}` per cell

## Companion model repos
- `{user}/sot-distill-vanilla-qwen3-4b-thinking-2507`
- `{user}/sot-distill-latent-qwen3-4b-thinking-2507`
"""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--user", required=True, help="HF user/org (e.g. leapeto)")
    p.add_argument("--dataset-repo", default="sot-distill-qwen3-4b-thinking-n16")
    p.add_argument("--vanilla-model-repo", default="sot-distill-vanilla-qwen3-4b-thinking-2507")
    p.add_argument("--latent-model-repo", default="sot-distill-latent-qwen3-4b-thinking-2507")
    p.add_argument("--skip-data", action="store_true")
    p.add_argument("--skip-models", action="store_true")
    p.add_argument("--private", action="store_true",
                   help="Create as private repos (default: public)")
    args = p.parse_args()

    api = HfApi()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_USER_KEY")
    if token:
        os.environ["HF_TOKEN"] = token  # huggingface_hub picks this up

    if not args.skip_data:
        repo_id = f"{args.user}/{args.dataset_repo}"
        print(f"\n=== uploading dataset repo: {repo_id} ===")
        create_repo(repo_id, repo_type="dataset", exist_ok=True, private=args.private)
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp) / "stage"
            stage.mkdir()
            _stage_dataset(stage)
            print(f"  uploading folder ({sum(1 for _ in stage.rglob('*'))} files)...")
            upload_folder(
                folder_path=str(stage),
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="upload SoT-distill data + eval JSONLs",
            )
        print(f"  done: https://huggingface.co/datasets/{repo_id}")

    if not args.skip_models:
        for label, model_subdir, repo_name, dual in [
            ("vanilla", "results/sft/sot_distill/vanilla",
             args.vanilla_model_repo, False),
            ("latent",  "results/sft/sot_distill/latent",
             args.latent_model_repo, True),
        ]:
            repo_id = f"{args.user}/{repo_name}"
            model_dir = Path(REPO_BASE) / model_subdir
            if not model_dir.exists():
                print(f"  [skip] {label}: {model_dir} missing")
                continue
            print(f"\n=== uploading model repo ({label}): {repo_id} ===")
            create_repo(repo_id, repo_type="model", exist_ok=True, private=args.private)
            with tempfile.TemporaryDirectory() as tmp:
                stage = Path(tmp) / "stage"
                stage.mkdir()
                _stage_model(stage, model_dir, include_dual_meta=dual)
                upload_folder(
                    folder_path=str(stage),
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"upload SoT-distilled {label} merged checkpoint",
                )
            print(f"  done: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
