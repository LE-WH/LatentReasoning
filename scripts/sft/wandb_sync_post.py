"""Post-hoc upload of HF Trainer metrics to W&B.

Reads ``trainer_state.json`` (preferred) or parses the stdout log if missing,
then creates a W&B run and re-emits each ``log_history`` entry with the
correct ``global_step``. Useful when a training was started with
``report_to: "none"`` and we want the same plots as the live runs.

Usage:
    python scripts/sft/wandb_sync_post.py \
        --trainer-state results/sft/sot_distill/vanilla/trainer_state.json \
        --project sot-distill --run-name sot-distill-vanilla-qwen3-4b-thinking
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import wandb


def load_trainer_state(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f).get("log_history", [])


def parse_stdout_log(path: Path, logging_steps: int = 10) -> list[dict]:
    """Fallback: parse stdout log when trainer_state.json missing.

    Looks for lines like:
        {'loss': '0.101', 'grad_norm': '1.469', 'learning_rate': '1e-05', 'epoch': '0.5557'}
    Reconstructs synthetic ``step`` field as ``logging_steps × index``.
    """
    text = path.read_text(errors="replace")
    rows: list[dict] = []
    pat = re.compile(r"\{'loss': '?([0-9.]+)'?,\s*'grad_norm': '?([0-9.einf+\-]+)'?,\s*'learning_rate': '?([0-9.e\-]+)'?,\s*'epoch': '?([0-9.]+)'?\}")
    for i, m in enumerate(pat.finditer(text)):
        loss, gn, lr, ep = m.groups()
        rows.append({
            "step": (i + 1) * logging_steps,
            "loss": float(loss),
            "grad_norm": float(gn) if gn not in ("inf", "nan") else float(gn),
            "learning_rate": float(lr),
            "epoch": float(ep),
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trainer-state", type=Path, default=None)
    p.add_argument("--stdout-log", type=Path, default=None)
    p.add_argument("--project", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--logging-steps", type=int, default=10,
                   help="logging_steps used during training (only needed for stdout fallback)")
    p.add_argument("--config-json", type=Path, default=None,
                   help="Optional path to a JSON file to attach as wandb config")
    args = p.parse_args()

    rows: list[dict] = []
    src = "trainer_state"
    if args.trainer_state and args.trainer_state.exists():
        rows = load_trainer_state(args.trainer_state)
    elif args.stdout_log and args.stdout_log.exists():
        rows = parse_stdout_log(args.stdout_log, args.logging_steps)
        src = "stdout_log"

    if not rows:
        raise SystemExit("No log rows found. Provide --trainer-state or --stdout-log.")

    cfg: dict = {"sync_source": src}
    if args.config_json and args.config_json.exists():
        cfg.update(json.loads(args.config_json.read_text()))

    wandb.init(project=args.project, name=args.run_name, config=cfg)
    for row in rows:
        # row may have 'step' from trainer_state, else we computed it.
        step = row.get("step")
        # Don't double-log final summary rows that have train_runtime (no per-step loss)
        loggable = {k: v for k, v in row.items() if k not in ("step",)}
        if step is None:
            wandb.log(loggable)
        else:
            wandb.log(loggable, step=int(step))
    wandb.finish()
    print(f"Uploaded {len(rows)} rows to wandb run '{args.run_name}' (source={src})")


if __name__ == "__main__":
    main()
