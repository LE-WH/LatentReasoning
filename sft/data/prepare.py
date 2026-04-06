"""Convert CoT samples into SFT training JSONL.

Usage:
    python -m sft.data.prepare \
        --method concise \
        --benchmark math \
        --cot-samples-path data/sft/cot_samples/math_cot_zero-shot.jsonl \
        --max-think-tokens 1023 \
        --output data/sft/math_concise_xml.jsonl
"""

import argparse
import json
import logging
from pathlib import Path

from datasets import load_dataset
from ragen.env.static.utils import REGISTERD_STATIC_ENV
from sft.methods.self_training_concise.select import ConciseMethod
from sft.methods.self_training_concise.prompts import get_system_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


DATASET_LOADERS = {
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "train",
    },
    "math": {
        "path": "nlile/hendrycks-MATH-benchmark",
        "name": None,
        "split": "train",
    },
}


def load_raw_data(benchmark: str, cache_dir: str = "./data") -> list[dict]:
    """Load raw data for a benchmark and return list of dicts."""
    loader = DATASET_LOADERS[benchmark]
    kwargs = {"path": loader["path"], "split": loader["split"], "cache_dir": cache_dir}
    if loader["name"]:
        kwargs["name"] = loader["name"]
    ds = load_dataset(**kwargs)

    processor = REGISTERD_STATIC_ENV[benchmark]["processor"]
    raw_data = []
    for idx, item in enumerate(ds):
        question, answer = processor(item)
        raw_data.append({
            "question": question,
            "answer": answer,
            "source_id": f"{benchmark}_train_{idx}",
        })
    return raw_data


def sample_to_messages(question: str, response: str, benchmark: str) -> list[dict]:
    """Convert a question + response into chat messages."""
    system_prompt = get_system_prompt(benchmark)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": response})
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT training data")
    parser.add_argument("--method", type=str, required=True, choices=["concise", "direct"])
    parser.add_argument("--benchmark", type=str, required=True, choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--cot-samples-path", type=str, default=None,
                        help="Comma-separated CoT sample files (required for concise)")
    parser.add_argument("--max-think-tokens", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="./data")
    parser.add_argument("--response-format", type=str, default="xml",
                        choices=["xml", "paper"],
                        help="xml: <think>/<answer> tags; paper: raw text")
    args = parser.parse_args()

    raw_data = load_raw_data(args.benchmark, cache_dir=args.cache_dir)
    logger.info(f"Loaded {len(raw_data)} questions from {args.benchmark}")

    if args.method == "concise":
        method = ConciseMethod()
        samples = method.build_samples(
            args.benchmark,
            raw_data,
            cot_samples_path=args.cot_samples_path,
            max_think_tokens=args.max_think_tokens,
        )
    elif args.method == "direct":
        from sft.methods.direct import DirectMethod
        method = DirectMethod()
        samples = method.build_samples(args.benchmark, raw_data)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    logger.info(f"Built {len(samples)} SFT samples")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for sample in samples:
            if args.response_format == "xml":
                if sample.reasoning:
                    response = f"<think>\n{sample.reasoning}\n</think>\n<answer>{sample.answer}</answer>"
                else:
                    response = f"<answer>{sample.answer}</answer>"
            else:
                # Paper raw text format (no XML tags)
                if sample.reasoning:
                    response = f"{sample.reasoning} The answer is {sample.answer}."
                else:
                    response = sample.answer

            messages = sample_to_messages(sample.question, response, args.benchmark)
            f.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
