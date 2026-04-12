"""Convert method-specific intermediate data into SFT training JSONL."""

import argparse
import logging

from datasets import load_dataset
from ragen.env.static.utils import REGISTERD_STATIC_ENV
from sft.data.base import save_samples
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SFT training data")
    parser.add_argument("--method", type=str, required=True, choices=["concise", "direct", "tokenskip"])
    parser.add_argument("--benchmark", type=str, required=True, choices=list(DATASET_LOADERS.keys()))
    parser.add_argument("--cot-samples-path", type=str, default=None,
                        help="Comma-separated CoT sample files (required for concise)")
    parser.add_argument("--max-think-tokens", type=int, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="./data")
    parser.add_argument("--response-format", type=str, default="xml",
                        choices=["xml", "paper", "tokenskip_paper"],
                        help="xml: RAGEN tags; paper: raw text; tokenskip_paper: boxed final answer")
    parser.add_argument("--original-cot-path", type=str, default=None,
                        help="TokenSkip original CoT jsonl")
    parser.add_argument("--compressed-cot-dir", type=str, default=None,
                        help="Directory containing TokenSkip compressed_ratio_*.jsonl files")
    parser.add_argument("--ratio-pool", type=str, default="1.0,0.9,0.8,0.7,0.6,0.5",
                        help="Comma-separated TokenSkip ratio pool")
    parser.add_argument("--seed", type=int, default=42,
                        help="TokenSkip ratio sampling seed")
    parser.add_argument("--model-family", type=str, default="qwen", choices=["qwen", "llama3"])
    args = parser.parse_args()

    raw_data = load_raw_data(args.benchmark, cache_dir=args.cache_dir)
    logger.info(f"Loaded {len(raw_data)} questions from {args.benchmark}")

    if args.method == "concise":
        from sft.methods.self_training_concise.select import ConciseMethod

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
    elif args.method == "tokenskip":
        from sft.methods.tokenskip.select import TokenSkipMethod

        if args.response_format == "xml":
            raise ValueError(
                "TokenSkip currently only supports --response-format tokenskip_paper."
            )
        method = TokenSkipMethod()
        samples = method.build_samples(
            args.benchmark,
            raw_data,
            original_cot_path=args.original_cot_path,
            compressed_cot_dir=args.compressed_cot_dir,
            ratio_pool=args.ratio_pool,
            seed=args.seed,
            model_family=args.model_family,
            fmt=args.response_format,
        )
    else:
        raise ValueError(f"Unknown method: {args.method}")

    logger.info(f"Built {len(samples)} SFT samples")
    if args.response_format == "xml":
        system_prompt = get_system_prompt(args.benchmark)
    elif args.method == "tokenskip" and args.response_format == "tokenskip_paper":
        from sft.methods.tokenskip.prompts import get_raw_chat_system_prompt

        system_prompt = get_raw_chat_system_prompt()
    else:
        system_prompt = ""
    save_samples(
        samples,
        args.output,
        response_format=args.response_format,
        system_prompt=system_prompt,
    )
    logger.info(f"Saved {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
