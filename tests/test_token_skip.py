from __future__ import annotations

import json

from token_skip_utils import append_inline_ratio_condition, build_inline_ratio_suffix
from sft.methods.tokenskip.prompts import build_tokenskip_raw_chat_messages
from sft.methods.tokenskip.select import TokenSkipMethod


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_inline_ratio_suffix_qwen():
    assert build_inline_ratio_suffix(1.0, model_family="qwen") == ""
    assert build_inline_ratio_suffix(0.5, model_family="qwen") == "<|eot_id|>0.5<|eot_id|>"
    assert append_inline_ratio_condition("question", 0.8, model_family="qwen") == "question<|eot_id|>0.8<|eot_id|>"


def test_inline_ratio_suffix_llama3_uses_eot_id():
    assert build_inline_ratio_suffix(0.6, model_family="llama3") == "<|eot_id|>0.6<|eot_id|>"


def test_tokenskip_raw_chat_messages_include_system_prompt():
    messages = build_tokenskip_raw_chat_messages("Q0", 0.5, model_family="qwen")
    assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert messages[1]["role"] == "user"
    assert messages[1]["content"].endswith("<|eot_id|>0.5<|eot_id|>")


def test_tokenskip_method_builds_conditioned_samples(tmp_path):
    original_path = tmp_path / "original.jsonl"
    compressed_dir = tmp_path / "compressed"
    compressed_dir.mkdir()

    write_jsonl(
        original_path,
        [
            {
                "source_id": "gsm8k_train_0",
                "question": "Q0",
                "answer": "42",
                "reasoning": "original reasoning",
                "is_correct": True,
            }
        ],
    )
    write_jsonl(
        compressed_dir / "compressed_ratio_0.5.jsonl",
        [
            {
                "source_id": "gsm8k_train_0",
                "answer": "42",
                "compressed_reasoning": "compressed reasoning",
            }
        ],
    )

    raw_data = [
        {
            "question": "Q0",
            "answer": "42",
            "source_id": "gsm8k_train_0",
        }
    ]

    method = TokenSkipMethod()
    samples = method.build_samples(
        "gsm8k",
        raw_data,
        original_cot_path=str(original_path),
        compressed_cot_dir=str(compressed_dir),
        ratio_pool="0.5",
        seed=42,
        model_family="qwen",
        fmt="tokenskip_paper",
    )

    assert len(samples) == 1
    assert samples[0].reasoning == "compressed reasoning"
    assert samples[0].question.endswith("<|eot_id|>0.5<|eot_id|>")
    assert samples[0].extra_metadata["compression_ratio"] == 0.5
