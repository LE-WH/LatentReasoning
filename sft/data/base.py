"""Core data structures and serialization helpers for SFT samples."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Optional


def _strip_xml_tags(text: str) -> str:
    """Remove XML tags used by the RAGEN-style reasoning format."""
    text = re.sub(r"<answer>.*", "", text, flags=re.DOTALL)
    text = re.sub(r"</?think>", "", text)
    text = re.sub(r"</?answer>", "", text)
    text = re.sub(r"</?[a-z]+>", "", text)
    return text.strip()


@dataclass
class SFTSample:
    """A single SFT training sample."""

    question: str
    answer: str
    reasoning: Optional[str] = None
    benchmark: str = ""
    method: str = ""
    source_id: str = ""
    extra_metadata: Optional[dict[str, Any]] = None

    def build_response(self, fmt: str = "xml") -> str:
        """Build the assistant-side training target."""
        if fmt == "xml":
            if self.reasoning:
                return f"<think>\n{self.reasoning}\n</think>\n<answer>{self.answer}</answer>"
            return f"<answer>{self.answer}</answer>"

        if fmt == "tokenskip_paper":
            if self.reasoning:
                clean = _strip_xml_tags(self.reasoning)
                return f"{clean}\n\n</think>\n\nThe final answer is: $\\boxed{{{self.answer}}}$"
            return f"</think>\n\nThe final answer is: $\\boxed{{{self.answer}}}$"

        if self.reasoning:
            clean = _strip_xml_tags(self.reasoning)
            return f"{clean}\nThe answer is {self.answer}"
        return f"The answer is {self.answer}"

    def to_messages(self, response_format: str = "xml", system_prompt: str = "") -> list[dict]:
        """Convert a sample to chat-style messages."""
        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": self.question})
        messages.append({"role": "assistant", "content": self.build_response(response_format)})
        return messages

    def to_jsonl_dict(self, response_format: str = "xml", system_prompt: str = "") -> dict:
        """Convert a sample to the JSONL schema used by the SFT trainer."""
        metadata = {
            "benchmark": self.benchmark,
            "method": self.method,
            "source_id": self.source_id,
        }
        if self.extra_metadata:
            metadata.update(self.extra_metadata)
        return {
            "messages": self.to_messages(response_format=response_format, system_prompt=system_prompt),
            "metadata": metadata,
        }


def save_samples(
    samples: list[SFTSample],
    output_path: str | Path,
    *,
    response_format: str = "xml",
    system_prompt: str = "",
) -> None:
    """Save SFT samples to jsonl."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in samples:
            f.write(
                json.dumps(
                    sample.to_jsonl_dict(
                        response_format=response_format,
                        system_prompt=system_prompt,
                    ),
                    ensure_ascii=False,
                )
                + "\n"
            )
