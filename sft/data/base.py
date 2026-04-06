"""Core data structures for SFT training samples."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SFTSample:
    """A single SFT training sample."""

    question: str
    answer: str
    reasoning: Optional[str] = None
    benchmark: str = ""
    method: str = ""
    source_id: str = ""
