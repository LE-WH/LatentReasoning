"""Base class for SFT data construction methods.

Each method defines how to transform raw benchmark data into SFT training samples.
Follows RAGEN's BaseEnv pattern (ragen/env/base.py).
"""

from abc import ABC, abstractmethod

from sft.data.base import SFTSample


class BaseSFTMethod(ABC):
    """SFT data construction method base class.

    Subclasses implement different strategies for constructing training data:
    - Direct: answer only, no reasoning
    - Concise: shortest correct CoT reasoning
    """

    name: str

    @abstractmethod
    def build_samples(
        self, benchmark: str, raw_data: list[dict], **kwargs
    ) -> list[SFTSample]:
        """Construct SFT samples from raw benchmark data.

        Args:
            benchmark: Benchmark name (e.g. "gsm8k", "deepcoder").
            raw_data: List of dicts from BaseBenchmark.load_raw_data(),
                      each with at least 'question', 'answer', 'source_id'.
            **kwargs: Method-specific parameters.

        Returns:
            List of SFTSample instances.
        """
        raise NotImplementedError
