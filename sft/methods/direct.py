"""Direct SFT method: answer only, no reasoning.

Constructs training samples with just <answer>...</answer>, no <think> tags.
"""

from sft.data.base import SFTSample
from .base import BaseSFTMethod


class DirectMethod(BaseSFTMethod):
    """Direct SFT: train on answer only, skip reasoning."""

    name = "direct"

    def build_samples(
        self, benchmark: str, raw_data: list[dict], **kwargs
    ) -> list[SFTSample]:
        """Build Direct SFT samples (answer-only).

        Each sample's assistant response will be: <answer>{answer}</answer>
        """
        samples = []
        for item in raw_data:
            samples.append(SFTSample(
                question=item["question"],
                answer=item["answer"],
                reasoning=None,
                benchmark=benchmark,
                method="direct",
                source_id=item.get("source_id", ""),
            ))
        return samples
