"""
constraint.py
=============
Dual-vocabulary logits processors for LatentReasoning.

Two interfaces:
  1. HuggingFace (transformers.LogitsProcessor) — used by VllmWrapperWg via
     vLLM's per-request logits_processors in SamplingParams.
  2. Plain callable compatible with vLLM's SamplingParams.logits_processors
     (signature: (List[int], torch.Tensor) -> torch.Tensor).

Vocab layout produced by build_dual_vocab.py / expand_model_to_dual_vocab.py:
  visible ids : [0,   V)
  latent  ids : [V,   V+L)

Constraint logic:
  - Before </think> appears in output: allow latent [V, V+L) + </think> + EOS
  - After  </think>: allow visible [0, V) + EOS
"""
from __future__ import annotations

from typing import List

import torch
from transformers import LogitsProcessor, LogitsProcessorList


# ---------------------------------------------------------------------------
# HuggingFace LogitsProcessor
# ---------------------------------------------------------------------------

class CoTVocabConstraint(LogitsProcessor):
    """
    Dual-vocab constraint for HuggingFace-style generation.

    Stateful: call reset() before each new generate() call when reusing
    the same instance across batches.
    """

    def __init__(
        self,
        V: int,
        think_end_id: int,
        eos_id: int,
        end_bias: float = 5.0,
    ) -> None:
        self.V = int(V)
        self.think_end_id = int(think_end_id)
        self.eos_id = int(eos_id)
        self.end_bias = float(end_bias)

        # Lazily initialised on first call (device-dependent)
        self.allowed_think: torch.Tensor | None = None
        self.allowed_answer: torch.Tensor | None = None
        self.ended: torch.Tensor | None = None

    def _init_allowed(self, device: torch.device, vocab_size: int) -> None:
        V = self.V

        allow_think = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        allow_think[V: 2 * V] = True
        allow_think[self.think_end_id] = True
        allow_think[self.eos_id] = True

        allow_answer = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        allow_answer[:V] = True
        allow_answer[self.eos_id] = True

        self.allowed_think = allow_think
        self.allowed_answer = allow_answer

    def reset(self) -> None:
        """Reset per-batch ended state before a fresh generate() call."""
        self.ended = None

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        bsz, vocab_size = scores.shape
        device = scores.device

        if self.allowed_think is None or self.allowed_think.numel() != vocab_size:
            self._init_allowed(device, vocab_size)

        if self.ended is None or self.ended.numel() != bsz:
            self.ended = torch.zeros(bsz, dtype=torch.bool, device=device)

        last_tokens = input_ids[:, -1]
        self.ended |= (last_tokens == self.think_end_id)

        for i in range(bsz):
            if self.ended[i].item():
                scores[i, ~self.allowed_answer] = float("-inf")
            else:
                scores[i, ~self.allowed_think] = float("-inf")
                scores[i, self.think_end_id] += self.end_bias

        return scores


def make_hf_logits_processor(
    V: int,
    think_end_id: int,
    eos_id: int,
    end_bias: float = 5.0,
) -> LogitsProcessorList:
    """Return a HuggingFace LogitsProcessorList for dual-vocab generation."""
    return LogitsProcessorList(
        [CoTVocabConstraint(V=V, think_end_id=think_end_id, eos_id=eos_id, end_bias=end_bias)]
    )


# ---------------------------------------------------------------------------
# vLLM per-request callable logits processor
# ---------------------------------------------------------------------------

class _VllmDualVocabProcessor:
    """
    Per-request callable logits processor compatible with vLLM SamplingParams.

    vLLM signature: (output_token_ids: List[int], logits: torch.Tensor) -> torch.Tensor
    Each request gets its own instance so no shared mutable state across requests.
    """

    def __init__(self, V: int, think_end_id: int, eos_id: int, end_bias: float = 5.0):
        self.V = V
        self.think_end_id = think_end_id
        self.eos_id = eos_id
        self.end_bias = end_bias

        # Mask tensors — lazily created on first call (device-aware)
        self._allow_think: torch.Tensor | None = None
        self._allow_answer: torch.Tensor | None = None
        self._vocab_size: int = -1

    def _build_masks(self, device: torch.device, vocab_size: int) -> None:
        V = self.V
        allow_think = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        allow_think[V: 2 * V] = True
        allow_think[self.think_end_id] = True
        allow_think[self.eos_id] = True

        allow_answer = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        allow_answer[:V] = True
        allow_answer[self.eos_id] = True

        self._allow_think = allow_think
        self._allow_answer = allow_answer
        self._vocab_size = vocab_size

    def __call__(self, output_token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        vocab_size = logits.shape[-1]
        if self._vocab_size != vocab_size:
            self._build_masks(logits.device, vocab_size)

        ended = self.think_end_id in output_token_ids
        if ended:
            logits[~self._allow_answer] = float("-inf")
        else:
            logits[~self._allow_think] = float("-inf")
            logits[self.think_end_id] += self.end_bias

        return logits


def make_vllm_logits_processor(
    V: int,
    think_end_id: int,
    eos_id: int,
    end_bias: float = 5.0,
) -> _VllmDualVocabProcessor:
    """
    Return a per-request vLLM-compatible logits processor.

    Usage with SamplingParams:
        proc = make_vllm_logits_processor(V, think_end_id, eos_id)
        params = SamplingParams(..., logits_processors=[proc])

    Each generate() call with this processor creates independent state per
    request automatically (no reset needed).
    """
    return _VllmDualVocabProcessor(V=V, think_end_id=think_end_id, eos_id=eos_id, end_bias=end_bias)
