"""Compatibility patches for transformers 5.x + vLLM 0.8.x.

Call ``apply_patches()`` before importing vLLM.
"""

import os
from functools import lru_cache


@lru_cache(maxsize=1)
def apply_patches() -> None:
    """Apply all environment and monkey-patch fixes."""

    # vLLM defaults to fork which causes "Cannot re-initialize CUDA in
    # forked subprocess" errors.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    # Keep hub downloads quiet without forcing the process offline.
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    # 1. transformers 5.x removed ``all_special_tokens_extended`` but
    #    vLLM's ``get_cached_tokenizer`` still accesses it.
    import transformers.tokenization_utils_base as _tub

    if not hasattr(_tub.PreTrainedTokenizerBase, "all_special_tokens_extended"):

        @property  # type: ignore[misc]
        def _all_special_tokens_extended(self):
            return list(set(self.all_special_tokens))

        _tub.PreTrainedTokenizerBase.all_special_tokens_extended = (
            _all_special_tokens_extended
        )

    # 2. vLLM 0.8.x ships a DisabledTqdm that unconditionally forwards
    #    disable=True, while newer huggingface_hub already passes disable=...
    #    into snapshot_download(). Make the class tolerate either path.
    try:
        from vllm.model_executor.model_loader import weight_utils as _weight_utils

        class _PatchedDisabledTqdm(_weight_utils.tqdm):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("disable", True)
                super().__init__(*args, **kwargs)

        _weight_utils.DisabledTqdm = _PatchedDisabledTqdm
    except Exception:
        # Some code paths do not touch HF downloads at all, so keep the
        # patch best-effort rather than failing module import eagerly.
        pass
