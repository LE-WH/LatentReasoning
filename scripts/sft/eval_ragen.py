"""Evaluate SFT model using RAGEN's evaluation framework.

Applies transformers 5.x compatibility patch, then runs ragen.llm_agent.agent_proxy.

Usage (from project root):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/sft/eval_ragen.py \
        --config-name _11_gsm8k \
        model_path=results/sft/gsm8k/qwen2.5-3b-self-training-concise
"""

# Fix transformers 5.x: vLLM accesses all_special_tokens_extended
# which was removed in transformers 5.x.
import transformers.tokenization_utils_base as _tub
if not hasattr(_tub.PreTrainedTokenizerBase, "all_special_tokens_extended"):
    @property
    def _all_special_tokens_extended(self):
        return list(set(self.all_special_tokens))
    _tub.PreTrainedTokenizerBase.all_special_tokens_extended = _all_special_tokens_extended

# Run as if: python -m ragen.llm_agent.agent_proxy
# The if __name__ guard is required for vllm v1 engine which uses
# multiprocessing spawn: without it, spawned subprocesses re-execute
# module-level code and trigger a recursive spawn error.
if __name__ == "__main__":
    import runpy
    runpy.run_module("ragen.llm_agent.agent_proxy", run_name="__main__")
