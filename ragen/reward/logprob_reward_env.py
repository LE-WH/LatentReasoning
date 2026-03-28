"""StaticEnv wrapper that replaces binary reward with log-prob scorer reward.

Instead of binary 0/1 from exact match, computes the mean token log-probability
of the gold answer conditioned on (prompt + CoT), providing a continuous reward
signal for RL training.

Requires the scorer server to be running:
    python -m ragen.reward.scorer_server --model_dir <dual_model> --port 8009

Set SCORER_URL env var if using non-default endpoint.
"""
from __future__ import annotations

import logging
from typing import Optional

from ragen.env.static.env import StaticEnv
from ragen.env.static.config import StaticEnvConfig
from .scorer_client import compute_score

logger = logging.getLogger(__name__)


class LogProbRewardEnv(StaticEnv):
    """StaticEnv that uses log-prob scoring from the remote scorer server.

    The env still uses StaticEnv's exact-match to determine done (episode
    termination on correct answer), but replaces the reward with the
    continuous log-prob score from the scorer server.

    Reward semantics:
        - Typical range: [-10, 0] (mean token log-prob)
        - Higher (closer to 0) = model is more confident in gold answer
        - -100.0 = invalid/empty answer
        - -2.0 penalty added if </think> token is missing
    """

    def __init__(self, config: StaticEnvConfig):
        super().__init__(config)
        self._prompt_messages: Optional[list[dict[str, str]]] = None

    def reset(self, seed=None, mode=None):
        obs = super().reset(seed=seed, mode=mode)
        # Store prompt messages for scorer
        self._prompt_messages = [
            {"role": "user", "content": obs},
        ]
        return obs

    def step(self, action):
        # Get done/info from parent (uses exact match)
        observation, _binary_reward, done, info = super().step(action)

        # Compute continuous reward via scorer server
        try:
            reward = compute_score(
                prompt_messages=self._prompt_messages,
                solution_str=action,
                ground_truth=self.correct_answer,
            )
        except Exception as e:
            logger.warning(f"Scorer server error, falling back to binary reward: {e}")
            reward = _binary_reward

        info["binary_reward"] = _binary_reward
        info["logprob_reward"] = reward

        return observation, reward, done, info
