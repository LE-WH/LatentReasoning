"""Environment wrappers for reward shaping.

LengthRewardEnvWrapper transforms binary reward into length-aware continuous
reward, shared between RL and SFT pipelines.
"""

from typing import Any, Dict, Tuple

from ragen.env.base import BaseEnv
from ragen.env.reward_funcs import init_reward_func


class LengthRewardEnvWrapper(BaseEnv):
    """Wraps any environment to modify reward based on response length.

    The inner environment provides correctness (binary reward).
    This wrapper transforms it: wrong -> 0, correct -> f(length).
    """

    def __init__(self, inner_env: BaseEnv, reward_func_config: dict):
        super().__init__()
        self.inner_env = inner_env
        self.reward_func = init_reward_func(reward_func_config)
        self._cumulative_length = 0

    def reset(self, seed=None, **kwargs) -> Any:
        self._cumulative_length = 0
        return self.inner_env.reset(seed=seed, **kwargs)

    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        obs, original_reward, done, info = self.inner_env.step(action)

        # Track cumulative response length (character count; token count
        # can be used via a subclass or config option if needed).
        self._cumulative_length += len(str(action))

        new_reward = self.reward_func(original_reward, self._cumulative_length)

        info["original_reward"] = original_reward
        info["response_length"] = self._cumulative_length

        return obs, new_reward, done, info

    # Delegate optional methods to inner env
    def render(self, mode: str = "text") -> Any:
        return self.inner_env.render(mode)

    def compute_reward(self, action, **kwargs) -> float:
        return self.inner_env.compute_reward(action, **kwargs)

    def close(self):
        return self.inner_env.close()
