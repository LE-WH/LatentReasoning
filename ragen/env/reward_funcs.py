"""Reward function factory for length-aware reward shaping.

Shared by both RL (via LengthRewardEnvWrapper) and SFT (via direct scoring).
"""

from typing import Callable


def linear_length_reward(original_reward: float, response_length: int, max_length: int) -> float:
    """Correct + short = high reward. Wrong = 0.

    reward = original_reward * max(0, 1 - length / max_length)

    Examples (max_length=512):
        wrong, any length   -> 0.0
        correct, length=0   -> 1.0
        correct, length=256 -> 0.5
        correct, length=512 -> 0.0
    """
    if original_reward <= 0:
        return 0.0
    return original_reward * max(0.0, 1.0 - response_length / max_length)


def init_reward_func(config: dict) -> Callable[[float, int], float]:
    """Factory: returns a reward function based on config.

    Args:
        config: dict with keys:
            - type: "linear" (more types can be added later)
            - max_length: int, maximum expected response length

    Returns:
        reward_func(original_reward, response_length) -> float
    """
    reward_type = config.get("type", "linear")
    max_length = config["max_length"]

    if reward_type == "linear":
        return lambda orig, length: linear_length_reward(orig, length, max_length)
    else:
        raise ValueError(f"Unknown reward function type: {reward_type}")
