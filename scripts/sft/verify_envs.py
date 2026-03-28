"""Verify GSM8k and DeepCoder environments work correctly.

Tests:
1. Direct Python instantiation of StaticEnv (GSM8k) and DeepCoderEnv
2. reset() → observation format
3. step(correct_answer) → reward check
4. step(wrong_answer) → reward check
5. Simulated Hydra config dict → StaticEnvConfig construction
"""

import sys


def test_gsm8k_direct() -> bool:
    """Test GSM8k via direct StaticEnv instantiation."""
    print("=" * 60)
    print("TEST 1: GSM8k StaticEnv — direct instantiation")
    print("=" * 60)

    from ragen.env.static.config import StaticEnvConfig
    from ragen.env.static.env import StaticEnv

    config = StaticEnvConfig(dataset_name="gsm8k", cache_dir="./data")
    env = StaticEnv(config)

    obs = env.reset(seed=42)
    print(f"  Observation type: {type(obs).__name__}")
    print(f"  Observation preview: {obs[:200]}...")
    print(f"  Correct answer: {env.correct_answer}")
    assert isinstance(obs, str), f"Expected str observation, got {type(obs)}"

    # Step with correct answer
    obs_correct, reward_correct, done_correct, info_correct = env.step(env.correct_answer)
    print(f"  step(correct): reward={reward_correct}, done={done_correct}, info={info_correct}")
    assert reward_correct == 1.0, f"Expected reward 1.0 for correct answer, got {reward_correct}"
    assert done_correct is True, "Expected done=True for correct answer"

    # Reset and step with wrong answer
    env.reset(seed=42)
    obs_wrong, reward_wrong, done_wrong, info_wrong = env.step("999999999")
    print(f"  step(wrong):   reward={reward_wrong}, done={done_wrong}, info={info_wrong}")
    assert reward_wrong == 0.0, f"Expected reward 0.0 for wrong answer, got {reward_wrong}"

    print("  PASSED\n")
    return True


def test_gsm8k_from_hydra_dict() -> bool:
    """Test constructing StaticEnv from a Hydra-style config dict.

    Simulates what happens when RAGEN loads envs.yaml:
        env_config:
          dataset_name: gsm8k
          cache_dir: ./data
    and passes it as **kwargs to StaticEnvConfig.
    """
    print("=" * 60)
    print("TEST 2: GSM8k StaticEnv — from Hydra config dict")
    print("=" * 60)

    from ragen.env.static.config import StaticEnvConfig
    from ragen.env.static.env import StaticEnv

    # Simulate the dict that would come from envs.yaml env_config
    hydra_env_config = {
        "dataset_name": "gsm8k",
        "cache_dir": "./data",
    }
    config = StaticEnvConfig(**hydra_env_config)
    env = StaticEnv(config)

    obs = env.reset(seed=123)
    print(f"  Observation type: {type(obs).__name__}")
    print(f"  Observation preview: {obs[:200]}...")
    print(f"  Correct answer: {env.correct_answer}")
    assert isinstance(obs, str), f"Expected str observation, got {type(obs)}"

    obs_correct, reward_correct, done_correct, info_correct = env.step(env.correct_answer)
    print(f"  step(correct): reward={reward_correct}, done={done_correct}")
    assert reward_correct == 1.0, f"Expected reward 1.0, got {reward_correct}"

    # Also verify via REGISTERED_ENVS lookup
    from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
    assert "static" in REGISTERED_ENVS, "StaticEnv not registered in REGISTERED_ENVS"
    assert "static" in REGISTERED_ENV_CONFIGS, "StaticEnvConfig not registered in REGISTERED_ENV_CONFIGS"
    env_cls = REGISTERED_ENVS["static"]
    config_cls = REGISTERED_ENV_CONFIGS["static"]
    config2 = config_cls(**hydra_env_config)
    env2 = env_cls(config2)
    obs2 = env2.reset(seed=123)
    assert obs2 == obs, "Observation mismatch between direct and registry-based instantiation"
    print("  Registry lookup matches direct instantiation")

    print("  PASSED\n")
    return True


def test_deepcoder() -> bool:
    """Test DeepCoder environment."""
    print("=" * 60)
    print("TEST 3: DeepCoderEnv")
    print("=" * 60)

    from ragen.env.deepcoder.config import DeepCoderEnvConfig
    from ragen.env.deepcoder.env import DeepCoderEnv

    config = DeepCoderEnvConfig(cache_dir="./data")
    env = DeepCoderEnv(config)

    obs = env.reset(seed=42)
    print(f"  Observation type: {type(obs).__name__}")
    print(f"  Observation preview: {str(obs)[:300]}...")
    assert obs is not None, "Expected non-None observation"

    # Step with empty action (should be invalid)
    obs_empty, reward_empty, done_empty, info_empty = env.step("")
    print(f"  step(empty):  reward={reward_empty}, done={done_empty}, valid={info_empty.get('action_is_valid')}")
    assert reward_empty == 0.0, f"Expected reward 0.0 for empty action, got {reward_empty}"

    # Step with a trivially wrong answer
    env.reset(seed=42)
    obs_wrong, reward_wrong, done_wrong, info_wrong = env.step("print('hello')")
    print(f"  step(wrong):  reward={reward_wrong}, done={done_wrong}")

    # Step with the canonical solution (if available)
    env.reset(seed=42)
    if env.current_solution:
        obs_sol, reward_sol, done_sol, info_sol = env.step(env.current_solution)
        print(f"  step(canonical): reward={reward_sol}, done={done_sol}, "
              f"passed={info_sol.get('passed_tests')}/{info_sol.get('total_tests')}")
    else:
        print("  No canonical solution available for this problem, skipping solution test")

    print("  PASSED\n")
    return True


def main() -> None:
    results = {}
    for name, test_fn in [
        ("gsm8k_direct", test_gsm8k_direct),
        ("gsm8k_hydra", test_gsm8k_from_hydra_dict),
        ("deepcoder", test_deepcoder),
    ]:
        try:
            results[name] = test_fn()
        except Exception as e:
            print(f"  FAILED: {e}\n")
            results[name] = False

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if not all_passed:
        sys.exit(1)
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
