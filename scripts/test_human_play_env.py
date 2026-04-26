#!/usr/bin/env python3
"""
Test script for HumanPlayEnvWrapper.

Run with: python scripts/test_human_play_env.py
Requires VizDoom to be installed and a display available.
"""

import sys
sys.path.insert(0, ".")

from src.human_play_env import HumanPlayEnvWrapper, DIFFICULTY_MAP


def test_difficulty_bounds():
    """Test that difficulty bounds are enforced."""
    env = HumanPlayEnvWrapper()

    for d in [0, 6, -1, 10]:
        try:
            env.reset(d)
            print(f"FAIL: reset({d}) should have raised ValueError")
            return False
        except ValueError as e:
            print(f"OK: reset({d}) raised ValueError: {e}")

    for d in [1, 2, 3, 4, 5]:
        try:
            env.reset(d)
            print(f"OK: reset({d}) succeeded")
            env.close()
        except Exception as e:
            print(f"FAIL: reset({d}) raised unexpected error: {e}")
            return False

    return True


def test_difficulty_mapping():
    """Verify difficulty mapping table."""
    print("\nDifficulty mapping:")
    for level, settings in DIFFICULTY_MAP.items():
        print(f"  Level {level}: skill={settings['bot_skill']}, "
              f"bots={settings['num_bots']}, "
              f"time={settings['time_limit']/35:.0f}s, "
              f"frags={settings['frag_limit']}")
    return True


def test_callback_registration():
    """Test callback mechanism."""
    env = HumanPlayEnvWrapper()

    callback_fired = []
    def my_callback(stats):
        callback_fired.append(stats)
        print(f"Callback received stats: {stats}")

    env.on_episode_end(my_callback)
    print(f"OK: Registered callback, {len(env._callbacks)} callbacks registered")
    env.close()
    return True


def test_stats_keys():
    """Verify expected stats keys are returned."""
    expected_keys = {
        "kills", "deaths", "kdr", "frags",
        "damage_dealt", "damage_taken", "accuracy",
        "duration_seconds", "health_mean", "difficulty"
    }

    env = HumanPlayEnvWrapper()
    stats = env.get_episode_stats()

    if not stats:
        print("OK: get_episode_stats() returns empty dict before reset (expected)")

    env.reset(3)
    stats = env.get_episode_stats()

    if set(stats.keys()) == expected_keys:
        print(f"OK: Stats have expected keys: {sorted(stats.keys())}")
        env.close()
        return True
    else:
        missing = expected_keys - set(stats.keys())
        extra = set(stats.keys()) - expected_keys
        print(f"FAIL: Missing keys: {missing}, Extra keys: {extra}")
        env.close()
        return False


def main():
    print("=== HumanPlayEnvWrapper Tests ===\n")

    results = []

    print("1. Testing difficulty mapping...")
    results.append(test_difficulty_mapping())

    print("\n2. Testing callback registration...")
    results.append(test_callback_registration())

    print("\n3. Testing stats keys...")
    results.append(test_stats_keys())

    print("\n4. Testing difficulty bounds...")
    results.append(test_difficulty_bounds())

    print("\n" + "=" * 40)
    if all(results):
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
