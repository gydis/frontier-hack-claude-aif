#!/usr/bin/env python3
"""
Stub end-to-end run script for Phase 1 verification.
Runs episodes with stub adapter/proxy, logs to RunLogger.

Usage:
    python scripts/stub_run.py
    python scripts/stub_run.py --real-env
    python scripts/stub_run.py --config config/example_experiment.yaml
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.run_logger import JsonlRunLogger
from src.stubs import StubAdapter, StubPlayerProxy


def run_stub_episode(
    episode_idx: int, adapter, proxy, logger, use_real_env: bool
) -> None:
    """Run one episode and log results."""
    difficulty = adapter.choose_difficulty()

    if use_real_env:
        from src.env_wrapper import DeathmatchEnvWrapper

        env = DeathmatchEnvWrapper(window_visible=False)
        env.reset(difficulty)

        done = False
        step_count = 0
        max_steps = 2000
        while not done and step_count < max_steps:
            obs = env.get_state()
            action = proxy.act(obs)
            _, done = env.step(action)
            step_count += 1

        raw_stats = env.get_episode_stats()
        env.close()
    else:
        raw_stats = {
            "frags": random.randint(0, 10),
            "deaths": random.randint(0, 5),
            "kdr": random.uniform(0.5, 3.0),
            "kill_rate": random.uniform(0.01, 0.1),
            "health_mean": random.uniform(50, 100),
            "damage_efficiency": random.uniform(0.5, 2.0),
            "survival_frac": random.uniform(0.5, 1.0),
            "duration_seconds": random.uniform(30, 120),
            "accuracy": random.uniform(0.1, 0.5),
            "damage_taken": random.randint(50, 200),
            "damage_dealt": random.randint(100, 500),
        }

    perf = 1
    kdr = raw_stats.get("kdr", 1.0)
    if kdr < 0.8:
        perf = 0
    elif kdr > 2.0:
        perf = 2

    obs_aif = {"performance": perf, "trend": 1}

    adapter.update(obs_aif)

    logger.write(
        {
            "episode_idx": episode_idx,
            "difficulty": difficulty,
            "raw_stats": raw_stats,
            "obs_aif": obs_aif,
            "belief": adapter.get_belief(),
            "proxy_id": proxy.id,
            "adapter_type": "stub",
        }
    )

    print(
        f"Episode {episode_idx}: frags={raw_stats.get('frags', 'N/A')}, "
        f"deaths={raw_stats.get('deaths', 'N/A')}, perf={['LOW','MED','HIGH'][perf]}"
    )


def main():
    parser = argparse.ArgumentParser(description="Stub end-to-end run")
    parser.add_argument("--real-env", action="store_true", help="Use real ViZDoom env")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--config", type=str, help="Path to YAML config (optional)")
    args = parser.parse_args()

    adapter = StubAdapter(bot_skill=3, num_bots=2)
    proxy = StubPlayerProxy("stub_random")
    logger = JsonlRunLogger("stub_run")

    print("\n=== Stub End-to-End Run ===")
    print(f"Episodes: {args.episodes}")
    print(f"Real env: {args.real_env}")
    print(f"Log file: {logger.filepath}\n")

    for ep in range(args.episodes):
        run_stub_episode(ep, adapter, proxy, logger, args.real_env)

    logger.close()
    print(f"\n=== Complete. Log written to {logger.filepath} ===")


if __name__ == "__main__":
    main()
