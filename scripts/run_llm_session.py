#!/usr/bin/env python3
"""Run a human-play session with LLM difficulty control."""

import argparse
import sys
sys.path.insert(0, ".")

import json
import requests
from datetime import datetime
from src.human_play_env import HumanPlayEnvWrapper
from src.discretizer import discretize_stats
from src.baseline_controllers import create_controller
from src.episode_logger import EpisodeLogger

API_URL = "http://localhost:8000"


def push_to_dashboard(stats: dict, labels: dict, decision: dict, difficulty: int):
    """Push data to the dashboard API."""
    try:
        # Push features (stats + labels)
        features = {**stats, **{f"label_{k}": v for k, v in labels.items()}}
        features["difficulty"] = difficulty
        requests.post(f"{API_URL}/features", json={"features": features}, timeout=0.5)

        # Push LLM output (decision)
        requests.post(f"{API_URL}/llm_output", json={"output_text": json.dumps(decision, indent=2)}, timeout=0.5)

        # Push state
        state = f"Episode complete | Difficulty: {difficulty} | Next: {decision.get('difficulty', '?')}"
        requests.post(f"{API_URL}/state", json={"state": state}, timeout=0.5)
    except requests.RequestException:
        pass  # Dashboard not running, ignore


def main():
    parser = argparse.ArgumentParser(description="Run DDA session with difficulty control")
    parser.add_argument("--controller", type=str, default="llm",
                        choices=["llm", "fixed", "rule_based"],
                        help="Controller type (default: llm)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes (default: 10)")
    parser.add_argument("--difficulty", type=int, default=3,
                        help="Initial difficulty 1-5 (default: 3)")
    parser.add_argument("--fixed-level", type=int, default=3,
                        help="Fixed difficulty level for fixed controller (default: 3)")
    args = parser.parse_args()

    # Initialize components
    env = HumanPlayEnvWrapper(window_visible=True)

    if args.controller == "fixed":
        controller = create_controller("fixed", difficulty=args.fixed_level)
    else:
        controller = create_controller(args.controller)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = EpisodeLogger(f"runs/{timestamp}_{args.controller}.jsonl")

    difficulty = args.difficulty
    history = []

    print(f"\n=== Starting {args.controller.upper()} Session ===")
    print(f"Episodes: {args.episodes}")
    print(f"Initial difficulty: {difficulty}")
    print("Use WASD to move, mouse to aim, click to shoot\n")

    try:
        for episode in range(args.episodes):
            print(f"\n--- Episode {episode + 1}/{args.episodes} (Difficulty: {difficulty}) ---")

            # Play episode
            env.reset(difficulty)
            stats = env.run_episode()

            # Process results
            labels = discretize_stats(stats)
            decision = controller.get_difficulty_decision(labels, difficulty, history)

            # Log episode
            logger.log_episode(
                episode_number=episode,
                raw_stats=stats,
                labels=labels,
                difficulty=difficulty,
                decision=decision,
            )

            # Push to dashboard API
            push_to_dashboard(stats, labels, decision, difficulty)

            # Print summary
            print(f"Stats: kills={stats['kills']}, deaths={stats['deaths']}, kdr={stats['kdr']}")
            print(f"Labels: {labels}")
            print(f"Decision: {decision['reasoning']}")
            print(f"Next difficulty: {decision['difficulty']}")

            # Update for next episode
            history.append(decision)
            difficulty = decision["difficulty"]

    except KeyboardInterrupt:
        print("\n\nSession interrupted by user")
    finally:
        env.close()
        print(f"\n=== Session Complete ===")
        print(f"Log saved to: {logger.log_path}")


if __name__ == "__main__":
    main()
