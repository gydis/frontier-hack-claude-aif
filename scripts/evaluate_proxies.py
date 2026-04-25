#!/usr/bin/env python3
"""Evaluate player proxies against VizDoom difficulty levels and save calibration stats."""

from src.player_proxy import BuiltInBotProxy, ModelCheckpointProxy
from src.env_wrapper import DeathmatchEnvWrapper
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


SCENARIO_CONFIGS = {
    "deathmatch": "config/dda_deathmatch.cfg",
    "deadly_corridor": "config/dda_deadly_corridor.cfg",
    "defend_center": "config/dda_defend_center.cfg",
}


def run_proxy_eval(proxy, env, difficulty, episodes):
    records = []
    for ep in range(episodes):
        env.reset({"bot_skill": difficulty, "num_bots": 2})
        if hasattr(proxy, "reset"):
            proxy.reset()

        done = False
        steps = 0
        while not done and steps < 2000:
            obs = env.get_state()
            action = proxy.act(obs)
            _, done = env.step(action)
            steps += 1

        stats = env.get_episode_stats()
        records.append({
            "episode": ep,
            "difficulty": difficulty,
            "proxy_id": proxy.id,
            "stats": stats,
        })
        print(
            f"proxy={proxy.id} diff={difficulty} ep={ep} "
            f"kdr={stats.get('kdr', 0):.2f} "
            f"damage={stats.get('damagecount', 0):.0f}"
        )
    return records


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate proxy performance for calibration.")
    parser.add_argument("--scenario", default="deathmatch",
                        choices=list(SCENARIO_CONFIGS))
    parser.add_argument("--proxy", default="builtin",
                        choices=["builtin", "model"])
    parser.add_argument("--model-path", type=str,
                        help="Path to a pretrained .pth model checkpoint")
    parser.add_argument("--difficulty-levels", default="1,3,5",
                        help="Comma-separated bot difficulty levels to evaluate")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out", default="data/calibration_stats.json")
    parser.add_argument("--visible", action="store_true")
    args = parser.parse_args()

    cfg_path = SCENARIO_CONFIGS[args.scenario]
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Scenario config not found: {cfg_path}")

    if args.proxy == "model" and not args.model_path:
        raise ValueError("--model-path is required when --proxy=model")

    env = DeathmatchEnvWrapper(
        config_path=cfg_path, window_visible=args.visible)
    proxy = None
    if args.proxy == "builtin":
        proxy = BuiltInBotProxy(skill_level=0.5, exploration_rate=0.1)
    else:
        proxy = ModelCheckpointProxy(
            model_path=args.model_path,
            skill_level=0.5,
            exploration_rate=0.05,
        )

    difficulty_levels = [
        int(x) for x in args.difficulty_levels.split(",") if x.strip()]
    all_records = []
    for difficulty in difficulty_levels:
        all_records.extend(run_proxy_eval(
            proxy, env, difficulty, args.episodes))

    env.close()

    payload = {
        "scenario": args.scenario,
        "proxy_type": args.proxy,
        "model_path": args.model_path or "",
        "difficulty_levels": difficulty_levels,
        "episodes_per_level": args.episodes,
        "records": all_records,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved calibration stats to {args.out}")


if __name__ == "__main__":
    main()
