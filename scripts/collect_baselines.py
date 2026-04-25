"""
Collect bot rollouts at all skill levels (1-5) for a given scenario.
Run this overnight before a demo session.

Usage:
    micromamba run -n doom python scripts/collect_baselines.py
    micromamba run -n doom python scripts/collect_baselines.py --scenario deathmatch --episodes 10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import vizdoom as vzd
from src.collector import BotRolloutRunner

SCENARIO_CONFIGS = {
    "deathmatch":     "config/dda_deathmatch.cfg",
    "deadly_corridor": "config/dda_deadly_corridor.cfg",
    "defend_center":   "config/dda_defend_center.cfg",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="deathmatch", choices=list(SCENARIO_CONFIGS))
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per skill level")
    parser.add_argument("--bots", type=int, default=3, help="Number of bots to add")
    parser.add_argument("--out", default="data/baselines", help="Output directory")
    parser.add_argument("--visible", action="store_true", help="Show game window")
    args = parser.parse_args()

    cfg_path = SCENARIO_CONFIGS[args.scenario]
    if not os.path.exists(cfg_path):
        print(f"Config not found: {cfg_path}")
        sys.exit(1)

    out_dir = os.path.join(args.out, args.scenario)
    print(f"Collecting {args.episodes} episodes × 5 skill levels for '{args.scenario}'")
    print(f"Output: {out_dir}")
    print()

    runner = BotRolloutRunner(
        scenario_cfg=cfg_path,
        num_episodes=args.episodes,
        num_bots=args.bots,
        window_visible=args.visible,
    )
    stats = runner.run_all_skills(out_dir)

    print("\nDone. Summary:")
    for skill, eps in stats.items():
        kdrs = [e["kdr"] for e in eps]
        print(f"  Skill {skill}: avg KDR = {sum(kdrs)/len(kdrs):.2f}")

    print(f"\nNow run: python scripts/fit_baselines.py --scenario {args.scenario}")


if __name__ == "__main__":
    main()
