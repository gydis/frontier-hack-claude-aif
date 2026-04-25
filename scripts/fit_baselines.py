"""
Compute baselines.json from collected bot rollout data.
Run after collect_baselines.py finishes.

Usage:
    micromamba run -n doom python scripts/fit_baselines.py
    micromamba run -n doom python scripts/fit_baselines.py --scenario deathmatch
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.baselines import BotBaselines

SCENARIO_CONFIGS = {
    "deathmatch":      "config/dda_deathmatch.cfg",
    "deadly_corridor": "config/dda_deadly_corridor.cfg",
    "defend_center":   "config/dda_defend_center.cfg",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="deathmatch", choices=list(SCENARIO_CONFIGS))
    parser.add_argument("--data-dir", default="data/baselines", help="Root of rollout data")
    parser.add_argument("--out", default="data/baselines.json", help="Output path")
    args = parser.parse_args()

    data_dir = os.path.join(args.data_dir, args.scenario)
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Run collect_baselines.py first.")
        sys.exit(1)

    print(f"Computing baselines from: {data_dir}")
    baselines = BotBaselines()
    baselines.compute_from_dir(data_dir, scenario=args.scenario)

    # Show what we found
    print("\nBaseline summary:")
    for skill in range(1, 6):
        m_kdr, s_kdr = baselines.get(skill, "kdr")
        m_hp, s_hp   = baselines.get(skill, "health_mean")
        print(f"  Skill {skill}: KDR={m_kdr:.2f}±{s_kdr:.2f}  HP={m_hp:.1f}±{s_hp:.1f}")

    baselines.save(args.out)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
