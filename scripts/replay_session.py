"""
Replay a recorded session and generate analysis plots.

Usage:
    micromamba run -n doom python scripts/replay_session.py --log data/sessions/<ts>/session_log.jsonl
    micromamba run -n doom python scripts/replay_session.py --log data/sessions/<ts>/session_log.jsonl --plot
    micromamba run -n doom python scripts/replay_session.py --lmp data/recordings/ep0000.lmp --scenario deathmatch
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

SCENARIO_CONFIGS = {
    "deathmatch":      "config/dda_deathmatch.cfg",
    "deadly_corridor": "config/dda_deadly_corridor.cfg",
    "defend_center":   "config/dda_defend_center.cfg",
}


def plot_session(log_entries: list[dict], out_path: str | None = None):
    import matplotlib
    if out_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes  = [e["episode"] for e in log_entries]
    skills    = [e["skill"] for e in log_entries]
    states    = [e["dominant_state"] for e in log_entries]
    perf_obs  = [e["perf_obs"] for e in log_entries]
    kdrs      = [e["stats"].get("kdr", 0) for e in log_entries]
    hp_means  = [e["stats"].get("health_mean", 0) for e in log_entries]

    state_colors = {"FRUSTRATED": "red", "FLOW": "green", "BORED": "gold"}
    colors = [state_colors.get(s, "gray") for s in states]

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("DDA Session Analysis — Active Inference DM", fontsize=14)

    # Skill level
    axes[0].plot(episodes, skills, color="purple", linewidth=2, marker="o", markersize=5)
    axes[0].set_ylabel("Skill Level")
    axes[0].set_ylim(0.5, 5.5)
    axes[0].set_yticks([1, 2, 3, 4, 5])
    axes[0].grid(True, alpha=0.3)
    for ep, s, c in zip(episodes, skills, colors):
        axes[0].plot(ep, s, "o", color=c, markersize=8, zorder=5)

    # KDR
    axes[1].bar(episodes, kdrs, color=colors, alpha=0.8)
    axes[1].set_ylabel("Kill/Death Ratio")
    axes[1].axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    # Health mean
    axes[2].plot(episodes, hp_means, color="blue", linewidth=2, marker="s", markersize=5)
    axes[2].set_ylabel("Mean Health")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylim(0, 110)
    axes[2].grid(True, alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend = [Patch(color=c, label=s) for s, c in state_colors.items()]
    axes[0].legend(handles=legend, loc="upper right", fontsize=9)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Plot saved to {out_path}")
    else:
        plt.show()


def replay_lmp(lmp_path: str, scenario: str):
    import vizdoom as vzd
    cfg = SCENARIO_CONFIGS.get(scenario)
    if not cfg or not os.path.exists(cfg):
        print(f"Unknown scenario or config missing: {scenario}")
        sys.exit(1)

    game = vzd.DoomGame()
    game.load_config(cfg)
    game.set_window_visible(True)
    game.init()
    game.replay_episode(lmp_path)
    while not game.is_episode_finished():
        game.advance_action(1)
    game.close()
    print("Replay finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",      help="Path to session_log.jsonl")
    parser.add_argument("--plot",     action="store_true", help="Show analysis plots")
    parser.add_argument("--save-plot", help="Save plot to this path instead of showing")
    parser.add_argument("--lmp",      help="Replay a specific .lmp file")
    parser.add_argument("--scenario", default="deathmatch", choices=list(SCENARIO_CONFIGS))
    args = parser.parse_args()

    if args.lmp:
        replay_lmp(args.lmp, args.scenario)
        return

    if not args.log:
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.log):
        print(f"Log not found: {args.log}")
        sys.exit(1)

    with open(args.log) as f:
        entries = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(entries)} episodes from {args.log}")
    print()

    # Text summary
    for e in entries:
        s = e["stats"]
        print(
            f"  ep{e['episode']:03d} | skill={e['skill']} "
            f"kills={s.get('final_kills', 0):.0f} deaths={s.get('final_deaths', 0):.0f} "
            f"kdr={s.get('kdr', 0):.2f} | {e['dominant_state']:10s} → {e['action_desc']}"
        )

    if args.plot or args.save_plot:
        plot_session(entries, out_path=args.save_plot)


if __name__ == "__main__":
    main()
