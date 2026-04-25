"""
Main entry point: run a live human-play DDA session.

Usage:
    micromamba run -n doom python scripts/run_session.py
    micromamba run -n doom python scripts/run_session.py --scenario deadly_corridor --skill 2 --episodes 15
    micromamba run -n doom python scripts/run_session.py --record              # saves .lmp replays
    micromamba run -n doom python scripts/run_session.py --dashboard           # show pygame overlay
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.game_loop import DDAPipeline

SCENARIO_CONFIGS = {
    "deathmatch":      "config/dda_deathmatch.cfg",
    "deadly_corridor": "config/dda_deadly_corridor.cfg",
    "defend_center":   "config/dda_defend_center.cfg",
}


def main():
    parser = argparse.ArgumentParser(description="Run DDA session with Active Inference DM")
    parser.add_argument("--scenario",   default="deathmatch", choices=list(SCENARIO_CONFIGS))
    parser.add_argument("--skill",      type=int, default=3, help="Starting skill level (1-5)")
    parser.add_argument("--episodes",   type=int, default=20)
    parser.add_argument("--baselines",  default="data/baselines.json")
    parser.add_argument("--session-dir", default="data/sessions")
    parser.add_argument("--record",     action="store_true", help="Record .lmp replay files")
    parser.add_argument("--dashboard",  action="store_true", help="Show pygame DM dashboard")
    parser.add_argument("--headless",   action="store_true", help="No game window (for bots/testing)")
    args = parser.parse_args()

    cfg = SCENARIO_CONFIGS[args.scenario]
    if not os.path.exists(cfg):
        print(f"Config not found: {cfg}")
        sys.exit(1)

    record_dir = "data/recordings" if args.record else None

    pipeline = DDAPipeline(
        scenario_cfg=cfg,
        baselines_path=args.baselines,
        initial_skill=args.skill,
        window_visible=not args.headless,
        record_dir=record_dir,
    )

    dashboard = None
    if args.dashboard:
        from src.dashboard import Dashboard
        dashboard = Dashboard()

    try:
        if dashboard:
            # Monkey-patch episode callback to update dashboard
            original_run_episode = pipeline._run_episode

            def episode_with_dashboard(ep_idx):
                entry = original_run_episode(ep_idx)
                dashboard.update(
                    dominant_state=entry["dominant_state"],
                    beliefs=entry["beliefs"],
                    action_desc=entry["action_desc"],
                    skill=entry["skill"],
                    perf_obs=entry["perf_obs"],
                )
                return entry

            pipeline._run_episode = episode_with_dashboard

        pipeline.run_session(
            num_episodes=args.episodes,
            session_dir=args.session_dir,
        )
    finally:
        pipeline.close()
        if dashboard:
            dashboard.close()


if __name__ == "__main__":
    main()
