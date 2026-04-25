#!/usr/bin/env python3
"""
CLI for running and inspecting VizDoom deathmatch recording sessions.

Run a new session:
    python run_session.py --bots 3 --minutes 5 --output sessions/

Run a short smoke-test (10 s):
    python run_session.py --minutes 0.17

Inspect a saved session:
    python run_session.py --load sessions/2025-01-15_abc123.parquet
"""

import argparse
import logging
import sys
from pathlib import Path


def _run(args: argparse.Namespace) -> None:
    from vizdoom_tracker import DeathmatchSession

    session = DeathmatchSession(
        num_bots=args.bots,
        episode_minutes=args.minutes,
        sample_hz=args.hz,
        output_dir=args.output,
        headless=not args.show,
        seed=args.seed,
        mode=args.mode,
        scenario=args.scenario
    )
    result = session.run()

    print(f"\n{'─'*60}")
    print(f"Session : {result.metadata.session_id}")
    print(f"Samples : {len(result.df)}")
    print(f"Columns : {list(result.df.columns)}")
    print(f"\n{result.df.describe().to_string()}")

    if args.output:
        print(f"\nSaved to: {args.output}/")


def _load(path: str) -> None:
    from vizdoom_tracker import SessionResult

    result = SessionResult.load(path)
    meta = result.metadata

    print(f"\n{'─'*60}")
    if meta:
        print(f"Session ID  : {meta.session_id}")
        print(f"Started     : {meta.start_utc}")
        print(f"Bots        : {meta.num_bots}")
        print(f"Duration    : {meta.episode_timeout_tics / meta.tic_rate / 60:.1f} min")
        print(f"Sample rate : {meta.tic_rate / meta.sample_interval_tics:.2f} Hz")
        print(f"Variables   : {len(meta.variables)}")
        print(f"Samples     : {meta.total_samples}")
    print(f"\nDataFrame shape: {result.df.shape}")
    print(f"\n{result.df.describe().to_string()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VizDoom deathmatch game-variable recorder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bots", type=int, default=3,
                        help="Number of VizDoom bots to add as opponents")
    parser.add_argument("--minutes", type=float, default=5.0,
                        help="Episode duration in minutes")
    parser.add_argument("--hz", type=float, default=1.0,
                        help="Sampling rate (snapshots per second)")
    parser.add_argument("--output", default="sessions",
                        help="Directory to write Parquet + JSON output")
    parser.add_argument("--show", action="store_true",
                        help="Show the game window (disables headless mode)")
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducibility")
    parser.add_argument("--load", metavar="PATH",
                        help="Load and inspect an existing session Parquet file")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging")
    parser.add_argument("--mode", dest="mode", type=str, default="player",
                        choices=["player", "spectator"],
                        help="VizDoom mode: 'player' (default) or 'spectator'")
    parser.add_argument("--scenario", dest="scenario", type=str, default="deathmatch.cfg",
                        help="Name of VizDoom scenario config file")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.load:
        _load(args.load)
    else:
        _run(args)


if __name__ == "__main__":
    main()
