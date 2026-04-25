#!/usr/bin/env python3
"""
CLI for extracting behavioral features from VizDoom session recordings.

Process one session file:
    python extract_features.py sessions/2026-04-25_abc123.parquet

Process all sessions in a directory:
    python extract_features.py sessions/

Inspect a saved features file:
    python extract_features.py --load sessions/features/2026-04-25_abc123_features.parquet

Custom rolling windows and output directory:
    python extract_features.py sessions/ --output custom_features/ --windows 1 10 60
"""

import argparse
import logging
import sys
from pathlib import Path


def _process_one(path: Path, output_dir: Path, windows: list[int]) -> None:
    from vizdoom_tracker import SessionResult
    from vizdoom_tracker.features import extract_features

    result = SessionResult.load(path)
    fr = extract_features(result, windows=windows, source_path=str(path.resolve()))

    saved = fr.save(output_dir)

    meta = fr.metadata
    skipped_str = (
        f"{len(meta.features_skipped)} skipped ({', '.join(meta.features_skipped)})"
        if meta.features_skipped
        else "none skipped"
    )

    print(f"\n{'─'*60}")
    print(f"Session  : {meta.session_id}")
    print(f"Source   : {path}")
    print(f"Samples  : {meta.num_samples}")
    print(f"Features : {meta.num_features} computed, {skipped_str}")
    print(f"Saved to : {saved}")
    print(f"\n{fr.df.describe().to_string()}")


def _process_dir(dir_path: Path, output_dir: Path, windows: list[int]) -> None:
    session_files = [
        p for p in sorted(dir_path.glob("*.parquet"))
        if not p.name.endswith("_features.parquet")
    ]
    if not session_files:
        print(f"No session parquet files found in {dir_path}", file=sys.stderr)
        return
    for p in session_files:
        _process_one(p, output_dir, windows)


def _load(path: str) -> None:
    from vizdoom_tracker.features import FeatureResult

    fr = FeatureResult.load(path)
    meta = fr.metadata

    print(f"\n{'─'*60}")
    if meta:
        print(f"Session ID  : {meta.session_id}")
        print(f"Source      : {meta.source_path}")
        print(f"Created     : {meta.created_utc}")
        print(f"Windows (s) : {meta.windows_s}")
        print(f"Samples     : {meta.num_samples}")
        print(f"Features    : {meta.num_features} computed")
        if meta.features_skipped:
            print(f"Skipped     : {', '.join(meta.features_skipped)}")
        print(f"Sample dt   : {meta.sample_interval_s:.4f} s")
    print(f"\nDataFrame shape: {fr.df.shape}")
    print(f"\n{fr.df.describe().to_string()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract behavioral features from VizDoom session recordings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input", nargs="?",
        help="Path to a session .parquet file or a directory of session files",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory for feature files (default: same as input directory)",
    )
    parser.add_argument(
        "--windows", nargs="+", type=int, default=[1, 5, 30],
        help="Rolling window sizes in seconds",
    )
    parser.add_argument(
        "--load", metavar="PATH",
        help="Load and inspect an existing features parquet file",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.load:
        _load(args.load)
        return

    if not args.input:
        parser.error("Provide an input file or directory, or use --load")

    input_path = Path(args.input)

    if input_path.is_file():
        output_dir = Path(args.output) if args.output else input_path.parent
        _process_one(input_path, output_dir, args.windows)
    elif input_path.is_dir():
        output_dir = Path(args.output) if args.output else input_path
        _process_dir(input_path, output_dir, args.windows)
    else:
        parser.error(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()
