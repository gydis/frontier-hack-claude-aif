"""
EpisodeLogger — JSONL logger for recording episode data.

One record per episode with raw stats, labels, decisions, and LLM context.
Crash-safe append mode ensures no data loss.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


class EpisodeLogger:
    """
    JSONL logger for episode data.

    Each record contains:
    - episode number
    - timestamp
    - raw stats from game
    - discretized labels
    - applied difficulty
    - controller decision
    - LLM prompt/response (if applicable)
    """

    def __init__(self, log_path: str):
        """
        Initialize logger.

        Args:
            log_path: Path to JSONL file (created if doesn't exist)
        """
        self.log_path = log_path
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    def log_episode(
        self,
        episode_number: int,
        raw_stats: dict,
        labels: dict,
        difficulty: int,
        decision: dict,
        llm_prompt: str | None = None,
        llm_response: str | None = None,
    ) -> None:
        """
        Append one episode record to JSONL file.

        Args:
            episode_number: Episode index (0-based)
            raw_stats: Raw stats from HumanPlayEnvWrapper.get_episode_stats()
            labels: Discretized labels from discretize_stats()
            difficulty: Applied difficulty level (1-5)
            decision: Controller decision {"difficulty": int, "reasoning": str}
            llm_prompt: Full LLM prompt (if LLM controller)
            llm_response: Full LLM response (if LLM controller)
        """
        record = {
            "episode": episode_number,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_stats": raw_stats,
            "labels": labels,
            "difficulty": difficulty,
            "decision": decision,
            "reasoning": decision.get("reasoning", ""),
            "llm_prompt": llm_prompt,
            "llm_response": llm_response,
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def load_log(self) -> list[dict]:
        """
        Load all records from log file.

        Returns:
            List of episode records
        """
        if not os.path.exists(self.log_path):
            return []

        records = []
        with open(self.log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def get_latest(self, n: int = 1) -> list[dict]:
        """
        Get the n most recent records.

        Args:
            n: Number of records to return

        Returns:
            List of most recent records (newest last)
        """
        records = self.load_log()
        return records[-n:] if records else []

    def clear(self) -> None:
        """Clear the log file."""
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
