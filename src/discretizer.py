"""
Discretizer — converts raw episode stats to qualitative labels for LLM consumption.

Hand-coded thresholds calibrated for human play. Output is a flat dict of string labels
("poor", "normal", "high") that the LLM can reason about.
"""

from __future__ import annotations

THRESHOLDS: dict[str, tuple[float, float]] = {
    "accuracy": (0.3, 0.6),
    "kdr": (0.5, 1.5),
    "frags_per_minute": (1.0, 3.0),
    "damage_ratio": (0.5, 1.5),
}


def _classify(value: float, low: float, high: float) -> str:
    """Classify a value into poor/normal/high based on thresholds."""
    if value < low:
        return "poor"
    elif value > high:
        return "high"
    else:
        return "normal"


def discretize_stats(stats: dict) -> dict:
    """
    Convert raw episode stats to qualitative labels for LLM.

    Input: stats dict from HumanPlayEnvWrapper.get_episode_stats()
        Expected keys: accuracy, kdr, frags, duration_seconds, damage_dealt, damage_taken

    Output: flat dict of string labels
        {"accuracy": "normal", "kdr": "high", "frags_per_minute": "poor", "damage_ratio": "normal"}
    """
    accuracy = stats.get("accuracy", 0.0)
    kdr = stats.get("kdr", 0.0)
    frags = stats.get("frags", 0)
    duration_seconds = stats.get("duration_seconds", 1.0)
    damage_dealt = stats.get("damage_dealt", 0)
    damage_taken = stats.get("damage_taken", 1)

    duration_minutes = max(duration_seconds / 60.0, 0.01)
    frags_per_minute = frags / duration_minutes
    damage_ratio = damage_dealt / max(damage_taken, 1)

    return {
        "accuracy": _classify(accuracy, *THRESHOLDS["accuracy"]),
        "kdr": _classify(kdr, *THRESHOLDS["kdr"]),
        "frags_per_minute": _classify(frags_per_minute, *THRESHOLDS["frags_per_minute"]),
        "damage_ratio": _classify(damage_ratio, *THRESHOLDS["damage_ratio"]),
    }
