"""
BotBaselines — loads bot rollout .npz/.json files and computes per-skill
mean/std baselines used by PlayerStateEstimator.
"""

import json
import os
from pathlib import Path

import numpy as np

STAT_KEYS = [
    "kdr", "kill_rate", "death_rate", "damage_efficiency",
    "health_mean", "health_min", "health_variance",
    "survival_frac", "movement_entropy", "hit_accuracy",
]

# Fallback baselines used when no data has been collected yet.
# Values are rough estimates for deathmatch at each skill 1-5.
_FALLBACK_BASELINES = {
    skill: {
        "kdr":               {"mean": [3.0, 2.0, 1.5, 1.0, 0.6][skill - 1], "std": 0.8},
        "kill_rate":         {"mean": [0.10, 0.08, 0.06, 0.05, 0.03][skill - 1], "std": 0.03},
        "death_rate":        {"mean": [0.02, 0.04, 0.06, 0.08, 0.12][skill - 1], "std": 0.03},
        "damage_efficiency": {"mean": [3.0, 2.5, 2.0, 1.5, 1.0][skill - 1], "std": 0.8},
        "health_mean":       {"mean": [80.0, 72.0, 65.0, 58.0, 45.0][skill - 1], "std": 12.0},
        "health_min":        {"mean": [50.0, 40.0, 30.0, 20.0, 10.0][skill - 1], "std": 15.0},
        "health_variance":   {"mean": [150.0, 200.0, 250.0, 300.0, 400.0][skill - 1], "std": 80.0},
        "survival_frac":     {"mean": [0.9, 0.8, 0.7, 0.6, 0.4][skill - 1], "std": 0.15},
        "movement_entropy":  {"mean": 2.5, "std": 0.5},
        "hit_accuracy":      {"mean": [0.5, 0.4, 0.3, 0.2, 0.1][skill - 1], "std": 0.1},
    }
    for skill in range(1, 6)
}


class BotBaselines:
    """
    Stores mean/std of episode statistics for each skill level.
    Can be computed from collected rollout data or fall back to heuristic priors.
    """

    def __init__(self):
        # baselines[skill] = {stat_key: {"mean": float, "std": float}}
        self.baselines: dict[int, dict] = {}
        self.scenario: str = "unknown"
        self._use_fallback = True

    # ------------------------------------------------------------------
    # Building baselines from rollout JSON files
    # ------------------------------------------------------------------

    def compute_from_dir(self, data_dir: str, scenario: str = "") -> None:
        """
        Scan data_dir for skill_N subdirectories containing episode .json files,
        compute mean/std per stat per skill.
        """
        self.scenario = scenario
        data_path = Path(data_dir)
        found_skills = []

        for skill in range(1, 6):
            skill_dir = data_path / f"skill_{skill}"
            if not skill_dir.exists():
                continue
            json_files = list(skill_dir.glob("*.json"))
            if not json_files:
                continue

            skill_stats: dict[str, list] = {k: [] for k in STAT_KEYS}
            for jf in json_files:
                with open(jf) as f:
                    ep = json.load(f)
                for k in STAT_KEYS:
                    if k in ep:
                        skill_stats[k].append(ep[k])

            if all(len(v) == 0 for v in skill_stats.values()):
                continue

            self.baselines[skill] = {}
            for k in STAT_KEYS:
                vals = np.array(skill_stats[k], dtype=np.float32)
                if len(vals) == 0:
                    vals = np.array([0.0])
                self.baselines[skill][k] = {
                    "mean": float(vals.mean()),
                    "std": float(max(vals.std(), 1e-3)),
                }
            found_skills.append(skill)

        if found_skills:
            self._use_fallback = False
            # Fill missing skills with interpolated neighbours
            self._fill_missing_skills()
            print(f"BotBaselines: loaded data for skills {found_skills}")
        else:
            print("BotBaselines: no rollout data found, using fallback priors")
            self._load_fallback()

    def _fill_missing_skills(self):
        for skill in range(1, 6):
            if skill not in self.baselines:
                neighbours = [s for s in self.baselines if s != skill]
                if neighbours:
                    closest = min(neighbours, key=lambda s: abs(s - skill))
                    self.baselines[skill] = self.baselines[closest]

    def _load_fallback(self):
        self.baselines = {k: dict(v) for k, v in _FALLBACK_BASELINES.items()}
        self._use_fallback = True

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        payload = {"scenario": self.scenario, "baselines": {str(k): v for k, v in self.baselines.items()}}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            payload = json.load(f)
        self.scenario = payload.get("scenario", "")
        self.baselines = {int(k): v for k, v in payload["baselines"].items()}
        self._use_fallback = False
        print(f"BotBaselines: loaded from {path} (scenario={self.scenario})")

    @classmethod
    def load_or_fallback(cls, path: str) -> "BotBaselines":
        obj = cls()
        if os.path.exists(path):
            obj.load(path)
        else:
            print(f"BotBaselines: {path} not found, using fallback priors")
            obj._load_fallback()
        return obj

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, skill: int, stat: str) -> tuple[float, float]:
        """Return (mean, std) for the given skill and stat key."""
        if not self.baselines:
            self._load_fallback()
        skill = max(1, min(5, skill))
        entry = self.baselines.get(skill, self.baselines[min(self.baselines)])
        val = entry.get(stat, {"mean": 0.0, "std": 1.0})
        return float(val["mean"]), float(val["std"])

    def zscore(self, skill: int, stat: str, value: float) -> float:
        """Return z-score of value against the baseline at the given skill."""
        mean, std = self.get(skill, stat)
        return (value - mean) / max(std, 1e-6)

    def is_fallback(self) -> bool:
        return self._use_fallback
