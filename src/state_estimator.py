"""
PlayerStateEstimator — maps episode statistics to discrete Active Inference observations.

Observation modalities:
  0 — Performance:  LOW=0, MED=1, HIGH=2
  1 — Trend:        DECLINING=0, STABLE=1, IMPROVING=2
"""

import numpy as np
from src.baselines import BotBaselines

# Observation indices
PERF_LOW, PERF_MED, PERF_HIGH = 0, 1, 2
TREND_DECLINING, TREND_STABLE, TREND_IMPROVING = 0, 1, 2

# Thresholds for performance z-score → category
PERF_LOW_THRESH = -0.5   # z < -0.5 → LOW
PERF_HIGH_THRESH = 0.5   # z > +0.5 → HIGH

# Thresholds for trend detection
TREND_DECLINE_THRESH = -0.15
TREND_IMPROVE_THRESH = 0.15

# Weights for the composite performance score
_WEIGHTS = {
    "kdr": 0.35,
    "kill_rate": 0.25,
    "health_mean": 0.20,
    "damage_efficiency": 0.10,
    "survival_frac": 0.10,
}


class PlayerStateEstimator:
    """
    Converts raw episode statistics into discretised observation indices
    for the Active Inference agent.
    """

    def __init__(self, baselines: BotBaselines):
        self.baselines = baselines
        self._prev_score: float | None = None

    def estimate(self, episode_stats: dict, current_skill: int) -> tuple[int, int]:
        """
        Returns (performance_obs, trend_obs).

        performance_obs: LOW=0, MED=1, HIGH=2
        trend_obs:       DECLINING=0, STABLE=1, IMPROVING=2
        """
        score = self._compute_performance_score(episode_stats, current_skill)
        perf_obs = self._bin_performance(score, current_skill)
        trend_obs = self._compute_trend(score)

        self._prev_score = score
        return perf_obs, trend_obs

    def _compute_performance_score(self, stats: dict, skill: int) -> float:
        """
        Weighted average of z-scores across key metrics.
        Returns a value roughly in [-2, +2]; positive = above baseline.
        """
        total_weight = 0.0
        weighted_z = 0.0
        for stat, weight in _WEIGHTS.items():
            val = stats.get(stat)
            if val is None:
                continue
            z = self.baselines.zscore(skill, stat, val)
            z = float(np.clip(z, -3.0, 3.0))
            weighted_z += weight * z
            total_weight += weight
        if total_weight < 1e-6:
            return 0.0
        return weighted_z / total_weight

    def _bin_performance(self, score: float, skill: int) -> int:
        """Discretise performance z-score into LOW/MED/HIGH."""
        if score < PERF_LOW_THRESH:
            return PERF_LOW
        elif score > PERF_HIGH_THRESH:
            return PERF_HIGH
        else:
            return PERF_MED

    def _compute_trend(self, current_score: float) -> int:
        """Compare to previous episode score."""
        if self._prev_score is None:
            return TREND_STABLE
        delta = current_score - self._prev_score
        if delta < TREND_DECLINE_THRESH:
            return TREND_DECLINING
        elif delta > TREND_IMPROVE_THRESH:
            return TREND_IMPROVING
        else:
            return TREND_STABLE

    def get_last_score(self) -> float | None:
        return self._prev_score

    def reset(self):
        """Call at the start of a new session."""
        self._prev_score = None
