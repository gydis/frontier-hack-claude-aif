"""
Tic-level game variable sampler.

`GameVariableRecorder.record(game)` is called once per tic; it snapshots all
tracked variables every `sample_every_n_tics` tics (default 35 = 1 s).

`to_dataframe()` returns a pandas DataFrame indexed by `game_time_s` (seconds
since episode start, in exact 1/tic_rate steps) — ready for time-series
analysis or feature engineering.
"""

import time
from typing import Dict, List, Optional, Sequence

import pandas as pd
import vizdoom as vzd

from .variables import DEATHMATCH_VARS

DOOM_TIC_RATE = 35  # engine constant: tics per second


class GameVariableRecorder:
    """Accumulates game variable snapshots at regular tic intervals."""

    def __init__(
        self,
        variables: Optional[Sequence[vzd.GameVariable]] = None,
        sample_every_n_tics: int = DOOM_TIC_RATE,
    ) -> None:
        self._variables: List[vzd.GameVariable] = list(variables or DEATHMATCH_VARS)
        self._sample_every = sample_every_n_tics
        self._records: List[Dict] = []
        self._tic: int = 0
        self._wall_t0: float = 0.0

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def episode_tic(self) -> int:
        """Number of tics advanced since last reset()."""
        return self._tic

    @property
    def num_samples(self) -> int:
        return len(self._records)

    def reset(self) -> None:
        """Clear accumulated records and restart the tic counter."""
        self._records.clear()
        self._tic = 0
        self._wall_t0 = time.monotonic()

    def record(self, game: vzd.DoomGame) -> bool:
        """Advance tic counter by 1; sample if the interval is reached.

        Call this once per `game.make_action()`.
        Returns True when a snapshot was taken.
        """
        self._tic += 1
        if self._tic % self._sample_every != 0:
            return False
        self._records.append(self._snapshot(game))
        return True

    def to_dataframe(self) -> pd.DataFrame:
        """Return all recorded snapshots as a time-indexed DataFrame.

        Index: `game_time_s` (float seconds, exact multiples of 1/tic_rate).
        Columns: episode_tic, game_tic, wall_time_s, <variable columns...>
        """
        if not self._records:
            return pd.DataFrame()
        df = pd.DataFrame(self._records)
        df = df.set_index("game_time_s").sort_index()
        return df

    # ── Internal ──────────────────────────────────────────────────────────────

    def _snapshot(self, game: vzd.DoomGame) -> Dict:
        wall_elapsed = time.monotonic() - self._wall_t0
        row: Dict = {
            "game_time_s": round(self._tic / DOOM_TIC_RATE, 6),
            "episode_tic": self._tic,
            "game_tic": game.get_episode_time(),
            "wall_time_s": round(wall_elapsed, 4),
        }
        for var in self._variables:
            row[var.name.lower()] = game.get_game_variable(var)
        return row
