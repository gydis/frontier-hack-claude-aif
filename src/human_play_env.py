"""
HumanPlayEnvWrapper — Opaque ViZDoom wrapper for human play with LLM difficulty control.

Exposes a simple integer difficulty (1-5) interface, hiding internal complexity
(bot_skill, num_bots, time/frag limits) from the LLM controller.
"""

from __future__ import annotations

import os
import time
from typing import Callable

import vizdoom as vzd

from src.collector import GAME_VARS, stats_from_df
from vizdoom_tracker.recorder import GameVariableRecorder
from vizdoom_tracker.variables import DEATHMATCH_VARS


DIFFICULTY_MAP: dict[int, dict] = {
    1: {"bot_skill": 1, "num_bots": 1, "time_limit": 3150, "frag_limit": 10},
    2: {"bot_skill": 2, "num_bots": 2, "time_limit": 3500, "frag_limit": 15},
    3: {"bot_skill": 3, "num_bots": 2, "time_limit": 4200, "frag_limit": 20},
    4: {"bot_skill": 4, "num_bots": 3, "time_limit": 4200, "frag_limit": 25},
    5: {"bot_skill": 5, "num_bots": 4, "time_limit": 4900, "frag_limit": 30},
}


class HumanPlayEnvWrapper:
    """
    Opaque ViZDoom wrapper for human play with LLM difficulty control.

    Exposes only:
    - reset(difficulty: int)  # 1-5
    - run_episode() -> dict   # blocking, returns stats
    - get_episode_stats() -> dict
    - on_episode_end(callback)
    - close()
    """

    def __init__(
        self,
        config_path: str = "config/dda-deathmatch.cfg",
        window_visible: bool = True,
    ):
        self._config_path = config_path
        self._window_visible = window_visible
        self._game: vzd.DoomGame | None = None
        self._recorder: GameVariableRecorder | None = None
        self._callbacks: list[Callable[[dict], None]] = []
        self._current_difficulty: int = 3
        self._current_settings: dict = DIFFICULTY_MAP[3]

    def reset(self, difficulty: int) -> None:
        """
        Reset the environment with a new difficulty level.

        Args:
            difficulty: Integer 1-5 (easy to hard)

        Raises:
            ValueError: If difficulty not in range 1-5
        """
        if difficulty not in range(1, 6):
            raise ValueError(f"Difficulty must be 1-5, got {difficulty}")

        self._current_difficulty = difficulty
        self._current_settings = DIFFICULTY_MAP[difficulty]

        if self._game is not None:
            self._game.close()

        self._game = vzd.DoomGame()
        self._game.load_config(self._config_path)

        wad_name = os.path.basename(self._game.get_doom_scenario_path())
        self._game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, wad_name))

        self._game.set_window_visible(self._window_visible)
        self._game.set_mode(vzd.Mode.SPECTATOR)
        self._game.set_available_game_variables(GAME_VARS)
        self._game.set_doom_skill(self._current_settings["bot_skill"])
        self._game.set_episode_timeout(self._current_settings["time_limit"])
        self._game.add_game_args("+sv_cheats 1")
        self._game.add_game_args(f"+fraglimit {self._current_settings['frag_limit']}")
        self._game.init()

        self._game.new_episode()

        for _ in range(self._current_settings["num_bots"]):
            self._game.send_game_command("addbot")

        self._recorder = GameVariableRecorder(DEATHMATCH_VARS)
        self._recorder.reset()

    def on_episode_end(self, callback: Callable[[dict], None]) -> None:
        """Register a callback to fire when episode ends."""
        self._callbacks.append(callback)

    def run_episode(self) -> dict:
        """
        Run episode until completion (blocking).
        Fires callbacks, returns episode stats.
        """
        if self._game is None:
            raise RuntimeError("Call reset() before run_episode()")

        tic_duration = 1.0 / 35.0
        _queued_respawn = False

        while not self._game.is_episode_finished():
            is_dead = self._game.is_player_dead()
            if not is_dead:
                try:
                    is_dead = bool(self._game.get_game_variable(vzd.GameVariable.DEAD))
                except Exception:
                    pass

            if is_dead:
                if not _queued_respawn:
                    self._game.respawn_player()
                    _queued_respawn = True
                self._game.advance_action(1)
            else:
                _queued_respawn = False
                t0 = time.time()
                self._game.advance_action(1)
                self._recorder.record(self._game)
                elapsed = time.time() - t0
                sleep_for = tic_duration - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

        stats = self.get_episode_stats()
        self._fire_callbacks(stats)
        return stats

    def get_episode_stats(self) -> dict:
        """
        Return simplified episode stats for LLM consumption.
        Hides internal complexity.
        """
        if self._recorder is None:
            return {}

        df = self._recorder.to_dataframe()
        if len(df) == 0:
            return {
                "kills": 0,
                "deaths": 0,
                "kdr": 0.0,
                "frags": 0,
                "damage_dealt": 0,
                "damage_taken": 0,
                "accuracy": 0.0,
                "duration_seconds": 0.0,
                "health_mean": 100.0,
                "difficulty": self._current_difficulty,
            }

        raw_stats = stats_from_df(
            df,
            self._recorder.episode_tic,
            self._current_settings["bot_skill"],
            "deathmatch",
        )

        def _last(col: str) -> float:
            return float(df[col].iloc[-1]) if col in df.columns and len(df) > 0 else 0.0

        return {
            "kills": int(raw_stats.get("final_kills", 0)),
            "deaths": int(raw_stats.get("final_deaths", 0)),
            "kdr": round(raw_stats.get("kdr", 0), 2),
            "frags": int(raw_stats.get("final_frags", 0)),
            "damage_dealt": int(_last("damagecount")),
            "damage_taken": int(_last("damage_taken")),
            "accuracy": round(raw_stats.get("hit_accuracy", 0), 2),
            "duration_seconds": round(raw_stats.get("duration_seconds", 0), 1),
            "health_mean": round(raw_stats.get("health_mean", 0), 1),
            "difficulty": self._current_difficulty,
        }

    def _fire_callbacks(self, stats: dict) -> None:
        for cb in self._callbacks:
            try:
                cb(stats)
            except Exception as e:
                print(f"Episode callback error: {e}")

    def close(self) -> None:
        """Clean up VizDoom resources."""
        if self._game is not None:
            self._game.close()
            self._game = None
