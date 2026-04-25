"""
ViZDoom environment wrapper for deathmatch scenario.
"""

import os
from datetime import datetime, timezone

import vizdoom as vzd

from src.collector import GAME_VARS, stats_from_df
from src.interfaces import EnvWrapper as EnvWrapperABC
from vizdoom_tracker.recorder import GameVariableRecorder
from vizdoom_tracker.session import SessionMetadata, SessionResult
from vizdoom_tracker.variables import DEATHMATCH_VARS


class DeathmatchEnvWrapper(EnvWrapperABC):
    """ViZDoom deathmatch environment wrapper."""

    def __init__(
        self,
        config_path: str = "config/dda_deathmatch.cfg",
        window_visible: bool = False,
    ):
        self._config_path = config_path
        self._window_visible = window_visible
        self._game: vzd.DoomGame | None = None
        self._recorder: GameVariableRecorder | None = None
        self._num_bots: int = 0
        self._current_skill: int = 3

    def reset(self, difficulty: dict) -> None:
        if self._game is not None:
            self._game.close()

        self._current_skill = difficulty.get("bot_skill", 3)
        self._num_bots = difficulty.get("num_bots", 2)

        self._game = vzd.DoomGame()
        self._game.load_config(self._config_path)

        wad_name = os.path.basename(self._game.get_doom_scenario_path())
        self._game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, wad_name))

        self._game.set_window_visible(self._window_visible)
        self._game.set_mode(vzd.Mode.PLAYER)
        self._game.set_available_game_variables(GAME_VARS)
        self._game.set_doom_skill(self._current_skill)
        self._game.add_game_args("+sv_cheats 1")
        self._game.init()

        self._game.new_episode()

        for _ in range(self._num_bots):
            self._game.send_game_command("addbot")

        self._recorder = GameVariableRecorder(DEATHMATCH_VARS)
        self._recorder.reset()

    def get_state(self):
        return self._game.get_state()

    def step(self, action) -> tuple[float, bool]:
        reward = self._game.make_action(action)
        self._recorder.record(self._game)
        done = self._game.is_episode_finished()
        return reward, done

    def get_episode_stats(self) -> dict:
        df = self._recorder.to_dataframe()
        stats = stats_from_df(df, self._recorder.episode_tic, self._current_skill, "deathmatch")
        return {
            "frags": stats.get("final_frags", 0),
            "deaths": stats.get("final_deaths", 0),
            "accuracy": stats.get("hit_accuracy", 0),
            "damage_taken": stats.get("damage_taken", 0),
            "damage_dealt": stats.get("damagecount", 0),
            "duration_seconds": stats.get("duration_seconds", 0),
            **stats,
        }

    def get_session_result(self, session_id: str, **extra) -> SessionResult:
        """Return a SessionResult for Parquet saving."""
        df = self._recorder.to_dataframe()
        meta = SessionMetadata(
            session_id=session_id,
            start_utc=datetime.now(timezone.utc).isoformat(),
            scenario="deathmatch",
            num_bots=self._num_bots,
            episode_timeout_tics=self._recorder.episode_tic,
            sample_interval_tics=self._recorder._sample_every,
            tic_rate=35,
            variables=[v.name for v in DEATHMATCH_VARS],
            total_tics=self._recorder.episode_tic,
            total_samples=self._recorder.num_samples,
            extra=extra,
        )
        return SessionResult(df=df, metadata=meta)

    def close(self):
        if self._game is not None:
            self._game.close()
            self._game = None
