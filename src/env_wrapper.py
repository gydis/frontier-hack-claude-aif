"""
ViZDoom environment wrapper for deathmatch scenario.
"""

import os

import vizdoom as vzd

from src.collector import GAME_VARS, EpisodeCollector
from src.interfaces import EnvWrapper as EnvWrapperABC


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
        self._collector: EpisodeCollector | None = None
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

        self._collector = EpisodeCollector(
            self._game, "deathmatch", self._current_skill
        )
        self._collector.reset()

    def get_state(self):
        return self._game.get_state()

    def step(self, action) -> tuple[float, bool]:
        reward = self._game.make_action(action)
        self._collector.step()
        done = self._game.is_episode_finished()
        return reward, done

    def get_episode_stats(self) -> dict:
        stats = self._collector.get_episode_stats()
        return {
            "frags": stats.get("final_frags", 0),
            "deaths": stats.get("final_deaths", 0),
            "accuracy": stats.get("hit_accuracy", 0),
            "damage_taken": stats.get("damage_taken", 0),
            "damage_dealt": stats.get("damagecount", 0),
            "duration_seconds": stats.get("duration_seconds", 0),
            **stats,
        }

    def close(self):
        if self._game is not None:
            self._game.close()
            self._game = None
