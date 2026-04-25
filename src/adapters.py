"""
Adapter implementations wrapping existing components.
"""

from src.dungeon_master import ADD_BOT, LOWER, MAINTAIN, RAISE, DungeonMasterAgent
from src.interfaces import Adapter


class AIFAdapter(Adapter):
    """
    Adapter wrapping the Active Inference DungeonMasterAgent.
    Maps between the Adapter interface and the existing DM implementation.
    """

    def __init__(self, initial_skill: int = 3, seed: int = 42):
        self._dm = DungeonMasterAgent(initial_skill=initial_skill, seed=seed)
        self._current_skill = initial_skill
        self._num_bots = 0
        self._last_action: int | None = None

    def choose_difficulty(self) -> dict:
        return {
            "bot_skill": self._current_skill,
            "num_bots": self._num_bots,
        }

    def update(self, obs_aif: dict) -> None:
        perf_obs = obs_aif.get("performance", 1)
        trend_obs = obs_aif.get("trend", 1)

        action = self._dm.step(perf_obs, trend_obs)
        self._last_action = action

        if action == LOWER:
            self._current_skill = max(1, self._current_skill - 1)
        elif action == RAISE:
            self._current_skill = min(5, self._current_skill + 1)
        elif action == ADD_BOT:
            self._num_bots = min(4, self._num_bots + 1)

    def get_belief(self) -> dict | None:
        return self._dm.get_belief_summary()


class FixedAdapter(Adapter):
    """Fixed difficulty adapter - never changes difficulty."""

    def __init__(self, bot_skill: int = 3, num_bots: int = 2):
        self._bot_skill = bot_skill
        self._num_bots = num_bots

    def choose_difficulty(self) -> dict:
        return {"bot_skill": self._bot_skill, "num_bots": self._num_bots}

    def update(self, obs_aif: dict) -> None:
        pass

    def get_belief(self) -> dict | None:
        return None
