"""
Stub implementations for testing the pipeline without real components.
"""

import random

from src.interfaces import Adapter, PlayerProxy


class StubAdapter(Adapter):
    """Stub adapter that returns fixed difficulty."""

    def __init__(self, bot_skill: int = 3, num_bots: int = 2):
        self._bot_skill = bot_skill
        self._num_bots = num_bots
        self._update_count = 0

    def choose_difficulty(self) -> dict:
        return {"bot_skill": self._bot_skill, "num_bots": self._num_bots}

    def update(self, obs_aif: dict) -> None:
        self._update_count += 1
        print(f"  [StubAdapter] update #{self._update_count}: {obs_aif}")

    def get_belief(self) -> dict | None:
        return None


class StubPlayerProxy(PlayerProxy):
    """Stub proxy that takes random actions."""

    def __init__(self, proxy_id: str = "stub_random", num_buttons: int = 20):
        self._id = proxy_id
        self._num_buttons = num_buttons

    @property
    def id(self) -> str:
        return self._id

    def act(self, obs) -> list:
        return [random.randint(0, 1) for _ in range(self._num_buttons)]
