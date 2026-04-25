"""
Simple action providers for the VizDoom player slot.

Swap these in `DeathmatchSession` to change agent behaviour without touching
recording or storage logic.
"""

import random
from typing import List, Optional

import vizdoom as vzd


class RandomAgent:
    """Uniformly-random single-button agent.

    Action space: press exactly one of the available buttons, or do nothing
    (no-op).  Actions are one-hot boolean lists matching `available_buttons`.

    This is useful as a baseline / data-collection agent; swap it for a
    trained policy when running evaluation sessions.
    """

    def __init__(self, game: vzd.DoomGame, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)
        n = game.get_available_buttons_size()
        # One button pressed at a time
        self._actions: List[List[bool]] = [[i == j for j in range(n)] for i in range(n)]
        # No-op
        self._actions.append([False] * n)

    def act(self, state: object = None) -> List[bool]:
        """Return a random action. `state` is accepted but ignored."""
        return random.choice(self._actions)
