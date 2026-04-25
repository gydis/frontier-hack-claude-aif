"""
DifficultyActuator — translates DM action indices into VizDoom API calls.

Enforces:
  - Cooldown: no same-direction change two episodes in a row
  - Clamp: skill stays in [1, 5]
  - Bot tracking: remembers how many bots are active
"""

import vizdoom as vzd
from src.dungeon_master import LOWER, MAINTAIN, RAISE, ADD_BOT

ACTION_NAMES = {LOWER: "LOWER", MAINTAIN: "MAINTAIN", RAISE: "RAISE", ADD_BOT: "ADD_BOT"}

MAX_BOTS = 4
MAX_SKILL = 5
MIN_SKILL = 1


class DifficultyActuator:
    def __init__(self, game: vzd.DoomGame, initial_skill: int = 3):
        self._game = game
        self.current_skill = initial_skill
        self.num_bots = 0
        self._last_direction: int | None = None  # +1 raise, -1 lower, 0 maintain

    def apply_action(self, action: int) -> tuple[int, str]:
        """
        Apply action before or after new_episode().
        Returns (new_skill, description_string).

        Skill changes must be applied before new_episode().
        Bot commands must be sent after new_episode().
        """
        direction = self._action_direction(action)
        description = ACTION_NAMES.get(action, "?")

        # Cooldown: block same direction twice in a row for skill changes
        if direction != 0 and direction == self._last_direction and action != ADD_BOT:
            description = f"{description}(blocked-cooldown)"
            action = MAINTAIN
            direction = 0

        if action == LOWER:
            new_skill = max(MIN_SKILL, self.current_skill - 1)
            if new_skill != self.current_skill:
                self.current_skill = new_skill
                self._game.set_doom_skill(new_skill)
                description = f"LOWER → skill {new_skill}"
            else:
                description = "LOWER(clamped)"

        elif action == RAISE:
            new_skill = min(MAX_SKILL, self.current_skill + 1)
            if new_skill != self.current_skill:
                self.current_skill = new_skill
                self._game.set_doom_skill(new_skill)
                description = f"RAISE → skill {new_skill}"
            else:
                description = "RAISE(clamped)"

        elif action == ADD_BOT:
            # Applied after new_episode() via add_bots()
            description = f"ADD_BOT (now {self.num_bots + 1})"

        # else MAINTAIN: no change

        self._last_direction = direction
        return self.current_skill, description

    def add_pending_bots(self, action: int):
        """Call after new_episode() to send addbot commands."""
        if action == ADD_BOT and self.num_bots < MAX_BOTS:
            self._game.send_game_command("addbot")
            self.num_bots += 1

    def check_mid_episode_dominance(self, stats_window: dict) -> bool:
        """
        Return True if the player is dominating mid-episode and we should add a bot now.
        Called during episode (only ADD_BOT is safe mid-episode).
        """
        kdr = stats_window.get("kdr", 0.0)
        deaths = stats_window.get("final_deaths", 0.0)
        kill_rate = stats_window.get("kill_rate", 0.0)
        elapsed = stats_window.get("duration_seconds", 0.0)
        return (
            elapsed >= 30.0
            and kdr > 5.0
            and deaths == 0.0
            and kill_rate > 0.1
            and self.num_bots < MAX_BOTS
        )

    def reset_bots(self):
        self.num_bots = 0

    @staticmethod
    def _action_direction(action: int) -> int:
        if action == LOWER:
            return -1
        elif action in (RAISE, ADD_BOT):
            return 1
        return 0
