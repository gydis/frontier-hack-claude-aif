"""
vizdoom_tracker — VizDoom deathmatch game-state recorder.

Quick-start
───────────
    from vizdoom_tracker import DeathmatchSession

    result = DeathmatchSession(num_bots=3, episode_minutes=5).run()
    result.save("sessions/")          # writes .parquet + .json sidecar
    df = result.df                    # game_time_s-indexed DataFrame

Load a saved session
────────────────────
    from vizdoom_tracker import SessionResult
    result = SessionResult.load("sessions/2025-01-15_abc123.parquet")

Select variable groups
──────────────────────
    from vizdoom_tracker import COMBAT, NAVIGATION
    df[COMBAT.column_names]
    df[NAVIGATION.column_names]

Custom variable set
───────────────────
    from vizdoom_tracker import COMBAT, NAVIGATION, PLAYER_STATE, DeathmatchSession
    import vizdoom as vzd

    my_vars = list(COMBAT) + list(NAVIGATION) + list(PLAYER_STATE)
    result = DeathmatchSession(variables=my_vars).run()
"""

from .session import DeathmatchSession, SessionResult, SessionMetadata
from .recorder import GameVariableRecorder, DOOM_TIC_RATE
from .agent import RandomAgent
from .features import FeatureResult, FeatureMetadata, extract_features
from .variables import (
    VariableGroup,
    COMBAT,
    NAVIGATION,
    PLAYER_STATE,
    WEAPONS_BASIC,
    WEAPONS_FULL,
    MULTIPLAYER,
    ALL_GROUPS,
    DEATHMATCH_VARS,
)

__all__ = [
    # Core session API
    "DeathmatchSession",
    "SessionResult",
    "SessionMetadata",
    # Feature extraction
    "FeatureResult",
    "FeatureMetadata",
    "extract_features",
    # Low-level components
    "GameVariableRecorder",
    "RandomAgent",
    "DOOM_TIC_RATE",
    # Variable groups
    "VariableGroup",
    "COMBAT",
    "NAVIGATION",
    "PLAYER_STATE",
    "WEAPONS_BASIC",
    "WEAPONS_FULL",
    "MULTIPLAYER",
    "ALL_GROUPS",
    "DEATHMATCH_VARS",
]
