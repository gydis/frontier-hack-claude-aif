"""
Core interfaces for the DDA system.
ML A and ML B should subclass these to implement their components.
"""

from abc import ABC, abstractmethod
from typing import Any


class Adapter(ABC):
    """
    Difficulty adapter interface.

    Implementations:
    - AIF adapter (ML A): uses Active Inference to infer player state
    - Fixed adapter (ML B): returns constant difficulty
    - RuleBased adapter (ML B): uses heuristic rules

    Lifecycle per episode:
    1. choose_difficulty() called before episode starts
    2. Episode runs
    3. update(obs_aif) called with discretized observations after episode ends
    """

    @abstractmethod
    def choose_difficulty(self) -> dict:
        """
        Returns difficulty dict for the next episode.
        Keys depend on scenario, typical: {"bot_skill": int, "num_bots": int}
        """
        ...

    @abstractmethod
    def update(self, obs_aif: dict) -> None:
        """
        Update internal state with discretized AIF observations from completed episode.
        obs_aif keys: {"performance": int, "trend": int}
        Values: 0=LOW/DECLINING, 1=MED/STABLE, 2=HIGH/IMPROVING
        """
        ...

    @abstractmethod
    def get_belief(self) -> dict | None:
        """
        Returns current belief state for logging/visualization.
        AIF adapter returns engagement distribution; baselines may return None.
        """
        ...


class PlayerProxy(ABC):
    """
    Player agent interface.

    Implementations:
    - Built-in bot (ML B): uses ViZDoom's native bots
    - PPO checkpoint (ML B): trained RL agent
    - Human: SPECTATOR mode (existing DDAPipeline handles this)
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """Unique identifier for this proxy (e.g., 'builtin_bot', 'ppo_v1')"""
        ...

    @abstractmethod
    def act(self, obs: Any) -> Any:
        """
        Given observation from env, return action.
        Obs/action format depends on the env wrapper implementation.
        """
        ...


class EnvWrapper(ABC):
    """
    Environment wrapper interface for ViZDoom scenarios.
    Handles difficulty application and episode statistics collection.
    """

    @abstractmethod
    def reset(self, difficulty: dict) -> None:
        """
        Start a new episode with the given difficulty settings.
        difficulty keys: {"bot_skill": int, "num_bots": int, ...}
        """
        ...

    @abstractmethod
    def get_state(self) -> Any:
        """
        Returns current observation for the player proxy.
        Type depends on implementation (e.g., screen buffer, game variables).
        """
        ...

    @abstractmethod
    def step(self, action: Any) -> tuple[float, bool]:
        """
        Execute action, return (reward, done).
        reward: float (can be 0 for non-RL usage)
        done: True if episode finished
        """
        ...

    @abstractmethod
    def get_episode_stats(self) -> dict:
        """
        Returns episode statistics after episode ends.
        Required keys: frags, deaths, accuracy, damage_taken, damage_dealt, duration_seconds
        """
        ...


class RunLogger(ABC):
    """
    Crash-safe JSONL logger for experiment runs.
    Writes one record per episode to runs/<HHMMSS>_<experiment_name>.jsonl
    """

    @abstractmethod
    def write(self, record: dict) -> None:
        """
        Append record to log file and flush immediately.
        Required fields: episode_idx, difficulty, raw_stats, obs_aif, belief,
                        proxy_id, adapter_type, experiment_name, timestamp
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the log file."""
        ...
