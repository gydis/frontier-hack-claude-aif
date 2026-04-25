"""
YAML experiment config loader.
"""

from dataclasses import dataclass

import yaml


@dataclass
class ExperimentConfig:
    experiment_name: str
    proxy_id: str
    adapter_type: str
    num_episodes: int
    scenario: str
    difficulty_bounds: dict
    seed: int


REQUIRED_FIELDS = [
    "experiment_name",
    "proxy_id",
    "adapter_type",
    "num_episodes",
    "scenario",
]


def load_experiment_config(path: str) -> ExperimentConfig:
    """Load and validate experiment YAML config."""
    with open(path) as f:
        data = yaml.safe_load(f)

    missing = [f for f in REQUIRED_FIELDS if f not in data]
    if missing:
        raise ValueError(f"Missing required fields in config: {missing}")

    data.setdefault(
        "difficulty_bounds",
        {
            "bot_skill_min": 1,
            "bot_skill_max": 5,
            "num_bots_min": 1,
            "num_bots_max": 4,
        },
    )
    data.setdefault("seed", 42)

    return ExperimentConfig(**data)
