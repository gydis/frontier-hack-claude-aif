# V2 Implementation - Module 2: Discretizer (Hardcoded)

## Overview

Converts raw episode stats to qualitative labels ("poor", "normal", "high") for LLM consumption. Uses hardcoded thresholds calibrated for human play.

## File Location

`src/discretizer.py`

## API

```python
from src.discretizer import discretize_stats

def discretize_stats(stats: dict) -> dict:
    """
    Convert raw stats to qualitative labels for LLM.
    
    Input: stats from HumanPlayEnvWrapper.get_episode_stats()
    Output: {"accuracy": "normal", "kdr": "high", "frags_per_minute": "poor", "damage_ratio": "normal"}
    """
```

## Thresholds

| Metric | Poor | Normal | High |
|--------|------|--------|------|
| accuracy | < 0.3 | 0.3 - 0.6 | > 0.6 |
| kdr | < 0.5 | 0.5 - 1.5 | > 1.5 |
| frags_per_minute | < 1.0 | 1.0 - 3.0 | > 3.0 |
| damage_ratio | < 0.5 | 0.5 - 1.5 | > 1.5 |

## Derived Metrics

- `frags_per_minute` = frags / (duration_seconds / 60)
- `damage_ratio` = damage_dealt / max(damage_taken, 1)

## Usage Example

```python
from src.human_play_env import HumanPlayEnvWrapper
from src.discretizer import discretize_stats

env = HumanPlayEnvWrapper()
env.reset(difficulty=3)
stats = env.run_episode()

labels = discretize_stats(stats)
# {"accuracy": "normal", "kdr": "high", "frags_per_minute": "normal", "damage_ratio": "high"}

# Pass labels to LLM controller
llm_input = f"Player performance: {labels}"
```

## Test Cases

```python
# Poor performance
stats = {"accuracy": 0.1, "kdr": 0.2, "frags": 0, "duration_seconds": 60, 
         "damage_dealt": 10, "damage_taken": 100}
# → {"accuracy": "poor", "kdr": "poor", "frags_per_minute": "poor", "damage_ratio": "poor"}

# Normal performance  
stats = {"accuracy": 0.4, "kdr": 1.0, "frags": 2, "duration_seconds": 60,
         "damage_dealt": 100, "damage_taken": 100}
# → {"accuracy": "normal", "kdr": "normal", "frags_per_minute": "normal", "damage_ratio": "normal"}

# High performance
stats = {"accuracy": 0.8, "kdr": 3.0, "frags": 10, "duration_seconds": 60,
         "damage_dealt": 500, "damage_taken": 50}
# → {"accuracy": "high", "kdr": "high", "frags_per_minute": "high", "damage_ratio": "high"}
```

## Dependencies

- Input comes from `HumanPlayEnvWrapper.get_episode_stats()` (Module 1)
- No external dependencies (no baselines or calibration data needed)
