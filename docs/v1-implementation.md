# V1 Implementation - Module 1: Human-Play Env Wrapper

## Overview

Opaque ViZDoom wrapper for human play with LLM difficulty control. Exposes a simple integer difficulty (1-5) interface, hiding internal complexity from the LLM controller.

## File Location

`src/human_play_env.py`

## API

```python
from src.human_play_env import HumanPlayEnvWrapper

class HumanPlayEnvWrapper:
    def __init__(self, config_path: str = "config/dda_deathmatch.cfg", window_visible: bool = True)
    def reset(self, difficulty: int) -> None      # 1-5 only
    def run_episode(self) -> dict                  # blocking, returns stats
    def get_episode_stats(self) -> dict
    def on_episode_end(self, callback: Callable[[dict], None]) -> None
    def close() -> None
```

## Difficulty Mapping

The LLM only sees integer 1-5. Internal settings are hidden:

| Level | bot_skill | num_bots | time_limit | frag_limit |
|-------|-----------|----------|------------|------------|
| 1     | 1         | 1        | 90s        | 10         |
| 2     | 2         | 2        | 100s       | 15         |
| 3     | 3         | 2        | 120s       | 20         |
| 4     | 4         | 3        | 120s       | 25         |
| 5     | 5         | 4        | 140s       | 30         |

## Stats Format

`get_episode_stats()` returns a simplified dict for LLM consumption:

```python
{
    "kills": int,           # Total kills
    "deaths": int,          # Total deaths
    "kdr": float,           # Kill/death ratio
    "frags": int,           # Net frag score
    "damage_dealt": int,    # Total damage dealt
    "damage_taken": int,    # Total damage taken
    "accuracy": float,      # Hit accuracy
    "duration_seconds": float,
    "health_mean": float,   # Average health
    "difficulty": int       # Current difficulty (1-5)
}
```

## Usage Example

```python
from src.human_play_env import HumanPlayEnvWrapper

env = HumanPlayEnvWrapper()

def handle_episode_end(stats: dict):
    print(f"Episode ended: {stats}")
    # LLM decides next difficulty here

env.on_episode_end(handle_episode_end)

# Start at medium difficulty
env.reset(difficulty=3)
stats = env.run_episode()  # Blocking - player plays until episode ends

# LLM chooses next difficulty based on stats
next_difficulty = llm_decide(stats)  # Returns 1-5
env.reset(difficulty=next_difficulty)
stats = env.run_episode()

env.close()
```

## Dependencies

Reuses existing modules:
- `vizdoom_tracker.recorder.GameVariableRecorder` - frame-level recording
- `vizdoom_tracker.variables.DEATHMATCH_VARS` - game variable set
- `src.collector.stats_from_df` - episode stats computation
- `src.collector.GAME_VARS` - variable list for VizDoom

## Test Script

```bash
python scripts/test_human_play_env.py
```

Verifies:
- Difficulty bounds (1-5 valid, others raise ValueError)
- Callback registration
- Stats dict has expected keys
- Difficulty mapping table correctness
