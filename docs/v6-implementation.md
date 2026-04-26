# V6 Implementation - Module 5: Episode Logger

## Overview

JSONL logger for recording episode data. One record per episode with raw stats, labels, decisions, and LLM context. Crash-safe append mode ensures no data loss.

## File Location

`src/episode_logger.py`

## API

```python
from src.episode_logger import EpisodeLogger

class EpisodeLogger:
    def __init__(self, log_path: str)
    
    def log_episode(
        self,
        episode_number: int,
        raw_stats: dict,
        labels: dict,
        difficulty: int,
        decision: dict,
        llm_prompt: str | None = None,
        llm_response: str | None = None,
    ) -> None
    
    def load_log(self) -> list[dict]
    def get_latest(self, n: int = 1) -> list[dict]
    def clear(self) -> None
```

## Record Format

Each JSONL record contains:

```json
{
    "episode": 0,
    "timestamp": "2026-04-26T12:00:00+00:00",
    "raw_stats": {"kills": 5, "deaths": 2, "kdr": 2.5, ...},
    "labels": {"accuracy": "normal", "kdr": "high", ...},
    "difficulty": 3,
    "decision": {"difficulty": 4, "reasoning": "Player doing well"},
    "reasoning": "Player doing well",
    "llm_prompt": "You are a difficulty controller...",
    "llm_response": "{\"difficulty\": 4, \"reasoning\": \"...\"}"
}
```

## Features

- **Crash-safe**: Uses `flush()` + `fsync()` after each write
- **Append mode**: Never overwrites existing data
- **Auto-creates directories**: Parent directories created if needed
- **Dashboard-ready**: `load_log()` returns list for Streamlit consumption

## Usage Example

```python
from src.human_play_env import HumanPlayEnvWrapper
from src.discretizer import discretize_stats
from src.baseline_controllers import create_controller
from src.episode_logger import EpisodeLogger

env = HumanPlayEnvWrapper()
controller = create_controller("llm")
logger = EpisodeLogger("runs/session_001.jsonl")

difficulty = 3
history = []

for episode in range(10):
    env.reset(difficulty)
    stats = env.run_episode()
    labels = discretize_stats(stats)
    
    decision = controller.get_difficulty_decision(labels, difficulty, history)
    
    logger.log_episode(
        episode_number=episode,
        raw_stats=stats,
        labels=labels,
        difficulty=difficulty,
        decision=decision,
    )
    
    history.append(decision)
    difficulty = decision["difficulty"]

# For dashboard
records = logger.load_log()
latest = logger.get_latest(5)  # Last 5 episodes
```

## Integration with Dashboard

The logger output is designed for Streamlit dashboard consumption:

```python
import streamlit as st
from src.episode_logger import EpisodeLogger

logger = EpisodeLogger("runs/active_session.jsonl")
records = logger.load_log()

# Plot difficulty over time
difficulties = [r["difficulty"] for r in records]
st.line_chart(difficulties)

# Show latest reasoning
if records:
    st.write(records[-1]["reasoning"])
```
