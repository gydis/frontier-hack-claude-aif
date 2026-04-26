# V1 System Test Script

## Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up Anthropic API key (for LLM controller):
```bash
export ANTHROPIC_API_KEY="your-api-key"
# Or create .env file with: ANTHROPIC_API_KEY=your-api-key
```

3. Ensure VizDoom is properly installed with WAD files available.

## Module Overview

| Module | File | Description |
|--------|------|-------------|
| Env Wrapper | `src/human_play_env.py` | Human-play VizDoom interface |
| Discretizer | `src/discretizer.py` | Stats → labels conversion |
| LLM Controller | `src/llm_controller.py` | Claude-based difficulty decisions |
| Baselines | `src/baseline_controllers.py` | Fixed/rule-based controllers |
| Logger | `src/episode_logger.py` | JSONL episode logging |
| Dashboard | `pivot/dashboard.py` | Streamlit live view (existing) |

## Running the Full System

### Step 1: Create Main Runner Script

Create `scripts/run_llm_session.py`:

```python
#!/usr/bin/env python3
"""Run a human-play session with LLM difficulty control."""

import sys
sys.path.insert(0, ".")

from datetime import datetime
from src.human_play_env import HumanPlayEnvWrapper
from src.discretizer import discretize_stats
from src.baseline_controllers import create_controller
from src.episode_logger import EpisodeLogger


def main():
    # Configuration
    controller_type = "llm"  # "llm", "fixed", or "rule_based"
    num_episodes = 10
    initial_difficulty = 3
    
    # Initialize components
    env = HumanPlayEnvWrapper(window_visible=True)
    controller = create_controller(controller_type)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = EpisodeLogger(f"runs/{timestamp}_{controller_type}.jsonl")
    
    difficulty = initial_difficulty
    history = []
    
    print(f"\n=== Starting {controller_type.upper()} Session ===")
    print(f"Episodes: {num_episodes}")
    print(f"Initial difficulty: {difficulty}")
    print("Use WASD to move, mouse to aim, click to shoot\n")
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} (Difficulty: {difficulty}) ---")
        
        # Play episode
        env.reset(difficulty)
        stats = env.run_episode()
        
        # Process results
        labels = discretize_stats(stats)
        decision = controller.get_difficulty_decision(labels, difficulty, history)
        
        # Log episode
        logger.log_episode(
            episode_number=episode,
            raw_stats=stats,
            labels=labels,
            difficulty=difficulty,
            decision=decision,
        )
        
        # Print summary
        print(f"Stats: kills={stats['kills']}, deaths={stats['deaths']}, kdr={stats['kdr']}")
        print(f"Labels: {labels}")
        print(f"Decision: {decision['reasoning']}")
        print(f"Next difficulty: {decision['difficulty']}")
        
        # Update for next episode
        history.append(decision)
        difficulty = decision["difficulty"]
    
    env.close()
    print(f"\n=== Session Complete ===")
    print(f"Log saved to: {logger.log_path}")


if __name__ == "__main__":
    main()
```

### Step 2: Run the Game Session

```bash
# With LLM controller
python scripts/run_llm_session.py

# Or modify script to use baselines:
# controller_type = "fixed"   # Always same difficulty
# controller_type = "rule_based"  # Simple threshold logic
```

### Step 3: Run Dashboard (Live View)

Run in **3 separate terminals**:

**Terminal 1 - API Server:**
```bash
uvicorn pivot.api:app --reload
```

**Terminal 2 - Dashboard:**
```bash
streamlit run pivot/dashboard.py
```

**Terminal 3 - Game:**
```bash
python scripts/run_llm_session.py
```

The dashboard will show:
- Live accuracy and stats
- Rolling average metrics
- LLM decision output (JSON)

## Quick Test (No Game Window)

Test the pipeline without VizDoom:

```python
from src.discretizer import discretize_stats
from src.baseline_controllers import create_controller
from src.episode_logger import EpisodeLogger

# Simulate stats
fake_stats = {
    "kills": 5, "deaths": 2, "kdr": 2.5, "frags": 3,
    "damage_dealt": 200, "damage_taken": 100,
    "accuracy": 0.4, "duration_seconds": 90, "health_mean": 75
}

# Test discretizer
labels = discretize_stats(fake_stats)
print(f"Labels: {labels}")

# Test controller
controller = create_controller("rule_based")
decision = controller.get_difficulty_decision(labels, current_difficulty=3)
print(f"Decision: {decision}")

# Test logger
logger = EpisodeLogger("/tmp/test_session.jsonl")
logger.log_episode(0, fake_stats, labels, 3, decision)
print(f"Logged to: {logger.log_path}")
```

## Controller Comparison

Run sessions with different controllers for offline comparison:

```bash
# Session 1: LLM controller
python scripts/run_llm_session.py  # Edit to use "llm"

# Session 2: Fixed difficulty
python scripts/run_llm_session.py  # Edit to use "fixed"

# Session 3: Rule-based
python scripts/run_llm_session.py  # Edit to use "rule_based"
```

Compare logs in `runs/` directory.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module 'vizdoom'` | `pip install vizdoom` |
| `No module 'anthropic'` | `pip install anthropic` |
| `ANTHROPIC_API_KEY not set` | Export key or create `.env` file |
| Game window not responding | Ensure window has focus |
| Dashboard not updating | Check log path matches session |

## File Locations

```
frontier-hack-claude-aif/
├── src/
│   ├── human_play_env.py      # Module 1: Env wrapper
│   ├── discretizer.py         # Module 2: Stats → labels
│   ├── llm_controller.py      # Module 3: LLM decisions
│   ├── baseline_controllers.py # Module 4: Baselines
│   ├── episode_logger.py      # Module 5: JSONL logging
│   └── prompts/
│       └── difficulty_controller.txt  # LLM prompt template
├── pivot/
│   └── dashboard.py           # Streamlit dashboard
├── runs/                      # Session logs (JSONL)
├── config/
│   └── dda_deathmatch.cfg     # VizDoom config
└── scripts/
    └── run_llm_session.py     # Main runner (create this)
```
