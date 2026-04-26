# V3 Implementation - Module 3: LLM Controller

## Overview

Uses Anthropic Claude to decide difficulty adjustments based on player performance. Takes discretized labels from Module 2, returns difficulty level with reasoning text.

## File Locations

- `src/llm_controller.py` - Main controller function
- `src/prompts/difficulty_controller.txt` - System prompt template

## API

```python
from src.llm_controller import get_difficulty_decision

def get_difficulty_decision(
    labels: dict,                    # From discretize_stats()
    current_difficulty: int,         # 1-5
    history: list[dict] | None = None,  # Recent decisions
    target: str = "keep player performance in the 'normal' band",
    model: str = "claude-sonnet-4-20250514"
) -> dict:
    """
    Returns: {"difficulty": int, "reasoning": str}
    On failure: {"difficulty": current_difficulty, "reasoning": "Parse error..."}
    """
```

## Features

- **JSON parsing with retry**: 1 retry on parse failure, then fallback to current difficulty
- **Handles markdown**: Extracts JSON from code blocks
- **Never crashes**: Always returns valid response
- **Prompt as separate file**: Easy to iterate without code changes

## Usage Example

```python
from src.discretizer import discretize_stats
from src.llm_controller import get_difficulty_decision

# Get stats from game
stats = env.get_episode_stats()
labels = discretize_stats(stats)

# Get LLM decision
history = []  # Track previous decisions
result = get_difficulty_decision(
    labels=labels,
    current_difficulty=3,
    history=history
)

print(f"New difficulty: {result['difficulty']}")
print(f"Reasoning: {result['reasoning']}")

# Track for next decision
history.append(result)
```

## Prompt Template

Located at `src/prompts/difficulty_controller.txt`:

- Explains the controller role
- Describes input format (labels, current_difficulty, history)
- Specifies output JSON schema
- Includes few-shot examples for common scenarios
- Decision rules to prevent oscillation

## Dependencies

- `anthropic>=0.40.0` (added to requirements.txt)
- Requires `ANTHROPIC_API_KEY` environment variable

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Malformed JSON | Retry once, then fallback |
| API error | Return current difficulty with error message |
| Out of range (not 1-5) | Treat as parse failure |
| Missing keys | Treat as parse failure |

## Integration with Pipeline

```python
from src.human_play_env import HumanPlayEnvWrapper
from src.discretizer import discretize_stats
from src.llm_controller import get_difficulty_decision

env = HumanPlayEnvWrapper()
history = []
difficulty = 3

while True:
    env.reset(difficulty)
    stats = env.run_episode()
    
    labels = discretize_stats(stats)
    result = get_difficulty_decision(labels, difficulty, history)
    
    history.append(result)
    difficulty = result["difficulty"]
```
