# Player Proxy System for Active Inference DDA

## Project Context
You are building a player simulation system for a VizDoom-based Dynamic Difficulty Adjustment (DDA) engine that uses Active Inference. This component generates synthetic player data across skill levels to calibrate hidden state models (Boredom, Flow, Frustration).

## Your Deliverables
1. `src/player_proxy.py` - Continuous skill-based player proxy
2. `scripts/run_baseline_collection.py` - Data collection pipeline
3. `data/proxy_distributions.json` - Statistical output for ML team

---

## Task 1: Create PlayerProxy Class

**File:** `src/player_proxy.py`

**Requirements:**
```python
class PlayerProxy:
    """
    Simulates players at different skill levels using interpolated behavior.
    
    Args:
        skill_level (float): 0.0 (noob) to 1.0 (pro)
        base_model_path (str): Path to Arnold .pth weights
    """
    
    def __init__(self, skill_level: float, base_model_path: str):
        # Load Arnold model weights using torch.load()
        # Initialize skill parameters based on interpolation
        pass
    
    def get_action(self, game_state) -> list:
        """
        Returns action with skill-appropriate corruption.
        
        Interpolation formulas:
        - frame_skip = int(1 + (10-1) * (1 - skill_level))
        - noise_prob = 0.35 * (1 - skill_level)
        - use_persistence = skill_level < 0.4
        
        Returns:
            List of button presses for VizDoom
        """
        pass
```

**Action Persistence Logic:**
When `skill_level < 0.4`, actions should repeat for a duration sampled from:
- Geometric distribution with p=0.3 (use `np.random.geometric(0.3)`)
- This gives mean ~3 frames of the same action

**Acceptance Criteria:**
- [ ] Loads .pth file without errors
- [ ] `get_action()` returns valid VizDoom action list
- [ ] At skill=1.0: no noise, frame_skip=1
- [ ] At skill=0.0: 35% noise, frame_skip=10, persistent actions
- [ ] At skill=0.5: interpolated values

---

## Task 2: Implement Movement Entropy Tracker

**File:** `src/player_proxy.py` (add as helper class)

**Requirements:**
```python
class MovementTracker:
    """
    Tracks position entropy over an 8x8 discretized grid.
    """
    
    def __init__(self, map_bounds: tuple):
        """
        Args:
            map_bounds: ((min_x, max_x), (min_y, max_y))
        """
        self.grid = np.zeros((8, 8))
        
    def record_position(self, x: float, y: float):
        """Add position sample to grid."""
        pass
    
    def compute_entropy(self) -> float:
        """
        Shannon entropy: H = -Σ(p_i × log₂(p_i))
        where p_i = grid[i] / total_samples
        
        Returns:
            Entropy in bits (0 = stayed in one cell, ~6 = uniform exploration)
        """
        pass
```

**Acceptance Criteria:**
- [ ] Returns 0.0 if agent never moves
- [ ] Returns ~6.0 if agent visits all cells uniformly
- [ ] Handles edge case of zero-visit cells (skip in sum)

---

## Task 3: Build Data Collection Script

**File:** `scripts/run_baseline_collection.py`

**Pseudocode:**
```python
import vizdoom as vzd
from src.player_proxy import PlayerProxy, MovementTracker
import json

SKILL_LEVELS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
PILOT_EPISODES = 50
FULL_EPISODES = 200  # May adjust after pilot

def run_episode(game, proxy, tracker):
    """
    Run one episode and return metrics dict:
    {
        'hit_ratio': float,
        'damage_taken_per_sec': float,
        'movement_entropy': float,
        'kill_count': int
    }
    """
    pass

def run_pilot():
    """Run 50 episodes at skill=0.0 and skill=1.0, compute CV of hit_ratio."""
    # If CV > 0.3, print warning to increase episodes
    pass

def run_full_collection():
    """Run FULL_EPISODES for each skill level, save to JSON."""
    results = {}
    for skill in SKILL_LEVELS:
        episodes = []
        for ep in range(FULL_EPISODES):
            metrics = run_episode(...)
            episodes.append(metrics)
        
        # Compute statistics
        results[skill] = {
            'hit_ratio': {'mean': ..., 'std': ..., 'p25': ..., 'p50': ..., 'p75': ...},
            # ... same for other metrics
        }
    
    with open('data/proxy_distributions.json', 'w') as f:
        json.dump(results, f, indent=2)
```

**Acceptance Criteria:**
- [ ] Pilot runs without errors and prints CV warning if needed
- [ ] Full collection produces valid JSON with all 7 skill levels
- [ ] Each metric includes: mean, std, min, max, p25, p50, p75
- [ ] Script runs headless (no rendering)

---

## Dependency Checklist (VERIFY BEFORE STARTING)

**Critical - You Cannot Proceed Without:**
- [ ] Arnold model weights path (`.pth` file location)
- [ ] VizDoom config file path (`config/dda_deathmatch.cfg`)
- [ ] Game environment has `reset(difficulty_params)` method

**If Any Are Missing:**
Stop and ask the human for:
1. Where are the Arnold .pth files?
2. Does the SWE environment wrapper exist yet?
3. What is the exact structure of `difficulty_params`?

---

## Execution Order

1. **First:** Verify all dependencies exist
2. **Then:** Implement PlayerProxy + MovementTracker
3. **Then:** Write unit tests for entropy calculation
4. **Then:** Implement `run_episode()` function
5. **Then:** Run pilot, check variance
6. **Finally:** Run full collection only if pilot succeeds

## Error Handling Requirements

- Wrap model loading in try/except, fail gracefully with clear message
- If episode crashes, log error and skip (don't halt entire collection)
- Validate all metrics are finite (no NaN/Inf) before saving