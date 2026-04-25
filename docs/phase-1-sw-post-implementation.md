# Phase 1 SWE Implementation Summary

## What Was Implemented

Phase 1 establishes the core interfaces and infrastructure for the DDA system. This unblocks ML A (AIF adapter) and ML B (proxies, baselines) to build their components in parallel.

### New Files

| File | Purpose |
|------|---------|
| `src/interfaces.py` | Abstract base classes defining the 4 core interfaces |
| `src/run_logger.py` | JSONL logger with crash-safe writes and numpy serialization |
| `src/env_wrapper.py` | ViZDoom deathmatch environment wrapper |
| `src/stubs.py` | Stub implementations for testing without real components |
| `src/config_loader.py` | YAML experiment config loader with validation |
| `src/adapters.py` | Adapter implementations (AIFAdapter wrapping DungeonMasterAgent, FixedAdapter) |
| `config/example_experiment.yaml` | Example experiment configuration |
| `scripts/stub_run.py` | End-to-end stub script for pipeline verification |

### Interfaces Defined

```python
# src/interfaces.py

class Adapter(ABC):
    def choose_difficulty(self) -> dict        # Returns {"bot_skill": int, "num_bots": int}
    def update(self, obs_aif: dict) -> None    # Receives {"performance": 0-2, "trend": 0-2}
    def get_belief(self) -> dict | None        # Returns belief state for logging

class PlayerProxy(ABC):
    @property
    def id(self) -> str                        # Unique identifier
    def act(self, obs) -> Any                  # Returns action given observation

class EnvWrapper(ABC):
    def reset(self, difficulty: dict) -> None  # Start episode with difficulty
    def get_state(self) -> Any                 # Current observation
    def step(self, action) -> (float, bool)    # Execute action, return (reward, done)
    def get_episode_stats(self) -> dict        # Episode statistics

class RunLogger(ABC):
    def write(self, record: dict) -> None      # Append record to JSONL
    def close() -> None                        # Close file
```

### Log Record Schema

Each episode produces a JSONL record with these fields:

```json
{
  "episode_idx": 0,
  "difficulty": {"bot_skill": 3, "num_bots": 2},
  "raw_stats": {
    "frags": 9,
    "deaths": 4,
    "kdr": 1.52,
    "accuracy": 0.17,
    "damage_taken": 117,
    "damage_dealt": 266,
    "duration_seconds": 41.8
  },
  "obs_aif": {"performance": 1, "trend": 1},
  "belief": null,
  "proxy_id": "stub_random",
  "adapter_type": "stub",
  "timestamp": "2026-04-25T14:12:54.072033",
  "experiment_name": "stub_run"
}
```

Log files are written to `runs/<HHMMSS>_<experiment_name>.jsonl`.

---

## How to Run

### Stub End-to-End (No ViZDoom Required)

```bash
python scripts/stub_run.py
```

This runs 5 episodes with fake stats, logs to `runs/`. Useful for testing the pipeline without ViZDoom.

Options:
- `--episodes N` — Number of episodes (default: 5)
- `--real-env` — Use actual ViZDoom (requires ViZDoom installed)

### With Real ViZDoom

```bash
python scripts/stub_run.py --real-env --episodes 3
```

Requires ViZDoom and the deathmatch scenario.

### Config Loader Test

```bash
python -c "from src.config_loader import load_experiment_config; print(load_experiment_config('config/example_experiment.yaml'))"
```

---

## How to Test

### 1. Verify Interfaces Import

```bash
python -c "from src.interfaces import Adapter, PlayerProxy, EnvWrapper, RunLogger"
```

### 2. Verify ABCs Cannot Be Instantiated

```bash
python -c "from src.interfaces import Adapter; Adapter()"
# Should raise TypeError
```

### 3. Verify Stub Run Produces Valid JSONL

```bash
python scripts/stub_run.py
head -1 runs/*.jsonl | python -m json.tool
```

### 4. Verify Stubs Work Independently

```bash
python -c "
from src.stubs import StubAdapter, StubPlayerProxy
a = StubAdapter()
p = StubPlayerProxy('test')
print(a.choose_difficulty())
print(p.act(None))
"
```

---

## Current Limitations

### Waiting for Sync

1. **Discretization function** — The `obs_aif` mapping (raw stats → AIF observations) uses placeholder logic. ML A needs to specify:
   - Bin edges for performance (currently: kdr < 0.8 → LOW, > 2.0 → HIGH)
   - Trend calculation method (currently hardcoded to STABLE)

2. **Real proxies** — `StubPlayerProxy` takes random actions. ML B will implement:
   - Built-in bot proxy
   - PPO checkpoint proxy

3. **AIFAdapter requires jax/pymdp** — The `AIFAdapter` wraps `DungeonMasterAgent` which depends on ML A's stack. Stubs work without these dependencies.

### Not Yet Implemented

- **Experiment runner** — `scripts/stub_run.py` is a proof-of-concept. Phase 2 adds a proper runner that reads YAML configs and supports adapter/proxy registration.
- **Dashboard integration** — The existing `src/dashboard.py` (pygame overlay) is not yet connected to the new `RunLogger`. Phase 3 builds a Streamlit dashboard that reads JSONL logs.

### Known Issues

- `pyyaml` was added as a dependency (not in any requirements file)
- EnvWrapper not tested with real ViZDoom in this phase

---

## For ML A

Implement your AIF adapter by subclassing `Adapter`:

```python
from src.interfaces import Adapter

class MyAIFAdapter(Adapter):
    def choose_difficulty(self) -> dict:
        # Return difficulty based on current belief
        return {"bot_skill": self._current_skill, "num_bots": self._num_bots}
    
    def update(self, obs_aif: dict) -> None:
        # obs_aif = {"performance": 0-2, "trend": 0-2}
        # Update your pymdp agent here
        pass
    
    def get_belief(self) -> dict | None:
        # Return engagement belief distribution
        return {"engagement": [0.2, 0.6, 0.2]}
```

Or use the provided `AIFAdapter` in `src/adapters.py` which wraps `DungeonMasterAgent`.

---

## For ML B

Implement your player proxy by subclassing `PlayerProxy`:

```python
from src.interfaces import PlayerProxy

class PPOProxy(PlayerProxy):
    @property
    def id(self) -> str:
        return "ppo_v1"
    
    def act(self, obs) -> list:
        # obs is from env.get_state() (ViZDoom state)
        # Return action as list of button presses
        return self.model.predict(obs)
```

Implement baseline adapters by subclassing `Adapter`:

```python
from src.interfaces import Adapter

class RuleBasedAdapter(Adapter):
    def choose_difficulty(self) -> dict:
        return {"bot_skill": self._skill, "num_bots": 2}
    
    def update(self, obs_aif: dict) -> None:
        # Simple heuristic: lower difficulty if struggling
        if obs_aif["performance"] == 0:  # LOW
            self._skill = max(1, self._skill - 1)
        elif obs_aif["performance"] == 2:  # HIGH
            self._skill = min(5, self._skill + 1)
    
    def get_belief(self) -> dict | None:
        return None
```

---

## Next Steps (Phase 2)

1. Agree on discretization function with ML A
2. Implement experiment runner with YAML config support
3. Add adapter/proxy registry for config-based instantiation
4. Integration test with real AIF adapter + real proxy
