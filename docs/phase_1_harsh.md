# Phase 1 ML B — Implementation Plan: Arnold & Multi-Tier Proxies

## Objective
Implement `PlayerProxy` using **Arnold** `.pth` models and internal ViZDoom bots, plus baseline adapters for performance comparison.

---

## 1. Proxy Development (`src/player_proxy.py`)
**Goal:** Implement the `PlayerProxy` interface to simulate a spectrum of skills from "Novice Bot" to "Arnold Champion."

* **Task 1.1: Arnold Model Proxy (The "Pro")**
    * Create `ArnoldProxy(PlayerProxy)`.
    * **Architecture:** Load the dual-network Arnold weights (Navigation DQN + Action DRQN).
    * **State Management:** Arnold is a **DRQN**; implement an internal `hidden_state` buffer that resets at the start of each episode.
    * **Preprocessing:** Implement frame resizing (color, typically $160 \times 120$ or $108 \times 60$) to match Arnold’s training input.
* **Task 1.2: Internal Bot Proxy (The "Variable")**
    * Create `BuiltInBotProxy(PlayerProxy)`.
    * Map `difficulty_input (0.0-1.0)` $\rightarrow$ `ViZDoom_skill (1-5)`.
    * Use `game.send_game_command("addbot")` logic to scale presence.
* **Task 1.3: Noise & Fatigue Injection**
    * Add an `epsilon` (exploration) parameter to **all** proxies. 
    * When triggered, the proxy returns a random action instead of the model's prediction.
    * **Purpose:** This allows us to simulate a "high skill but frustrated/tired" user for the AIF to detect.

---

## 2. Data Collection (`scripts/collect_baselines.py`)
**Goal:** Quantify performance to provide ML A with "Observation Bin" definitions.

* **Task 2.1: Profiling Matrix**
    * Run Arnold and Built-in Bots (at Skills 1, 3, 5) across 20 episodes each.
    * Track: KDR, Damage Dealt, and Accuracy.
* **Task 2.2: Data Export (`data/proxy_stats.json`)**
    * Calculate the **mean and variance** of KDR for each proxy type.
    * **Critical Output:** This JSON will be used by ML A to set the thresholds for "LOW," "MEDIUM," and "HIGH" performance observations.

---

## 3. Baseline Adapters (`src/baselines.py`)
**Goal:** Create the non-AIF control group for the hackathon pitch.

* **Task 3.1: Rule-Based DDA**
    * Implement `RuleBasedAdapter(Adapter)`.
    * **Logic:** Increment `bot_skill` if `performance == 2 (HIGH)`, decrement if `performance == 0 (LOW)`.
* **Task 3.2: Static Adapter**
    * Implement `FixedAdapter(Adapter)` that returns a hardcoded difficulty (e.g., always Skill 3).

---

## Technical Constraints for the Agent
* **Frameworks:** Arnold requires `torch`. Ensure the agent handles device selection (`cpu` vs `cuda`) automatically.
* **Interfaces:** All classes **must** inherit from `src/interfaces.py`.
* **Action Mapping:** Arnold's output must be mapped to the `EnvWrapper` action list format.
* **Cleanup:** Implement a `__del__` or `close()` method to clear GPU memory when a proxy is swapped.

---

## Exit Criteria
1.  `ArnoldProxy` can complete a full ViZDoom episode without crashing.
2.  `data/proxy_stats.json` is generated with distinct distributions for "Arnold" vs. "Skill 1 Bot."
3.  `python scripts/stub_run.py --adapter rule_based` runs successfully using the new `RuleBasedAdapter`.