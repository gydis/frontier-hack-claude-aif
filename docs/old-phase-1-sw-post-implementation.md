This application demonstrates dynamic difficulty adjustment in a ViZDoom deathmatch environment using a large language model as the difficulty controller. A human player plays a series of deathmatch rounds against opponent bots. Between rounds, the system collects performance statistics (frags, deaths, accuracy, damage), discretizes them into qualitative labels, and passes them — along with recent decision history — to an LLM. The LLM returns a new difficulty level and a short explanation of its reasoning. The system applies the new difficulty and the next round begins. A live dashboard displays the player's performance over time, the difficulty trajectory, and the LLM's reasoning at each decision point. Two control conditions — fixed difficulty and a hand-coded rule-based adjuster — are supported for offline comparison.

IMPORTANT! We already have code that tracks ‘features’ of the gameplay in @vizdoom_tracker. Keep it in mind, don’t rewrite it, but use and integrate it
V1 Modules
Env wrapper for human-play mode. ViZDoom in real-time with a visible window, keyboard/mouse player input, opponent bots spawned per config. Exposes reset(difficulty) / get_episode_stats() and fires a callback at episode end. The difficulty argument is a single integer 1–5, mapped internally to bot count, bot skill, and time/frag limit. Keep this opaque — LLM only sees the integer level, not the underlying knobs. May already partly exist in actuator.py / game_loop.py.


Discretizer (hardcoded). Hand-coded thresholds: accuracy < 0.3 → "poor", 0.3–0.6 → "normal", > 0.6 → "high". Same shape for frags-per-minute, K/D ratio, damage taken. Output: a flat dict of label strings. Thresholds calibrated for human play, not bot play. Likely lives in state_estimator.py.


LLM controller module. Single function that takes discretized labels + current difficulty + recent decision history + a one-line target ("keep player performance in the 'normal' band") and returns a parsed response with the next difficulty and reasoning text. Use the Anthropic SDK directly; no agent framework.


Prompt template. System prompt explaining the role ("you're a difficulty controller for a deathmatch game played by a human, your goal is to keep them engaged"), the input format, the output JSON schema, and 1–2 few-shot examples. Iterate the prompt as a separate concern from the code.


JSON parser with retry. LLMs occasionally return malformed JSON. One retry on failure, then fall back to "keep current difficulty." Never crash the run on a parse error.


Baselines. Fixed-difficulty controller (constant integer) and rule-based controller (if-stats-good-then-harder, simple thresholds). Same interface as the LLM controller — swappable via config.


Logger. One JSONL record per episode: episode number, raw stats, discretized labels, applied difficulty, controller decision, full LLM prompt, full LLM response, reasoning text, timestamp. Crash-safe append. Extends what's in collector.py.


Dashboard (Streamlit). Three views:


Live — tails an active log. Stats over time, difficulty over time, latest LLM reasoning text in a side panel. Recording-friendly: large fonts, clear sections, color-coded difficulty changes. The reasoning panel is the killer feature.
Comparison — load multiple completed runs, plot LLM vs. fixed vs. rule-based on shared axes.
Replay — scrub through a finished run, see the LLM's reasoning at each episode.