# Phase 1 ML B Implementation Plan

## Overview

This plan is for Harsh / ML B. The goal is to build player proxies and baselines for VizDoom Phase 1, using `deathmatch` as the default scenario and the pretrained Arnold checkpoints in `agents/*.pth`.

### Objectives
- Wrap at least one built-in-style player proxy and one pretrained model proxy behind `src.interfaces.PlayerProxy`
- Add an `exploration_rate` parameter to proxies
- Collect fixed-difficulty calibration stats for ML A
- Add baseline adapters for fixed difficulty and rule-based difficulty adjustment
- Validate the stub pipeline with real VizDoom runs

## Files to implement
- `src/player_proxy.py`
- `src/adapters.py` (add `RuleBasedAdapter`)
- `scripts/evaluate_proxies.py`
- `data/calibration_stats.json` (output from proxy evaluation)

## Implementation steps

1. Confirm the default scenario and difficulty knobs with the SWE track.
   - Default scenario: `deathmatch`
   - Default config: `config/dda_deathmatch.cfg`
   - Difficulty values: `bot_skill` in `[1, 5]`, `num_bots` default `2`

2. Build proxy wrappers.
   - `BuiltInBotProxy`: heuristic player controller with skill-scaled action selection
   - `ModelCheckpointProxy`: loads Arnold `.pth` checkpoints and predicts actions from game state
   - Add `exploration_rate` to both proxies

3. Collect proxy calibration data.
   - Use `scripts/evaluate_proxies.py` to run episodes at fixed `bot_skill` levels
   - Record episode stats like `kdr`, `damagecount`, `duration_seconds`
   - Save output to `data/calibration_stats.json`

4. Implement baseline adapters.
   - `FixedAdapter`: constant difficulty
   - `RuleBasedAdapter`: adjust difficulty based on discretized performance observations

5. Validate integration.
   - Run `python scripts/collect_baselines.py` to confirm baseline tooling
   - Run `python scripts/evaluate_proxies.py --scenario deathmatch --proxy builtin --episodes 3`
   - Optionally run `python scripts/evaluate_proxies.py --scenario deathmatch --proxy model --model-path agents/deathmatch_shotgun.pth --episodes 3`

## Verification
- `src/player_proxy.py` imports cleanly and exposes `BuiltInBotProxy` / `ModelCheckpointProxy`
- `src/adapters.py` includes `RuleBasedAdapter`
- `scripts/evaluate_proxies.py` writes calibration data to `data/calibration_stats.json`
- `scripts/collect_baselines.py` and new proxy evaluation script run without syntax errors

## Notes
- The Arnold checkpoints in `agents/` may require input preprocessing matching the `ArnoldDRQN` architecture.
- If model loading fails, the checkpoint proxy falls back to safe random actions to maintain pipeline stability.
- This plan is intentionally focused on working, distinguishable proxies rather than perfect performance.
