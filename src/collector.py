"""
stats_from_df — compute episode-level scalar stats from a GameVariableRecorder DataFrame.
BotRolloutRunner — drives bots at multiple skill levels for baseline collection.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import vizdoom as vzd

from vizdoom_tracker.recorder import GameVariableRecorder
from vizdoom_tracker.session import SessionMetadata, SessionResult
from vizdoom_tracker.variables import DEATHMATCH_VARS

# 14-variable list used for agent observation registration (set_available_game_variables).
# The recorder tracks the full DEATHMATCH_VARS set independently.
GAME_VARS = [
    vzd.GameVariable.HEALTH,
    vzd.GameVariable.ARMOR,
    vzd.GameVariable.KILLCOUNT,
    vzd.GameVariable.DEATHCOUNT,
    vzd.GameVariable.DAMAGE_TAKEN,
    vzd.GameVariable.DAMAGECOUNT,
    vzd.GameVariable.HITS_TAKEN,
    vzd.GameVariable.HITCOUNT,
    vzd.GameVariable.FRAGCOUNT,
    vzd.GameVariable.SELECTED_WEAPON_AMMO,
    vzd.GameVariable.POSITION_X,
    vzd.GameVariable.POSITION_Y,
    vzd.GameVariable.VELOCITY_X,
    vzd.GameVariable.VELOCITY_Y,
]

SAMPLE_INTERVAL = 35  # tics per second; kept for any external references


def stats_from_df(df: pd.DataFrame, duration_tics: int, skill: int, scenario: str) -> dict:
    """Compute episode-level scalar stats from a GameVariableRecorder DataFrame.

    Column names are GameVariable.name.lower() (e.g. "position_x", "selected_weapon_ammo").
    Returns the same keys as the legacy compute_episode_stats() for compatibility
    with PlayerStateEstimator and JsonlRunLogger.
    """
    dur_sec = max(duration_tics / 35.0, 1.0)

    def _last(col: str) -> float:
        return float(df[col].iloc[-1]) if col in df.columns and len(df) > 0 else 0.0

    kills = _last("killcount")
    deaths = _last("deathcount")
    damage_taken = _last("damage_taken")
    damage_dealt = _last("damagecount")
    hits_taken = _last("hits_taken")
    frags = _last("fragcount")

    health_arr = df["health"].values if "health" in df.columns and len(df) > 0 else np.array([100.0])

    # Movement entropy: discretise 2D position into a 20x20 grid
    px = df["position_x"].values if "position_x" in df.columns else np.array([])
    py = df["position_y"].values if "position_y" in df.columns else np.array([])
    movement_entropy = 0.0
    if len(px) > 1:
        px_bins = np.clip(((px - px.min()) / (px.max() - px.min() + 1e-6) * 20).astype(int), 0, 19)
        py_bins = np.clip(((py - py.min()) / (py.max() - py.min() + 1e-6) * 20).astype(int), 0, 19)
        hist, _ = np.histogramdd(np.stack([px_bins, py_bins], axis=1), bins=20)
        hist = hist / (hist.sum() + 1e-9)
        movement_entropy = float(-np.sum(hist * np.log(hist + 1e-9)))

    return {
        "scenario": scenario,
        "skill": skill,
        "duration_tics": duration_tics,
        "duration_seconds": dur_sec,
        "kdr": kills / max(deaths, 1.0),
        "kill_rate": kills / dur_sec,
        "death_rate": deaths / dur_sec,
        "damage_efficiency": damage_dealt / max(damage_taken, 1.0),
        "health_mean": float(health_arr.mean()),
        "health_min": float(health_arr.min()),
        "health_variance": float(health_arr.var()),
        "survival_frac": max(0.0, 1.0 - deaths / max(dur_sec / 60.0 * 5.0, 1.0)),
        "movement_entropy": movement_entropy,
        "hit_accuracy": kills / max(hits_taken, 1.0),
        "final_kills": kills,
        "final_deaths": deaths,
        "final_frags": frags,
    }


class BotRolloutRunner:
    """
    Runs VizDoom episodes with the engine's built-in bots as the only player.
    Used to collect performance baselines at each skill level.
    Saves session data as Parquet files via SessionResult.
    """

    def __init__(self, scenario_cfg: str, num_episodes: int = 10, num_bots: int = 3,
                 window_visible: bool = False):
        self.scenario_cfg = scenario_cfg
        self.num_episodes = num_episodes
        self.num_bots = num_bots
        self.window_visible = window_visible

        cfg_name = Path(scenario_cfg).stem
        self.scenario_name = cfg_name.replace("dda_", "")

    def _make_game(self, skill: int) -> vzd.DoomGame:
        game = vzd.DoomGame()
        game.load_config(self.scenario_cfg)
        wad_name = os.path.basename(game.get_doom_scenario_path())
        game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, wad_name))
        game.set_doom_skill(skill)
        game.set_window_visible(self.window_visible)
        game.set_mode(vzd.Mode.PLAYER)
        game.add_game_args("+sv_cheats 1")
        game.set_available_game_variables(GAME_VARS)
        game.init()
        return game

    def run_skill_level(self, skill: int, save_dir: str) -> list[dict]:
        """Run num_episodes bot episodes at the given skill, return list of stats."""
        print(f"  Running {self.num_episodes} episodes at skill {skill}...")
        game = self._make_game(skill)
        recorder = GameVariableRecorder(DEATHMATCH_VARS)
        episode_stats = []

        for ep in range(self.num_episodes):
            game.new_episode()
            for _ in range(self.num_bots):
                game.send_game_command("addbot")
            recorder.reset()

            while not game.is_episode_finished():
                game.advance_action(1)
                recorder.record(game)

            df = recorder.to_dataframe()
            stats = stats_from_df(df, recorder.episode_tic, skill, self.scenario_name)

            meta = SessionMetadata(
                session_id=f"skill{skill}_ep{ep:04d}",
                start_utc=datetime.now(timezone.utc).isoformat(),
                scenario=self.scenario_name,
                num_bots=self.num_bots,
                episode_timeout_tics=recorder.episode_tic,
                sample_interval_tics=recorder._sample_every,
                tic_rate=35,
                variables=[v.name for v in DEATHMATCH_VARS],
                total_tics=recorder.episode_tic,
                total_samples=recorder.num_samples,
                extra={"skill": skill, "episode": ep},
            )
            SessionResult(df=df, metadata=meta).save(save_dir)
            episode_stats.append(stats)
            print(f"    ep {ep}: kills={stats['final_kills']:.0f} deaths={stats['final_deaths']:.0f} "
                  f"kdr={stats['kdr']:.2f} health={stats['health_mean']:.1f}")

        game.close()
        return episode_stats

    def run_all_skills(self, save_dir: str) -> dict:
        """Run all skill levels and return nested stats dict."""
        all_stats: dict[int, list] = {}
        for skill in range(1, 6):
            skill_dir = os.path.join(save_dir, f"skill_{skill}")
            all_stats[skill] = self.run_skill_level(skill, skill_dir)
        return all_stats
