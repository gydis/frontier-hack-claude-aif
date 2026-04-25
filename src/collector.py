"""
EpisodeCollector — samples game variables each tic and computes episode statistics.
BotRolloutRunner — drives bots at multiple skill levels for baseline collection.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import vizdoom as vzd


# Ordered list matching available_game_variables in DDA configs
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

VAR_NAMES = [
    "health", "armor", "killcount", "deathcount",
    "damage_taken", "damagecount", "hits_taken", "hitcount",
    "fragcount", "ammo", "pos_x", "pos_y", "vel_x", "vel_y",
]

# Sample every N tics (35 tics = 1 second at default ticrate)
SAMPLE_INTERVAL = 35


def compute_episode_stats(arrays: dict, duration_tics: int, scenario: str, skill: int) -> dict:
    """Compute scalar episode-level features from time-series arrays."""
    dur_sec = max(duration_tics / 35.0, 1.0)

    health = arrays["health"]
    kills = float(arrays["killcount"][-1]) if len(arrays["killcount"]) > 0 else 0.0
    deaths = float(arrays["deathcount"][-1]) if len(arrays["deathcount"]) > 0 else 0.0
    damage_taken = float(arrays["damage_taken"][-1]) if len(arrays["damage_taken"]) > 0 else 0.0
    damage_dealt = float(arrays["damagecount"][-1]) if len(arrays["damagecount"]) > 0 else 0.0
    hits_taken = float(arrays["hits_taken"][-1]) if len(arrays["hits_taken"]) > 0 else 0.0
    frags = float(arrays["fragcount"][-1]) if len(arrays["fragcount"]) > 0 else 0.0

    health_arr = np.array(health, dtype=np.float32) if len(health) > 0 else np.array([100.0])

    # Movement entropy: discretise 2D position into a 20x20 grid
    px = np.array(arrays["pos_x"], dtype=np.float32)
    py = np.array(arrays["pos_y"], dtype=np.float32)
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


class EpisodeCollector:
    """Collects game variable time-series during a single episode."""

    def __init__(self, game: vzd.DoomGame, scenario: str, skill: int):
        self.game = game
        self.scenario = scenario
        self.skill = skill
        self._buffers: dict[str, list] = {n: [] for n in VAR_NAMES}
        self._tic_counter = 0
        self._start_tic = 0

    def reset(self):
        self._buffers = {n: [] for n in VAR_NAMES}
        self._tic_counter = 0
        state = self.game.get_state()
        self._start_tic = state.tic if state is not None else 0

    def step(self):
        """Call once per game tic."""
        self._tic_counter += 1
        if self._tic_counter % SAMPLE_INTERVAL != 0:
            return
        state = self.game.get_state()
        if state is None:
            return
        for i, name in enumerate(VAR_NAMES):
            self._buffers[name].append(float(state.game_variables[i]))

    def get_episode_stats(self) -> dict:
        state = self.game.get_state()
        current_tic = state.tic if state is not None else self._tic_counter
        duration = current_tic - self._start_tic
        return compute_episode_stats(self._buffers, duration, self.scenario, self.skill)

    def save(self, save_dir: str, episode_idx: int):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        base = os.path.join(save_dir, f"ep{episode_idx:04d}")
        arrays = {n: np.array(v, dtype=np.float32) for n, v in self._buffers.items()}
        np.savez(base + ".npz", **arrays)
        stats = self.get_episode_stats()
        with open(base + ".json", "w") as f:
            json.dump(stats, f, indent=2)
        return stats


class BotRolloutRunner:
    """
    Runs VizDoom episodes with the engine's built-in bots as the only player.
    Used to collect performance baselines at each skill level.
    """

    def __init__(self, scenario_cfg: str, num_episodes: int = 10, num_bots: int = 3,
                 window_visible: bool = False):
        self.scenario_cfg = scenario_cfg
        self.num_episodes = num_episodes
        self.num_bots = num_bots
        self.window_visible = window_visible

        # Derive scenario name from config path
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
        # Register all game variables we need
        game.set_available_game_variables(GAME_VARS)
        game.init()
        return game

    def run_skill_level(self, skill: int, save_dir: str) -> list[dict]:
        """Run num_episodes bot episodes at the given skill, return list of stats."""
        print(f"  Running {self.num_episodes} episodes at skill {skill}...")
        game = self._make_game(skill)
        collector = EpisodeCollector(game, self.scenario_name, skill)
        episode_stats = []

        for ep in range(self.num_episodes):
            game.new_episode()
            for _ in range(self.num_bots):
                game.send_game_command("addbot")
            collector.reset()

            while not game.is_episode_finished():
                # Pass empty action — bots drive themselves
                game.advance_action(1)
                collector.step()

            stats = collector.save(save_dir, ep + skill * 100)
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
