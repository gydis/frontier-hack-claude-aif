"""
DDAPipeline — orchestrates the full DDA game loop.

The player plays VizDoom normally; at each episode boundary the DM
observes episode statistics and adjusts difficulty for the next episode.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import vizdoom as vzd

from src.actuator import DifficultyActuator
from src.baselines import BotBaselines
from src.collector import EpisodeCollector, GAME_VARS
from src.dungeon_master import DungeonMasterAgent
from src.state_estimator import PlayerStateEstimator


class DDAPipeline:
    """
    Full DDA session: init game → episode loop → DM decision → repeat.

    Args:
        scenario_cfg:    path to one of config/dda_*.cfg
        baselines_path:  path to data/baselines.json (may not exist yet)
        initial_skill:   starting difficulty (1-5)
        window_visible:  show the game window (True for human play)
        record_dir:      if set, records .lmp replay files here
    """

    def __init__(
        self,
        scenario_cfg: str,
        baselines_path: str = "data/baselines.json",
        initial_skill: int = 3,
        window_visible: bool = True,
        record_dir: str | None = None,
    ):
        self.scenario_cfg = scenario_cfg
        self.initial_skill = initial_skill
        self.record_dir = record_dir

        scenario_stem = Path(scenario_cfg).stem.replace("dda_", "")
        self.scenario_name = scenario_stem

        # ---- Components ----
        self.baselines = BotBaselines.load_or_fallback(baselines_path)
        self.dm = DungeonMasterAgent(initial_skill=initial_skill)
        self.estimator = PlayerStateEstimator(self.baselines)

        # ---- Game ----
        self.game = vzd.DoomGame()
        self.game.load_config(scenario_cfg)
        # Resolve WAD path against the installed vizdoom scenarios directory
        wad_name = os.path.basename(self.game.get_doom_scenario_path())
        self.game.set_doom_scenario_path(
            os.path.join(vzd.scenarios_path, wad_name))
        self.game.set_window_visible(window_visible)
        # SPECTATOR: human's keyboard drives the player; code actions are ignored.
        # advance_action(1) pumps SDL events (keeps window responsive) and steps the sim.
        # A sleep caps the loop at 35 fps so the game runs at normal Doom speed.
        self.game.set_mode(vzd.Mode.SPECTATOR)
        self.game.set_available_game_variables(GAME_VARS)
        self.game.set_doom_skill(initial_skill)
        self.game.add_game_args("+sv_cheats 1")
        self.game.init()

        self.actuator = DifficultyActuator(self.game, initial_skill)
        self.collector = EpisodeCollector(
            self.game, self.scenario_name, initial_skill)

        # ---- Session state ----
        self._session_log: list[dict] = []
        self._session_dir: str | None = None

    # ------------------------------------------------------------------

    def run_session(self, num_episodes: int = 20, session_dir: str = "data/sessions") -> list[dict]:
        """
        Run a full human-play session.

        Between episodes the DM observes stats and adjusts difficulty.
        Returns the session log (one dict per episode).
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = os.path.join(session_dir, ts)
        Path(self._session_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n=== DDA Session starting — {num_episodes} episodes ===")
        print(
            f"Scenario: {self.scenario_name}  |  Initial skill: {self.initial_skill}")
        if self.baselines.is_fallback():
            print(
                "WARNING: using fallback baselines — run collect_baselines.py for better estimates")
        print()

        for ep in range(num_episodes):
            log_entry = self._run_episode(ep)
            self._session_log.append(log_entry)
            self._save_session_log()
            self._print_episode_summary(ep, log_entry)

        print("\n=== Session complete ===")
        self._print_session_summary()
        return self._session_log

    # ------------------------------------------------------------------

    def _run_episode(self, ep_idx: int) -> dict:
        rec_path = ""
        if self.record_dir:
            Path(self.record_dir).mkdir(parents=True, exist_ok=True)
            rec_path = os.path.join(self.record_dir, f"ep{ep_idx:04d}.lmp")

        self.collector.skill = self.actuator.current_skill
        self.collector.reset()

        if rec_path:
            self.game.new_episode(rec_path)
        else:
            self.game.new_episode()

        # Add any pending bots from the DM's ADD_BOT action
        if ep_idx > 0:
            self.actuator.add_pending_bots(
                self._last_action if hasattr(self, "_last_action") else 1)

        t_start = time.time()
        mid_ep_bot_added = False

        tic_duration = 1.0 / 35.0
        _queued_respawn = False
        while not self.game.is_episode_finished():
            # Detect death via is_player_dead() OR DEAD game variable (covers
            # the full respawn-countdown window, not just the brief death animation).
            is_dead = self.game.is_player_dead()
            if not is_dead:
                try:
                    is_dead = bool(self.game.get_game_variable(
                        vzd.GameVariable.DEAD))
                except Exception:
                    pass

            if is_dead:
                if not _queued_respawn:
                    self.game.respawn_player()   # queue respawn immediately
                    _queued_respawn = True
                # With ticrate=10000 advance_action is non-blocking, so no sleep
                # needed here — the loop races through the countdown at ~3000 tics/s.
                self.game.advance_action(1)
            else:
                _queued_respawn = False
                t0 = time.time()
                # pumps SDL events; SPECTATOR ignores the action
                self.game.advance_action(1)
                self.collector.step()
                elapsed = time.time() - t0
                sleep_for = tic_duration - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

            # Mid-episode dominance check (only before 60s mark)
            if not mid_ep_bot_added:
                w = self.collector.get_episode_stats()
                if self.actuator.check_mid_episode_dominance(w):
                    self.game.send_game_command("addbot")
                    self.actuator.num_bots += 1
                    mid_ep_bot_added = True

        t_end = time.time()
        episode_stats = self.collector.save(self._session_dir, ep_idx)
        episode_stats["wall_time"] = t_end - t_start

        # DM decision
        perf_obs, trend_obs = self.estimator.estimate(
            episode_stats, self.actuator.current_skill)
        action = self.dm.step(perf_obs, trend_obs)
        beliefs = self.dm.get_belief_summary()

        # Apply action (skill change takes effect on next new_episode())
        new_skill, action_desc = self.actuator.apply_action(action)
        self._last_action = action

        log_entry = {
            "episode": ep_idx,
            "scenario": self.scenario_name,
            "skill": self.actuator.current_skill,
            "perf_obs": int(perf_obs),
            "trend_obs": int(trend_obs),
            "action": int(action),
            "action_desc": action_desc,
            "new_skill": new_skill,
            "dominant_state": self.dm.dominant_state(),
            "beliefs": beliefs,
            "stats": episode_stats,
        }
        return log_entry

    # ------------------------------------------------------------------

    def _save_session_log(self):
        log_path = os.path.join(self._session_dir, "session_log.jsonl")
        with open(log_path, "w") as f:
            for entry in self._session_log:
                f.write(json.dumps(entry) + "\n")

    def _print_episode_summary(self, ep: int, entry: dict):
        s = entry["stats"]
        print(
            f"  ep{ep:03d} | skill={entry['skill']} "
            f"kills={s.get('final_kills', 0):.0f} deaths={s.get('final_deaths', 0):.0f} "
            f"kdr={s.get('kdr', 0):.2f} hp={s.get('health_mean', 0):.0f} "
            f"| state={entry['dominant_state']:10s} "
            f"perf={'LOW MED HIG'.split()[entry['perf_obs']]} "
            f"trend={'DEC STA IMP'.split()[entry['trend_obs']]} "
            f"→ {entry['action_desc']}"
        )

    def _print_session_summary(self):
        if not self._session_log:
            return
        skills = [e["skill"] for e in self._session_log]
        states = [e["dominant_state"] for e in self._session_log]
        print(f"  Skill range: {min(skills)}–{max(skills)}")
        from collections import Counter
        state_counts = Counter(states)
        print(f"  State distribution: {dict(state_counts)}")
        flow_frac = state_counts.get("FLOW", 0) / len(self._session_log)
        print(f"  Flow fraction: {flow_frac:.0%}")

    def reset_with_difficulty(self, skill: int, record_path: str = None):
        """
        Reset episode with specific difficulty level.
        Used by proxy baseline collection.

        Args:
            skill: Doom skill level (1-5)
            record_path: Optional .lmp recording path
        """
        self.game.set_doom_skill(skill)
        self.actuator.current_skill = skill
        self.collector.skill = skill
        self.collector.reset()

        if record_path:
            self.game.new_episode(record_path)
        else:
            self.game.new_episode()

        return self.game.get_state()

    def close(self):
        self.game.close()
