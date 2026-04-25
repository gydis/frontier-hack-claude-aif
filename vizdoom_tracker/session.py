"""
Top-level session orchestrator and storage.

DeathmatchSession
    Initialises VizDoom, adds bots, runs the game loop, and returns a
    SessionResult when the episode finishes.

SessionResult
    Thin container (DataFrame + SessionMetadata) with save / load helpers.
    Parquet format: snappy-compressed, session metadata embedded in schema
    metadata for fully self-contained files. A human-readable JSON sidecar
    is also written alongside each file.

Typical usage
─────────────
    result = DeathmatchSession(num_bots=3, episode_minutes=5).run()
    result.save("sessions/")

    # Later:
    result = SessionResult.load("sessions/2025-01-15_abc123def456.parquet")
    df = result.df           # game_time_s-indexed DataFrame
    df[COMBAT.column_names]  # slice to a variable group
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import vizdoom as vzd

from .agent import RandomAgent
from .recorder import DOOM_TIC_RATE, GameVariableRecorder
from .variables import DEATHMATCH_VARS

logger = logging.getLogger(__name__)

_SCHEMA_META_KEY = b"session_metadata"


# ── Metadata ──────────────────────────────────────────────────────────────────

@dataclass
class SessionMetadata:
    session_id: str
    start_utc: str
    scenario: str
    num_bots: int
    episode_timeout_tics: int
    sample_interval_tics: int
    tic_rate: int
    variables: List[str]       # GameVariable.name strings, in column order
    total_tics: int = 0
    total_samples: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SessionMetadata":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ── Result container ──────────────────────────────────────────────────────────

class SessionResult:
    """Recorded time-series DataFrame and associated session metadata."""

    def __init__(self, df: pd.DataFrame, metadata: SessionMetadata) -> None:
        self.df = df
        self.metadata = metadata

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, directory: str | Path) -> Path:
        """Write <date>_<session_id>.parquet and a .json sidecar to `directory`.

        Session metadata is embedded in the Parquet schema so the file is
        fully self-contained; the JSON sidecar makes inspection trivial.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        date_str = self.metadata.start_utc[:10]
        stem = f"{date_str}_{self.metadata.session_id}"
        parquet_path = directory / f"{stem}.parquet"
        json_path = directory / f"{stem}.json"

        table = pa.Table.from_pandas(self.df, preserve_index=True)
        schema_meta = {
            **(table.schema.metadata or {}),
            _SCHEMA_META_KEY: json.dumps(self.metadata.to_dict()).encode(),
        }
        table = table.replace_schema_metadata(schema_meta)
        pq.write_table(table, parquet_path, compression="snappy")

        with open(json_path, "w") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

        logger.info("Saved %d samples → %s", len(self.df), parquet_path)
        return parquet_path

    @classmethod
    def load(cls, path: str | Path) -> "SessionResult":
        """Load a previously saved Parquet session file."""
        path = Path(path)
        table = pq.read_table(path)
        df = table.to_pandas()

        metadata: Optional[SessionMetadata] = None
        raw_meta = (table.schema.metadata or {}).get(_SCHEMA_META_KEY)
        if raw_meta:
            metadata = SessionMetadata.from_dict(json.loads(raw_meta))
        else:
            json_path = path.with_suffix(".json")
            if json_path.exists():
                with open(json_path) as f:
                    metadata = SessionMetadata.from_dict(json.load(f))

        return cls(df=df, metadata=metadata)

    def __repr__(self) -> str:
        sid = self.metadata.session_id if self.metadata else "?"
        return f"SessionResult(session_id={sid!r}, samples={len(self.df)})"


# ── Session orchestrator ──────────────────────────────────────────────────────

class DeathmatchSession:
    """Run a headless VizDoom deathmatch episode and record game state.

    The controlled player uses a `RandomAgent`; `num_bots` VizDoom built-in
    bots are added as opponents after the episode starts.

    Parameters
    ----------
    num_bots:
        Number of VizDoom bots to add as opponents (1–7 recommended).
    episode_minutes:
        Wall-time episode length; translated to tics at `DOOM_TIC_RATE`.
    variables:
        Sequence of `vzd.GameVariable` to track. Defaults to `DEATHMATCH_VARS`
        (all meaningful deathmatch variables, ~51 columns).
    sample_hz:
        How many snapshots per second. 1.0 → one row per second (35 tics).
    output_dir:
        If set, `run()` automatically calls `result.save(output_dir)`.
    headless:
        Suppress the game window (default True; set False to watch).
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        num_bots: int = 3,
        episode_minutes: float = 5.0,
        variables: Optional[Sequence[vzd.GameVariable]] = None,
        sample_hz: float = 1.0,
        output_dir: Optional[str | Path] = None,
        headless: bool = True,
        seed: Optional[int] = None,
        mode: str = "player",
        scenario: str = "doom.cfg",
    ) -> None:
        self.num_bots = num_bots
        self.episode_tics = round(episode_minutes * 60 * DOOM_TIC_RATE)
        self.variables: List[vzd.GameVariable] = list(variables or DEATHMATCH_VARS)
        self.sample_interval_tics = max(1, round(DOOM_TIC_RATE / sample_hz))
        self.output_dir = Path(output_dir) if output_dir else None
        self.headless = headless
        self.seed = seed
        self.mode = vzd.Mode.PLAYER if mode == "player" else vzd.Mode.SPECTATOR
        self.scenario = scenario

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> SessionResult:
        """Execute one episode, return a `SessionResult`."""
        session_id = uuid.uuid4().hex[:12]
        start_utc = datetime.now(timezone.utc).isoformat()

        recorder = GameVariableRecorder(self.variables, self.sample_interval_tics)
        game = self._build_game()
        game.init()

        try:
            df, total_tics = self._episode_loop(game, recorder, session_id)
        finally:
            game.close()

        metadata = SessionMetadata(
            session_id=session_id,
            start_utc=start_utc,
            scenario=self.scenario,
            num_bots=self.num_bots,
            episode_timeout_tics=self.episode_tics,
            sample_interval_tics=self.sample_interval_tics,
            tic_rate=DOOM_TIC_RATE,
            variables=[v.name for v in self.variables],
            total_tics=total_tics,
            total_samples=len(df),
        )

        result = SessionResult(df=df, metadata=metadata)
        if self.output_dir is not None:
            result.save(self.output_dir)
        return result

    # ── Internals ─────────────────────────────────────────────────────────────

    def _episode_loop(
        self,
        game: vzd.DoomGame,
        recorder: GameVariableRecorder,
        session_id: str,
    ):
        game.new_episode()
        agent = RandomAgent(game, seed=self.seed)

        for _ in range(self.num_bots):
            game.send_game_command("addbot")

        recorder.reset()
        logger.info(
            "Session %s started | bots=%d  episode=%.1f min  sample_interval=%d tics",
            session_id,
            self.num_bots,
            self.episode_tics / DOOM_TIC_RATE / 60,
            self.sample_interval_tics,
        )

        while not game.is_episode_finished() or not game.is_player_dead():
            state = game.get_state()
            if state is None:
                break
            action = agent.act(state)
            game.make_action(action)
            recorder.record(game)

        logger.info(
            "Session %s finished | tics=%d  samples=%d",
            session_id,
            recorder.episode_tic,
            recorder.num_samples,
        )
        return recorder.to_dataframe(), recorder.episode_tic

    def _build_game(self) -> vzd.DoomGame:
        game = vzd.DoomGame()

        game.set_doom_scenario_path(
            str(Path(vzd.scenarios_path) / self.scenario)
        )
        game.set_mode(self.mode)
        game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        game.set_screen_format(vzd.ScreenFormat.RGB24)
        game.set_render_hud(True)
        game.set_render_crosshair(True)
        game.set_render_particles(True)
        game.set_window_visible(not self.headless)
        game.set_episode_timeout(self.episode_tics)
        game.set_episode_start_time(1)
        game.set_doom_skill(3)

        game.set_available_buttons([
            vzd.Button.ATTACK,
            vzd.Button.MOVE_FORWARD,
            vzd.Button.MOVE_BACKWARD,
            vzd.Button.MOVE_LEFT,
            vzd.Button.MOVE_RIGHT,
            vzd.Button.TURN_LEFT,
            vzd.Button.TURN_RIGHT,
            vzd.Button.SELECT_NEXT_WEAPON,
            vzd.Button.SELECT_PREV_WEAPON,
            vzd.Button.SPEED,
            vzd.Button.JUMP,
            vzd.Button.USE,
        ])

        # Registering variables here also makes them available via state.game_variables
        game.set_available_game_variables(self.variables)

        # +sv_nomonsters prevents random monster spawns on maps that have them;
        # -deathmatch (true multiplayer flag) is intentionally omitted — it
        # triggers network initialisation that segfaults in headless environments.
        # Bots are added via send_game_command("addbot") after new_episode().
        # game.add_game_args("+sv_nomonsters 1")

        if self.seed is not None:
            game.set_seed(self.seed)

        return game
