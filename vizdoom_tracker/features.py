"""
Feature extraction pipeline for VizDoom deathmatch sessions.

Computes behavioral and kinematic features from recorded game-variable time
series. Features are designed for modeling player engagement and driving
active-inference-based difficulty adjustment.

Quick-start
───────────
    from vizdoom_tracker import SessionResult, extract_features

    result = SessionResult.load("sessions/2026-04-25_abc123.parquet")
    fr = extract_features(result)
    fr.save("sessions/")   # writes sessions/features/<date>_<id>_features.parquet

Load saved features
───────────────────
    from vizdoom_tracker.features import FeatureResult
    fr = FeatureResult.load("sessions/features/2026-04-25_abc123_features.parquet")
    df = fr.df   # game_time_s-indexed DataFrame of floats
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import entropy as shannon_entropy

logger = logging.getLogger(__name__)

_FEATURE_META_KEY = b"feature_metadata"

ROLLING_WINDOWS = (1, 5, 30)   # seconds; equals samples at 1 Hz

_VERTICAL_VZ_THRESHOLD = 0.5   # units/tic; flags jumping or falling
_HEALTH_PICKUP_MIN_DELTA = 5.0  # health units; threshold for pickup detection
_ENTROPY_BINS = 8               # histogram bins per axis for 2D positional entropy
_DISPLACEMENT_WINDOW = 30       # seconds used for displacement_efficiency


# ── Metadata ──────────────────────────────────────────────────────────────────

@dataclass
class FeatureMetadata:
    session_id: str
    source_path: str
    created_utc: str
    windows_s: List[int]
    features_computed: List[str]
    features_skipped: List[str]
    sample_interval_s: float
    num_samples: int
    num_features: int

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeatureMetadata":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


# ── Result container ──────────────────────────────────────────────────────────

class FeatureResult:
    """Feature DataFrame and associated metadata; mirrors SessionResult."""

    def __init__(self, df: pd.DataFrame, metadata: FeatureMetadata) -> None:
        self.df = df
        self.metadata = metadata

    def save(self, directory: str | Path) -> Path:
        """Write <date>_<session_id>_features.parquet to `directory/features/`."""
        out_dir = Path(directory) / "features"
        out_dir.mkdir(parents=True, exist_ok=True)

        date_str = self.metadata.created_utc[:10]
        stem = f"{date_str}_{self.metadata.session_id}_features"
        path = out_dir / f"{stem}.parquet"

        table = pa.Table.from_pandas(self.df, preserve_index=True)
        schema_meta = {
            **(table.schema.metadata or {}),
            _FEATURE_META_KEY: json.dumps(self.metadata.to_dict()).encode(),
        }
        table = table.replace_schema_metadata(schema_meta)
        pq.write_table(table, path, compression="snappy")

        logger.info("Saved %d features × %d samples → %s",
                    self.metadata.num_features, self.metadata.num_samples, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FeatureResult":
        """Load a previously saved feature Parquet file."""
        path = Path(path)
        table = pq.read_table(path)
        df = table.to_pandas()

        metadata: Optional[FeatureMetadata] = None
        raw_meta = (table.schema.metadata or {}).get(_FEATURE_META_KEY)
        if raw_meta:
            metadata = FeatureMetadata.from_dict(json.loads(raw_meta))

        return cls(df=df, metadata=metadata)

    def __repr__(self) -> str:
        sid = self.metadata.session_id if self.metadata else "?"
        n = self.metadata.num_features if self.metadata else len(self.df.columns)
        return f"FeatureResult(session_id={sid!r}, features={n}, samples={len(self.df)})"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_columns(df: pd.DataFrame, cols: List[str], feature_name: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning("Skipping %s: missing columns %s", feature_name, missing)
        return False
    return True


def _safe_div(a: pd.Series, b: pd.Series, fill: float = 0.0) -> pd.Series:
    return a.div(b.replace(0, np.nan)).fillna(fill)


# ── Rate / delta features ─────────────────────────────────────────────────────

def _compute_rate_features(
    df: pd.DataFrame, dt: float, skipped: List[str]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}

    if _check_columns(df, ["killcount"], "kill_rate"):
        out["kill_rate"] = df["killcount"].diff().fillna(0.0) / dt
    else:
        skipped.append("kill_rate")

    if _check_columns(df, ["damagecount", "damage_taken"], "damage_efficiency"):
        d_dmg = df["damagecount"].diff().fillna(0.0)
        d_tak = df["damage_taken"].diff().fillna(0.0)
        out["damage_efficiency"] = _safe_div(d_dmg, d_tak)
    else:
        skipped.append("damage_efficiency")

    if _check_columns(df, ["itemcount"], "item_acquisition_rate"):
        out["item_acquisition_rate"] = df["itemcount"].diff().fillna(0.0) / dt
    else:
        skipped.append("item_acquisition_rate")

    if _check_columns(df, ["health"], "health_velocity"):
        out["health_velocity"] = df["health"].diff().fillna(0.0) / dt
    else:
        skipped.append("health_velocity")

    return out


# ── Kinematic features ────────────────────────────────────────────────────────

def _displacement_efficiency_loop(
    x: np.ndarray, y: np.ndarray, window: int
) -> np.ndarray:
    result = np.full(len(x), np.nan)
    for i in range(window - 1, len(x)):
        cx = x[i - window + 1:i + 1]
        cy = y[i - window + 1:i + 1]
        dx, dy = cx[-1] - cx[0], cy[-1] - cy[0]
        euclidean = np.sqrt(dx * dx + dy * dy)
        dists = np.sqrt(np.diff(cx) ** 2 + np.diff(cy) ** 2)
        path_len = dists.sum()
        result[i] = euclidean / path_len if path_len > 0 else 0.0
    return result


def _compute_kinematic_features(
    df: pd.DataFrame, dt: float, skipped: List[str]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}

    if _check_columns(df, ["velocity_x", "velocity_y"], "speed_2d"):
        out["speed_2d"] = np.sqrt(df["velocity_x"] ** 2 + df["velocity_y"] ** 2)
    else:
        skipped.append("speed_2d")

    if _check_columns(df, ["angle"], "heading_change_rate"):
        raw = df["angle"].diff()
        wrapped = ((raw + 180) % 360) - 180
        hcr = wrapped.abs() / dt
        hcr.iloc[0] = 0.0
        out["heading_change_rate"] = hcr
    else:
        skipped.append("heading_change_rate")

    if _check_columns(df, ["position_x", "position_y"], "displacement_efficiency"):
        out["displacement_efficiency"] = pd.Series(
            _displacement_efficiency_loop(
                df["position_x"].to_numpy(),
                df["position_y"].to_numpy(),
                _DISPLACEMENT_WINDOW,
            ),
            index=df.index,
        )
    else:
        skipped.append("displacement_efficiency")

    if _check_columns(df, ["velocity_z"], "vertical_exposure"):
        out["vertical_exposure"] = (
            df["velocity_z"].abs() > _VERTICAL_VZ_THRESHOLD
        ).astype(float)
    else:
        skipped.append("vertical_exposure")

    return out


# ── Positional entropy ────────────────────────────────────────────────────────

def _positional_entropy_loop(
    x: np.ndarray, y: np.ndarray, window: int, bins: int
) -> np.ndarray:
    result = np.full(len(x), np.nan)
    for i in range(window - 1, len(x)):
        cx = x[i - window + 1:i + 1]
        cy = y[i - window + 1:i + 1]
        H, _, _ = np.histogram2d(cx, cy, bins=bins)
        p = H.ravel()
        total = p.sum()
        if total == 0:
            continue
        p = p[p > 0] / total
        result[i] = shannon_entropy(p)
    return result


def _compute_positional_entropy_features(
    df: pd.DataFrame, skipped: List[str]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}

    if not _check_columns(df, ["position_x", "position_y"], "positional_entropy"):
        skipped.extend(["positional_entropy_5s", "positional_entropy_30s"])
        return out

    x = df["position_x"].to_numpy()
    y = df["position_y"].to_numpy()

    for w in (5, 30):
        out[f"positional_entropy_{w}s"] = pd.Series(
            _positional_entropy_loop(x, y, w, _ENTROPY_BINS),
            index=df.index,
        )

    return out


# ── Event-based features ──────────────────────────────────────────────────────

def _iki_loop(
    index_arr: np.ndarray, kill_events: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(index_arr)
    iki_mean = np.full(n, np.nan)
    iki_cv = np.full(n, np.nan)
    for i in range(window - 1, n):
        ev_times = index_arr[i - window + 1:i + 1][kill_events[i - window + 1:i + 1] > 0]
        if len(ev_times) < 2:
            continue
        intervals = np.diff(ev_times)
        mean = intervals.mean()
        iki_mean[i] = mean
        iki_cv[i] = intervals.std() / mean if mean > 0 else 0.0
    return iki_mean, iki_cv


def _compute_event_features(
    df: pd.DataFrame, dt: float, window: int, skipped: List[str]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}

    if _check_columns(df, ["killcount"], "inter_kill_interval"):
        kill_event = (df["killcount"].diff().fillna(0) > 0).astype(float).to_numpy()
        iki_mean, iki_cv = _iki_loop(df.index.to_numpy(), kill_event, window)
        out["inter_kill_interval_mean"] = pd.Series(iki_mean, index=df.index)
        out["inter_kill_interval_cv"] = pd.Series(iki_cv, index=df.index)
    else:
        skipped.extend(["inter_kill_interval_mean", "inter_kill_interval_cv"])

    if _check_columns(df, ["hits_taken"], "hit_taken_event_rate"):
        hit_event = (df["hits_taken"].diff().fillna(0) > 0).astype(float)
        out["hit_taken_event_rate"] = hit_event.rolling(window, min_periods=1).mean()
    else:
        skipped.append("hit_taken_event_rate")

    if _check_columns(df, ["selected_weapon"], "weapon_switch_frequency"):
        switch = (df["selected_weapon"].diff().fillna(0) != 0).astype(float)
        switch.iloc[0] = 0.0
        out["weapon_switch_frequency"] = switch.rolling(window, min_periods=1).mean()
    else:
        skipped.append("weapon_switch_frequency")

    if _check_columns(df, ["health", "dead"], "health_pickup_count"):
        prev_dead = df["dead"].shift(1).fillna(0)
        delta_health = df["health"].diff().fillna(0)
        pickup = (
            (delta_health > _HEALTH_PICKUP_MIN_DELTA)
            & (df["dead"] == 0)
            & (prev_dead == 0)
        ).astype(float)
        out["health_pickup_count"] = pickup.rolling(window, min_periods=1).sum()
    else:
        skipped.append("health_pickup_count")

    return out


# ── Derived ratio features ────────────────────────────────────────────────────

def _compute_ratio_features(
    df: pd.DataFrame, skipped: List[str]
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}

    if _check_columns(df, ["damagecount", "killcount"], "damage_per_kill"):
        d_dmg = df["damagecount"].diff().fillna(0.0)
        d_kill = df["killcount"].diff().fillna(0.0)
        out["damage_per_kill"] = _safe_div(d_dmg, d_kill)
    else:
        skipped.append("damage_per_kill")

    ammo_cols = [f"ammo{i}" for i in range(10) if f"ammo{i}" in df.columns]
    if ammo_cols and _check_columns(df, ["hitcount"], "hit_rate"):
        consumed = (
            df[ammo_cols].diff().clip(upper=0).abs().sum(axis=1).fillna(0)
        )
        d_hit = df["hitcount"].diff().fillna(0.0)
        out["hit_rate"] = _safe_div(d_hit, consumed)
    elif not ammo_cols:
        logger.warning("Skipping hit_rate: no ammo columns found")
        skipped.append("hit_rate")
    else:
        skipped.append("hit_rate")

    if _check_columns(df, ["secretcount", "position_x", "position_y"],
                      "secret_discovery_rate"):
        dx = df["position_x"].diff().fillna(0)
        dy = df["position_y"].diff().fillna(0)
        step_dist = np.sqrt(dx ** 2 + dy ** 2)
        d_secret = df["secretcount"].diff().fillna(0.0)
        out["secret_discovery_rate"] = _safe_div(d_secret, step_dist)
    else:
        skipped.append("secret_discovery_rate")

    return out


# ── Rolling statistical features ──────────────────────────────────────────────

def _compute_rolling_stats(
    df: pd.DataFrame,
    computed: Dict[str, pd.Series],
    windows: Sequence[int],
    skipped: List[str],
) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}

    base_signals = {
        "health": df.get("health"),
        "armor": df.get("armor"),
        "speed_2d": computed.get("speed_2d"),
        "heading_change_rate": computed.get("heading_change_rate"),
    }

    for sig_name, sig in base_signals.items():
        if sig is None:
            skipped.extend([
                f"{sig_name}_{stat}_{w}s"
                for stat in ("mean", "std", "min", "max")
                for w in windows
            ])
            continue

        for w in windows:
            suffix = f"{w}s"
            roll = sig.rolling(w, min_periods=1)
            mean = roll.mean()
            std = roll.std().fillna(0.0)
            out[f"{sig_name}_mean_{suffix}"] = mean
            out[f"{sig_name}_std_{suffix}"] = std
            out[f"{sig_name}_min_{suffix}"] = roll.min()
            out[f"{sig_name}_max_{suffix}"] = roll.max()
            if w > 1:
                out[f"{sig_name}_cv_{suffix}"] = _safe_div(std, mean)
            if w >= 5:
                autocorr = sig.rolling(w, min_periods=3).apply(
                    lambda x: pd.Series(x).autocorr(lag=1),
                    raw=False,
                ).fillna(0.0)
                out[f"{sig_name}_autocorr_{suffix}"] = autocorr

    kill_rate = computed.get("kill_rate")
    if kill_rate is not None:
        for w in windows:
            if w > 1:
                roll = kill_rate.rolling(w, min_periods=1)
                std = roll.std().fillna(0.0)
                mean = roll.mean()
                out[f"kill_rate_cv_{w}s"] = _safe_div(std, mean)
    else:
        skipped.extend([f"kill_rate_cv_{w}s" for w in windows if w > 1])

    return out


# ── Public API ────────────────────────────────────────────────────────────────

def extract_features(
    result: Any,
    windows: Sequence[int] = ROLLING_WINDOWS,
    source_path: str = "",
) -> "FeatureResult":
    """Compute all behavioral and kinematic features from a SessionResult.

    Features requiring unavailable columns are skipped with a logged warning.
    The returned DataFrame shares the same game_time_s index as the source.

    Parameters
    ----------
    result:
        A SessionResult loaded from a session Parquet file.
    windows:
        Rolling window sizes in seconds. Default: (1, 5, 30).
    source_path:
        Optional path to the source Parquet file (stored in metadata).
    """
    df = result.df
    meta = result.metadata
    dt = meta.sample_interval_tics / meta.tic_rate

    out: Dict[str, pd.Series] = {}
    skipped: List[str] = []

    out.update(_compute_rate_features(df, dt, skipped))
    out.update(_compute_kinematic_features(df, dt, skipped))
    out.update(_compute_positional_entropy_features(df, skipped))
    out.update(_compute_event_features(df, dt, window=30, skipped=skipped))
    out.update(_compute_ratio_features(df, skipped))
    out.update(_compute_rolling_stats(df, out, windows, skipped))

    feature_df = pd.DataFrame(out, index=df.index).astype("float64")

    feature_meta = FeatureMetadata(
        session_id=meta.session_id,
        source_path=source_path,
        created_utc=datetime.now(timezone.utc).isoformat(),
        windows_s=list(windows),
        features_computed=list(feature_df.columns),
        features_skipped=skipped,
        sample_interval_s=dt,
        num_samples=len(feature_df),
        num_features=len(feature_df.columns),
    )
    return FeatureResult(df=feature_df, metadata=feature_meta)
