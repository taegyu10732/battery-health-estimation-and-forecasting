"""Leakage-conscious feature tables for the curated notebook baselines."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from battery_soh.data import CellSummary

EARLY_FEATURE_COLUMNS = (
    "qd_first",
    "qd_last",
    "qd_mean",
    "qd_std",
    "qd_slope",
    "qd_retention",
    "log_qd_variance",
    "ir_last",
    "ir_change",
    "temperature_mean",
    "temperature_max",
    "charge_time_mean",
)

SOH_FEATURE_COLUMNS = (
    "cycle",
    "charge_capacity",
    "internal_resistance",
    "temperature_avg",
    "temperature_min",
    "temperature_max",
    "charge_time",
)


def _finite(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _first(values: np.ndarray) -> float:
    array = _finite(values)
    return float(array[0]) if len(array) else np.nan


def _last(values: np.ndarray) -> float:
    array = _finite(values)
    return float(array[-1]) if len(array) else np.nan


def _mean(values: np.ndarray) -> float:
    array = _finite(values)
    return float(np.mean(array)) if len(array) else np.nan


def _maximum(values: np.ndarray) -> float:
    array = _finite(values)
    return float(np.max(array)) if len(array) else np.nan


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2 or np.ptp(x[valid]) == 0:
        return np.nan
    return float(np.polyfit(x[valid], y[valid], deg=1)[0])


def build_early_life_features(
    cells: Iterable[CellSummary], observation_cycle: int = 100
) -> pd.DataFrame:
    """Build one fixed-horizon feature row per eligible cell.

    Only samples at or before ``observation_cycle`` are used. Recorded cycle life and RUL
    are returned as targets/metadata and are never part of ``EARLY_FEATURE_COLUMNS``.
    """

    if observation_cycle < 2:
        raise ValueError("observation_cycle must be at least 2")

    rows: list[dict[str, object]] = []
    for cell in cells:
        frame = cell.to_frame()
        window = frame.loc[(frame["cycle"] >= 1) & (frame["cycle"] <= observation_cycle)].copy()
        if cell.cycle_life <= observation_cycle or len(window) < 2:
            continue

        cycle = window["cycle"].to_numpy(dtype=float)
        qd = window["discharge_capacity"].to_numpy(dtype=float)
        ir = window["internal_resistance"].to_numpy(dtype=float)
        qd_valid = _finite(qd)
        qd_first = _first(qd)
        qd_last = _last(qd)
        qd_variance = float(np.var(qd_valid)) if len(qd_valid) else np.nan
        rows.append(
            {
                "cell_id": cell.cell_id,
                "batch": cell.batch,
                "charge_policy": cell.charge_policy,
                "observation_cycle": observation_cycle,
                "cycle_life": cell.cycle_life,
                "rul": cell.cycle_life - observation_cycle,
                "qd_first": qd_first,
                "qd_last": qd_last,
                "qd_mean": _mean(qd),
                "qd_std": float(np.std(qd_valid)) if len(qd_valid) else np.nan,
                "qd_slope": _slope(cycle, qd),
                "qd_retention": (
                    qd_last / qd_first if np.isfinite(qd_first) and qd_first else np.nan
                ),
                "log_qd_variance": np.log10(max(qd_variance, np.finfo(float).tiny)),
                "ir_last": _last(ir),
                "ir_change": _last(ir) - _first(ir),
                "temperature_mean": _mean(window["temperature_avg"].to_numpy()),
                "temperature_max": _maximum(window["temperature_max"].to_numpy()),
                "charge_time_mean": _mean(window["charge_time"].to_numpy()),
            }
        )
    return pd.DataFrame(rows)


def build_soh_samples(
    cells: Iterable[CellSummary],
    nominal_capacity_ah: float = 1.1,
    sample_every: int = 10,
    minimum_cycle: int = 1,
) -> pd.DataFrame:
    """Build a cycle-level SOH table with identifiers kept separate from features."""

    if nominal_capacity_ah <= 0:
        raise ValueError("nominal_capacity_ah must be positive")
    if sample_every < 1:
        raise ValueError("sample_every must be at least 1")

    frames: list[pd.DataFrame] = []
    for cell in cells:
        frame = cell.to_frame()
        frame = frame.loc[
            (frame["cycle"] >= minimum_cycle)
            & (frame["cycle"] <= cell.cycle_life)
            & np.isfinite(frame["discharge_capacity"])
            & (frame["discharge_capacity"] > 0)
        ].iloc[::sample_every]
        if frame.empty:
            continue
        frame = frame.copy()
        frame["soh"] = frame["discharge_capacity"] / nominal_capacity_ah
        frames.append(frame[["cell_id", "batch", *SOH_FEATURE_COLUMNS, "soh"]])

    if not frames:
        return pd.DataFrame(columns=["cell_id", "batch", *SOH_FEATURE_COLUMNS, "soh"])
    return pd.concat(frames, ignore_index=True)
