"""Shared numerical preprocessing used by the research notebooks."""

from __future__ import annotations

import numpy as np


def interpolate_timeseries(data: np.ndarray, new_length: int) -> np.ndarray:
    """Linearly resample a one-dimensional signal to ``new_length`` points."""

    values = np.asarray(data, dtype=float).reshape(-1)
    if new_length < 1:
        raise ValueError("new_length must be positive")
    if len(values) == 0:
        raise ValueError("data must contain at least one value")
    if len(values) == 1:
        return np.full(new_length, values[0], dtype=float)
    source_position = np.arange(len(values), dtype=float)
    target_position = np.linspace(0, len(values) - 1, new_length)
    return np.interp(target_position, source_position, values)

