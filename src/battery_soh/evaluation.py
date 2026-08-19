"""Evaluation helpers shared by the notebooks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return common regression metrics with a zero-safe percentage error."""

    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    denominator = np.maximum(np.abs(actual), np.finfo(float).eps)
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAPE_percent": float(np.mean(np.abs((actual - predicted) / denominator)) * 100),
        "R2": float(r2_score(actual, predicted)),
    }


def MAPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Backward-compatible percentage MAPE used by the research notebooks."""

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    denominator = np.maximum(np.abs(actual), np.finfo(float).eps)
    return float(np.mean(np.abs((actual - predicted) / denominator)) * 100)


def RMSPE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Backward-compatible root mean square percentage error as a fraction."""

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    denominator = np.maximum(np.abs(actual), np.finfo(float).eps)
    return float(np.sqrt(np.mean(np.square((predicted - actual) / denominator))))


def grouped_train_test_indices(
    groups: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create one split in which group identifiers cannot cross the boundary."""

    group_array = np.asarray(groups)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    placeholder = np.zeros(len(group_array))
    train_index, test_index = next(splitter.split(placeholder, groups=group_array))
    return train_index, test_index
