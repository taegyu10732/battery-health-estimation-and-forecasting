from __future__ import annotations

import numpy as np

from battery_soh.data import CellSummary
from battery_soh.evaluation import grouped_train_test_indices, regression_metrics
from battery_soh.features import (
    EARLY_FEATURE_COLUMNS,
    SOH_FEATURE_COLUMNS,
    build_early_life_features,
    build_soh_samples,
)


def make_cell(cell_id: str = "b1c1", life: int = 200) -> CellSummary:
    cycle = np.arange(1, 121, dtype=float)
    capacity = 1.1 - 0.001 * cycle
    return CellSummary(
        cell_id=cell_id,
        batch=cell_id[:2],
        channel_id=int(cell_id.split("c")[-1]),
        cycle_life=life,
        charge_policy="test-policy",
        internal_resistance=0.01 + cycle * 1e-5,
        charge_capacity=capacity + 0.01,
        discharge_capacity=capacity,
        temperature_avg=np.full_like(cycle, 30.0),
        temperature_min=np.full_like(cycle, 28.0),
        temperature_max=np.full_like(cycle, 33.0),
        charge_time=np.full_like(cycle, 12.0),
        cycle=cycle,
    )


def test_early_features_use_fixed_horizon() -> None:
    table = build_early_life_features([make_cell()], observation_cycle=100)
    assert len(table) == 1
    assert table.loc[0, "rul"] == 100
    assert table.loc[0, "qd_last"] == 1.0
    assert not {"cycle_life", "rul"}.intersection(EARLY_FEATURE_COLUMNS)


def test_soh_samples_have_expected_target_and_no_discharge_capacity_feature() -> None:
    table = build_soh_samples([make_cell()], sample_every=20)
    assert np.isclose(table.loc[0, "soh"], 1.099 / 1.1)
    assert "discharge_capacity" not in SOH_FEATURE_COLUMNS


def test_group_split_is_disjoint() -> None:
    groups = np.repeat(["a", "b", "c", "d"], 5)
    train, test = grouped_train_test_indices(groups, test_size=0.5)
    assert set(groups[train]).isdisjoint(groups[test])


def test_regression_metrics() -> None:
    metrics = regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert metrics["RMSE"] == 0.0
    assert metrics["MAPE_percent"] == 0.0

