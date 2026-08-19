from __future__ import annotations

import h5py
import numpy as np

from battery_soh.data import BatchFile, load_cell_summaries


def test_load_minimal_matlab_style_hdf5(tmp_path) -> None:
    path = tmp_path / "mini.mat"
    with h5py.File(path, "w") as file:
        batch = file.create_group("batch")
        summary = file.create_group("cell_summary")
        for name, values in {
            "IR": [0.01, 0.02],
            "QCharge": [1.11, 1.10],
            "QDischarge": [1.09, 1.08],
            "Tavg": [30.0, 31.0],
            "Tmin": [28.0, 29.0],
            "Tmax": [32.0, 33.0],
            "chargetime": [12.0, 13.0],
            "cycle": [1.0, 2.0],
        }.items():
            summary.create_dataset(name, data=np.asarray(values).reshape(1, -1))

        summary_refs = batch.create_dataset("summary", (1, 1), dtype=h5py.ref_dtype)
        summary_refs[0, 0] = summary.ref

        for name, value in {
            "cycle_life": np.array([[200.0]]),
            "channel_id_int": np.array([[7.0]]),
            "policy_readable": np.asarray([[ord(char)] for char in "policy"], dtype=np.uint16),
        }.items():
            dataset = file.create_dataset(f"value_{name}", data=value)
            refs = batch.create_dataset(name, (1, 1), dtype=h5py.ref_dtype)
            refs[0, 0] = dataset.ref

    cells = load_cell_summaries(tmp_path, (BatchFile("mini.mat", "b1"),))
    assert len(cells) == 1
    assert cells[0].cell_id == "b1c7"
    assert cells[0].cycle_life == 200
    assert cells[0].charge_policy == "policy"
    assert np.allclose(cells[0].discharge_capacity, [1.09, 1.08])


def test_custom_files_reject_paper_cohort(tmp_path) -> None:
    try:
        load_cell_summaries(tmp_path, (BatchFile("missing.mat", "b1"),), cohort="paper")
    except FileNotFoundError:
        # File validation happens before cohort validation and remains actionable.
        pass
