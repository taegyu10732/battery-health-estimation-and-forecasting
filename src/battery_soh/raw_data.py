"""Legacy-compatible raw-cycle loading for the consolidated research notebooks.

The curated notebooks should prefer :func:`battery_soh.data.load_cell_summaries`, which is
far more memory efficient. This module exists for architectures that consume raw within-cycle
current, capacity, voltage, temperature, and time traces.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np

from battery_soh.data import (
    DEFAULT_BATCH_FILES,
    BatchFile,
    _read_scalar_reference,
    _read_summary_vector,
    _read_text_reference,
    resolve_data_dir,
)

RAW_CYCLE_FIELDS = {
    "I": "I",
    "Qc": "Qc",
    "Qd": "Qd",
    "Qdlin": "Qdlin",
    "T": "T",
    "Tdlin": "Tdlin",
    "V": "V",
    "dQdV": "discharge_dQdV",
    "t": "t",
}

LEGACY_SUMMARY_FIELDS = {
    "IR": "IR",
    "QC": "QCharge",
    "QD": "QDischarge",
    "Tavg": "Tavg",
    "Tmin": "Tmin",
    "Tmax": "Tmax",
    "chargetime": "chargetime",
    "cycle": "cycle",
}


def _read_raw_vector(file: h5py.File, dataset: h5py.Dataset, row: int) -> np.ndarray:
    reference = dataset[row, 0]
    return np.asarray(file[reference][()]).reshape(-1)


def _load_raw_batch(path: Path, prefix: str, include_cycles: bool) -> dict[str, dict[str, object]]:
    batteries: dict[str, dict[str, object]] = {}
    with h5py.File(path, "r") as file:
        batch = file["batch"]
        for row in range(batch["summary"].shape[0]):
            channel_id = int(_read_scalar_reference(file, batch["channel_id_int"], row))
            summary_group = file[batch["summary"][row, 0]]
            summary = {
                public: _read_summary_vector(summary_group, source)
                for public, source in LEGACY_SUMMARY_FIELDS.items()
            }
            cycles: dict[str, dict[str, np.ndarray]] = {}
            if include_cycles:
                cycle_group = file[batch["cycles"][row, 0]]
                for cycle_index in range(cycle_group["I"].shape[0]):
                    cycles[str(cycle_index)] = {
                        public: _read_raw_vector(file, cycle_group[source], cycle_index)
                        for public, source in RAW_CYCLE_FIELDS.items()
                    }
            batteries[f"{prefix}c{channel_id}"] = {
                # Preserve the array shape used by the historical research code.
                "cycle_life": np.asarray(file[batch["cycle_life"][row, 0]][()]),
                "charge_policy": _read_text_reference(file, batch["policy_readable"], row),
                "summary": summary,
                "cycles": cycles,
            }
    return batteries


def load_battery_dictionary(
    data_dir: str | Path | None = None,
    *,
    batches: Iterable[str] = ("b1", "b2", "b3"),
    include_cycles: bool = True,
) -> dict[str, dict[str, object]]:
    """Load a legacy-compatible nested dictionary once for a research notebook.

    Loading raw cycles can require many gigabytes of memory. Use ``include_cycles=False``
    for descriptor/summary-only work, or use ``load_cell_summaries`` in new code.
    """

    requested = set(batches)
    unknown = requested.difference(item.prefix for item in DEFAULT_BATCH_FILES)
    if unknown:
        raise ValueError(f"Unknown batch labels: {', '.join(sorted(unknown))}")
    directory = resolve_data_dir(data_dir)
    selected: tuple[BatchFile, ...] = tuple(
        item for item in DEFAULT_BATCH_FILES if item.prefix in requested
    )
    batteries: dict[str, dict[str, object]] = {}
    for item in selected:
        batteries.update(_load_raw_batch(directory / item.filename, item.prefix, include_cycles))
    return batteries
