"""Memory-conscious access to summary arrays in the Severson battery dataset."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BatchFile:
    """One source file and the prefix used for its cell identifiers."""

    filename: str
    prefix: str


DEFAULT_BATCH_FILES = (
    BatchFile("2017-05-12_batchdata_updated_struct_errorcorrect.mat", "b1"),
    BatchFile("2017-06-30_batchdata_updated_struct_errorcorrect.mat", "b2"),
    BatchFile("2018-04-12_batchdata_updated_struct_errorcorrect.mat", "b3"),
)

SUMMARY_FIELDS = {
    "internal_resistance": "IR",
    "charge_capacity": "QCharge",
    "discharge_capacity": "QDischarge",
    "temperature_avg": "Tavg",
    "temperature_min": "Tmin",
    "temperature_max": "Tmax",
    "charge_time": "chargetime",
    "cycle": "cycle",
}


@dataclass(frozen=True)
class CellSummary:
    """Compact per-cycle data for one cell.

    Arrays are copied out of the HDF5 file, so the file can be closed immediately after
    loading a batch.
    """

    cell_id: str
    batch: str
    channel_id: int
    cycle_life: int
    charge_policy: str
    internal_resistance: np.ndarray
    charge_capacity: np.ndarray
    discharge_capacity: np.ndarray
    temperature_avg: np.ndarray
    temperature_min: np.ndarray
    temperature_max: np.ndarray
    charge_time: np.ndarray
    cycle: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        """Return aligned per-cycle summary values as a tidy table."""

        arrays = {name: np.asarray(getattr(self, name)).reshape(-1) for name in SUMMARY_FIELDS}
        length = min(map(len, arrays.values()))
        frame = pd.DataFrame({name: values[:length] for name, values in arrays.items()})
        frame.insert(0, "cycle_life", self.cycle_life)
        frame.insert(0, "batch", self.batch)
        frame.insert(0, "cell_id", self.cell_id)
        return frame


def _missing_files(data_dir: Path, batch_files: Iterable[BatchFile]) -> list[str]:
    return [item.filename for item in batch_files if not (data_dir / item.filename).is_file()]


def resolve_data_dir(path: str | Path | None = None) -> Path:
    """Resolve the dataset directory without embedding a machine-specific path.

    Resolution order is: explicit argument, ``BATTERY_DATA_DIR``, then ``data``/``Data``
    directories under the current directory and each of its parents.
    """

    if path is not None:
        candidates = [Path(path).expanduser()]
    elif os.getenv("BATTERY_DATA_DIR"):
        candidates = [Path(os.environ["BATTERY_DATA_DIR"]).expanduser()]
    else:
        current = Path.cwd().resolve()
        candidates = []
        for parent in (current, *current.parents):
            candidates.extend((parent / "data", parent / "Data"))

    checked: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in checked:
            continue
        checked.append(resolved)
        if resolved.is_dir() and not _missing_files(resolved, DEFAULT_BATCH_FILES):
            return resolved

    expected = ", ".join(item.filename for item in DEFAULT_BATCH_FILES)
    locations = ", ".join(str(item) for item in checked) or "<none>"
    raise FileNotFoundError(
        "Battery data directory was not found. Set BATTERY_DATA_DIR or place the files "
        f"under data/. Expected: {expected}. Checked: {locations}"
    )


def _read_reference(file: h5py.File, reference: object) -> np.ndarray:
    if isinstance(reference, h5py.Reference):
        return np.asarray(file[reference][()])
    return np.asarray(reference)


def _read_scalar_reference(file: h5py.File, dataset: h5py.Dataset, row: int) -> object:
    value = dataset[row, 0]
    array = _read_reference(file, value)
    return np.asarray(array).reshape(-1)[0]


def _read_text_reference(file: h5py.File, dataset: h5py.Dataset, row: int) -> str:
    value = dataset[row, 0]
    array = _read_reference(file, value).reshape(-1)
    if array.dtype.kind in {"i", "u"}:
        return "".join(chr(int(code)) for code in array if int(code)).strip()
    if array.dtype.kind == "S":
        return b"".join(array.tolist()).decode("utf-8", errors="replace").strip("\x00")
    return str(array[0]).strip()


def _read_summary_vector(group: h5py.Group, field: str) -> np.ndarray:
    """Flatten either a numeric MATLAB vector or a vector of HDF5 references."""

    dataset = group[field]
    raw = np.asarray(dataset[()])
    if raw.dtype == h5py.ref_dtype or raw.dtype.kind == "O":
        chunks = []
        for reference in raw.reshape(-1):
            if not reference:
                continue
            chunks.append(np.asarray(group.file[reference][()]).reshape(-1))
        if not chunks:
            return np.array([], dtype=float)
        return np.concatenate(chunks).astype(float, copy=False)
    return raw.reshape(-1).astype(float, copy=False)


def _load_batch(path: Path, prefix: str) -> list[CellSummary]:
    cells: list[CellSummary] = []
    with h5py.File(path, "r") as file:
        batch = file["batch"]
        cell_count = int(batch["summary"].shape[0])
        for row in range(cell_count):
            summary = file[batch["summary"][row, 0]]
            values = {
                public_name: _read_summary_vector(summary, source_name)
                for public_name, source_name in SUMMARY_FIELDS.items()
            }
            channel_id = int(_read_scalar_reference(file, batch["channel_id_int"], row))
            cycle_life_value = float(_read_scalar_reference(file, batch["cycle_life"], row))
            cycle_life = int(round(cycle_life_value)) if np.isfinite(cycle_life_value) else -1
            policy = _read_text_reference(file, batch["policy_readable"], row)
            cells.append(
                CellSummary(
                    cell_id=f"{prefix}c{channel_id}",
                    batch=prefix,
                    channel_id=channel_id,
                    cycle_life=cycle_life,
                    charge_policy=policy,
                    **values,
                )
            )
    return cells


def _apply_paper_cohort(cells: list[CellSummary]) -> list[CellSummary]:
    """Apply the authors' 124-cell correction and continuation rules.

    The official loading notebook indexes cells by their row position, whereas this loader
    uses the recorded channel identifier. The identifiers below are the row-index rules
    translated to channel IDs for the three corrected HDF5 files.
    """

    by_id = {cell.cell_id: cell for cell in cells}

    # Five batch-2 cells continue the first five rows of batch 1.
    continuations = (
        ("b1c1", "b2c1", 662),
        ("b1c2", "b2c2", 981),
        ("b1c3", "b2c3", 1060),
        ("b1c5", "b2c5", 208),
        ("b1c6", "b2c6", 482),
    )
    for base_id, continuation_id, added_life in continuations:
        base = by_id[base_id]
        continuation = by_id[continuation_id]
        updates: dict[str, np.ndarray | int] = {"cycle_life": base.cycle_life + added_life}
        for field in SUMMARY_FIELDS:
            base_values = np.asarray(getattr(base, field))
            continuation_values = np.asarray(getattr(continuation, field))
            if field == "cycle":
                continuation_values = continuation_values + len(base_values)
            updates[field] = np.concatenate((base_values, continuation_values))
        by_id[base_id] = replace(base, **updates)

    # Translated from the row-index exclusions in the authors' Load Data notebook.
    excluded = {
        "b1c19",
        "b1c13",
        "b1c21",
        "b1c22",
        "b1c31",
        *(continuation_id for _, continuation_id, _ in continuations),
        "b3c46",
        "b3c12",
        "b3c33",
        "b3c41",
        "b3c6",
        "b3c7",
    }
    result = [by_id[cell.cell_id] for cell in cells if cell.cell_id not in excluded]
    if len(result) != 124:
        raise RuntimeError(f"Expected the corrected 124-cell cohort, found {len(result)} cells")
    return result


def load_cell_summaries(
    data_dir: str | Path | None = None,
    batch_files: Iterable[BatchFile] = DEFAULT_BATCH_FILES,
    cohort: str | None = None,
) -> list[CellSummary]:
    """Load summary arrays from all requested batches.

    The function deliberately ignores the much larger raw cycle traces. Each file is
    closed before the next batch is opened.
    """

    files = tuple(batch_files)
    using_default_files = files == DEFAULT_BATCH_FILES
    directory = (
        resolve_data_dir(data_dir)
        if using_default_files
        else Path(data_dir or ".").expanduser().resolve()
    )
    missing = _missing_files(directory, files)
    if missing:
        raise FileNotFoundError(f"Missing dataset files in {directory}: {', '.join(missing)}")

    cells: list[CellSummary] = []
    for item in files:
        cells.extend(_load_batch(directory / item.filename, item.prefix))

    selected_cohort = cohort or ("paper" if using_default_files else "all_valid")
    if selected_cohort == "paper":
        if not using_default_files:
            raise ValueError("The paper cohort is defined only for the three default batch files")
        return _apply_paper_cohort(cells)
    if selected_cohort == "all_valid":
        return [cell for cell in cells if cell.cycle_life > 0]
    raise ValueError("cohort must be 'paper', 'all_valid', or None")
