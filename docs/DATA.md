# Dataset setup

This project uses the public battery-aging data associated with Severson et al. (2019):

- Dataset page: [https://data.matr.io/1/](https://data.matr.io/1/)
- Paper: [https://doi.org/10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8)
- Authors' loading-code repository:
  [rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation](https://github.com/rdbraatz/data-driven-prediction-of-battery-cycle-life-before-capacity-degradation)

## Required files

```text
2017-05-12_batchdata_updated_struct_errorcorrect.mat
2017-06-30_batchdata_updated_struct_errorcorrect.mat
2018-04-12_batchdata_updated_struct_errorcorrect.mat
```

## Point the project at an existing copy

Avoid duplicating the multi-gigabyte files. Set the environment variable before running a
workflow script.

Windows PowerShell:

```powershell
$env:BATTERY_DATA_DIR = "D:\datasets\severson_battery"
python workflows\01_dataset_overview.py
```

macOS or Linux:

```bash
export BATTERY_DATA_DIR=/datasets/severson_battery
python workflows/01_dataset_overview.py
```

If the variable is not set, the loader searches `data/` under the project root and then
walks upward from the current directory looking for a matching `data/` or `Data/` folder.

## Memory behavior

The raw files contain cycle-level time series and are large. The curated workflow scripts call
`load_cell_summaries`, which extracts only per-cycle summary arrays and closes each HDF5
file before opening the next one. Raw current, voltage, and temperature traces are not
loaded.

## Corrected 124-cell cohort

The three raw files contain 140 rows before corrections. The curated loader follows the
authors' official `Load Data.ipynb`: five batch-2 continuation cells are appended to their
batch-1 cells, five batch-1 cells that do not reach 80% capacity are removed, and six noisy
batch-3 channels are removed. The default result contains 41 + 43 + 40 = 124 cells.

The official notebook names cells by row index; the HDF5 loader in this repository names
them by recorded channel ID. `src/battery_soh/data.py` documents the translated IDs and
asserts the final cohort size so this difference cannot silently change the split.

## Data responsibility

The dataset is not redistributed here. Obtain it from the source, cite the paper, and
follow the terms shown by the data provider. Do not commit raw files, derived private
data, or trained models containing sensitive inputs.
