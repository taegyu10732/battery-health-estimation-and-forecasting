# Battery Health Forecasting

Clean, reproducible SOH, RUL, and degradation-forecasting research on the MIT–Stanford
fast-charging battery dataset.

This repository is a cleaned, publication-ready view of exploratory work on the public
MIT/Stanford fast-charging battery dataset associated with Severson et al. It focuses on
three questions:

- How do cell trajectories and operating summaries differ across batteries?
- Can state of health (SOH) be estimated without mixing cycles from the same cell across
  train and test sets?
- How well can remaining useful life (RUL) be estimated from a fixed early-life window?

The original workspace contained many branches of these ideas: raw-curve and engineered
feature clustering, MLP/CNN/LSTM/sequence-to-sequence models, multi-cycle inputs,
operating-condition features, knee-point experiments, variational autoencoders, and
physics-informed or parameter-constrained losses. Repeated branches are consolidated by
task and exposed as parameters instead of separate near-duplicate scripts. The public
organization is documented in [docs/CONSOLIDATION.md](docs/CONSOLIDATION.md), and the full
research scope is summarized in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

The release contains **4 curated workflow scripts plus 35 consolidated research
scripts**, covering 11 architecture families and all major methodology branches found
in the 106 private source notebooks. See the
[architecture catalog](docs/ARCHITECTURES.md) and
[research script index](research_scripts/README.md).

This public version intentionally contains **no generated output, raw data, trained
weights, result tables, or hard-coded GPU selection**. The workflow scripts are compact
CPU baselines designed to establish a reproducible starting point; they do not claim to
reproduce every legacy experiment or the paper's reported benchmark.

## Repository layout

```text
.
├── workflows/
│   ├── 01_dataset_overview.py
│   ├── 02_early_life_clustering.py
│   ├── 03_soh_estimation.py
│   └── 04_rul_estimation.py
├── research_scripts/         # 35 architecture/method representatives
│   ├── clustering/
│   ├── soh/
│   ├── rul/
│   ├── sequence/
│   ├── physics/
│   └── method_development/
├── src/battery_soh/          # reusable loading, feature, split, and metric code
├── data/README.md            # data placement and environment-variable instructions
├── docs/                     # experiment map, architecture catalog, and data notes
├── scripts/                  # release audit tool
├── tests/                    # lightweight unit tests
└── artifacts/                # local outputs; ignored by Git
```

## Quick start

Python 3.10 or newer is required. No GPU is required.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
$env:BATTERY_DATA_DIR = (Resolve-Path "..\Data")
python workflows\01_dataset_overview.py
```

### macOS or Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
export BATTERY_DATA_DIR=/absolute/path/to/the/mat/files
python workflows/01_dataset_overview.py
```

Run the workflow scripts in numeric order. Their `# %%` markers provide optional cell-by-cell
execution in VS Code and Spyder. The loader reads only compact per-cycle summary arrays; it
does not load the entire multi-gigabyte raw time-series collection into memory.

To inspect or rerun the original PyTorch/Optuna research implementations, install the
optional research dependencies:

```bash
python -m pip install -e ".[research]"
```

## Data

Download the dataset from the authors' data page:
[data.matr.io/1](https://data.matr.io/1/). The work is associated with:

> Severson, K. A. et al. *Data-driven prediction of battery cycle life before capacity
> degradation*. Nature Energy 4, 383–391 (2019).
> [https://doi.org/10.1038/s41560-019-0356-8](https://doi.org/10.1038/s41560-019-0356-8)

The loader expects these three files:

```text
2017-05-12_batchdata_updated_struct_errorcorrect.mat
2017-06-30_batchdata_updated_struct_errorcorrect.mat
2018-04-12_batchdata_updated_struct_errorcorrect.mat
```

Set `BATTERY_DATA_DIR` to the directory containing them, or place them under `data/`.
Raw data is deliberately excluded from Git. See [docs/DATA.md](docs/DATA.md) for details.
By default, the curated loader applies the continuation and exclusion rules from the
authors' loading notebook and returns the corrected 124-cell cohort. Pass
`cohort="all_valid"` only when intentionally inspecting all finite-lifetime raw records.

## Workflows

| Script | Purpose | Split discipline |
| --- | --- | --- |
| [`01_dataset_overview.py`](workflows/01_dataset_overview.py) | Load cell summaries, inspect lifetimes, capacity fade, temperature, and internal resistance | Descriptive only |
| [`02_early_life_clustering.py`](workflows/02_early_life_clustering.py) | Cluster cells using features available through a fixed observation cycle | Unsupervised; lifetime is used only after clustering for interpretation |
| [`03_soh_estimation.py`](workflows/03_soh_estimation.py) | Estimate cycle-level SOH with a compact tree-based baseline | Group split by cell; no cell appears in both train and test |
| [`04_rul_estimation.py`](workflows/04_rul_estimation.py) | Estimate RUL at a fixed early-life observation cycle | Batch 1–2 train, batch 3 test |

Change the configuration block near the top of a script to run a former variant (for
example, observation horizon, feature ablation, cluster count, or held-out batch) without
copying the entire script.

The feature code keeps identifiers and targets separate from model inputs. In particular,
cycle life is never used as an SOH feature, and test batteries are not used to fit imputers,
scalers, or models.

## Definitions used here

- **SOH**: discharge capacity divided by the nominal capacity, which defaults to 1.1 Ah.
- **RUL**: recorded cycle life minus the fixed observation cycle.
- **Early-life features**: statistics computed only from samples at or before the selected
  observation cycle.

These definitions are explicit and configurable in `src/battery_soh/features.py`. They
should be revisited when comparing against work that uses a different end-of-life
threshold, nominal capacity, cycle indexing convention, or corrected cell cohort.

## Quality checks

```bash
python -m pip install -e ".[dev]"
pytest
ruff check src tests scripts workflows
python scripts/audit_release.py
```

The audit compiles all 39 public scripts and fails on leftover notebooks, hard-coded GPU
selection, shell GPU probes, absolute local paths, large files, or model/data artifacts.

## Scope and limitations

- The dataset is small at the cell level. Metrics can vary substantially with the split.
- Summary features are convenient baselines, not a substitute for careful raw-curve
  processing or electrochemical validation.
- The SOH workflow includes charge capacity as a proxy feature. For an online estimator,
  replace it with signals actually available at inference time.
- The deep-learning and physics-informed branches are included as cleaned research
  scripts, but they are historical implementations rather than verified benchmarks.
- No license is selected for this code yet. Choose an appropriate code license and verify
  the dataset's terms before reuse or redistribution.

## Citation

If you use the dataset, cite the original publication:

```bibtex
@article{severson2019data,
  title   = {Data-driven prediction of battery cycle life before capacity degradation},
  author  = {Severson, Kristen A. and Attia, Peter M. and Jin, Norman and others},
  journal = {Nature Energy},
  volume  = {4},
  pages   = {383--391},
  year    = {2019},
  doi     = {10.1038/s41560-019-0356-8}
}
```
