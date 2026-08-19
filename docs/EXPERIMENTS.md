# Experiment map

The source workspace grew as an exploratory notebook collection. This document turns the
filename history into a research map without publishing stale outputs or presenting every
branch as a verified result.

## 1. Data inspection and normalization

- Visualization of capacity, current, voltage, temperature, and cycle trajectories
- Interpolation of variable-length charge/discharge segments
- Normalization horizons at 90, 120, 180, 240, and 300 cycles
- Input ablations: voltage only, charge only, and no temperature
- Older preprocessing variants retained under `Before_new_norm`

## 2. Battery grouping and early-life features

- Raw-curve clustering
- Clustering with early-life statistics inspired by the Nature Energy dataset workflow
- Early/middle/long-life class prediction
- Changed-data and feature-ablation variants

The cleaned `02_early_life_clustering.ipynb` keeps a modest, label-free K-means baseline.
Lifetime is inspected only after the clusters have been assigned.

## 3. SOH estimation

- MLP and convolutional models over interpolated cycle signals
- Observation-window comparisons and scarce/rare-cell variants
- VAE representations, including cycle-conditioned variants
- Knee-point detection and post-knee prediction
- Surrogate modeling
- Physics-informed / DeepHPM-style objectives and empirical parameter variants

The cleaned `03_soh_estimation.ipynb` is intentionally simpler. Its purpose is to make the
data definition, battery-level split, baseline, and metrics auditable before reintroducing
specialized architectures.

## 4. RUL and cycle-life estimation

- Baseline RUL regression and LSTM variants
- Cycle-gradient and operating-profile features
- Dual-task objectives
- Multiple-cycle input, concatenation, weighted fusion, and SOH-assisted variants
- Fixed 30- and 100-cycle observation windows
- Short-, middle-, and long-life subsets

The cleaned `04_rul_estimation.ipynb` uses only data available through a fixed observation
cycle and holds out batch 3.

## 5. Sequence and physics-informed branches

- Dense-to-sequence and sequence-to-sequence capacity forecasting
- Masked and beta-weighted objectives
- Physics-informed RUL models with fixed, separated, constrained, or supplied parameters
- Additional-information, memory-module, and NTK experiments
- Ensemble transformer and Bayesian-optimization variants

These branches are included as output-free representatives under `research_notebooks/`.
Related copies and horizon-only variants are consolidated by the principles in
`CONSOLIDATION.md`; original outputs and checkpoints remain excluded. Treat the advanced
models as research implementations until their split, target, feature availability, seed,
metric, and artifact format have been revalidated.

## Recommended next experiment contract

For any new model, record the following at the top of its notebook:

1. Prediction time and which signals are available at that time
2. SOH/RUL/end-of-life definition and cycle-index convention
3. Cell exclusions and correction rules
4. Cell-level train/validation/test split
5. Random seed and package versions
6. Baseline and evaluation metrics
7. Artifact path under `artifacts/`

This contract prevents target leakage and makes architectural comparisons meaningful.
