# Architecture and methodology catalog

Static analysis of the 106 private source notebooks found 11 recurring architecture
families. Closely related class names such as `SOH_model` and `RUL_model` are counted by
their layer topology rather than as separate architectures.

## Architecture families

| # | Architecture family | Main use in this workspace | Public research groups |
| --- | --- | --- | --- |
| 1 | MLP | Engineered-feature clustering and SOH regression | G01, G04 |
| 2 | DenseNet-style CNN | Raw-curve clustering, RUL, and physics-loss baselines | G02, G12, G26–G27 |
| 3 | CNN + Transformer encoder | Normalized SOH/RUL, multi-cycle fusion, knee, and ensemble models | G03, G05, G13, G15–G22, G32–G35 |
| 4 | CNN + LSTM | Sequential RUL estimation | G14 |
| 5 | LSTM encoder-decoder | Capacity sequence forecasting | G23–G24 |
| 6 | DenseNet/CNN encoder + GRU decoder | Dense-to-sequence forecasting | G25 |
| 7 | VAE-style CNN + Transformer | Latent SOH representation and cycle conditioning | G06–G07 |
| 8 | CNN-Transformer + MLP + multihead-attention DeepHPM | Physics-informed SOH estimation | G08 |
| 9 | Post-knee LSTM | Capacity forecasting after the detected knee | G10 |
| 10 | Standard LSTM + custom recurrent cell | Surrogate degradation modeling | G11 |
| 11 | CNN-LSTM physics-parameter networks | Constrained, supplied, or memory-based degradation parameters | G28–G31 |

## Methodology branches retained

- Engineered early-life features and raw-curve classification
- Observation horizons from 60/90 through 300 cycles
- Optuna/Bayesian hyperparameter search and Transformer ensembles
- Voltage-only, charge-only, and no-temperature input ablations
- Rare/scarce-cell cohort studies
- VAE and cycle-conditioned latent representations
- DeepHPM/empirical physics losses
- Operating-profile inputs and cycle-gradient regularization
- Dual-task SOH + RUL learning
- Multi-cycle concat/additive/SOH-assisted/weighted fusion
- Fixed 30/70/100-cycle windows and short/middle/long-life cohorts
- Knee detection, knee-aware RUL, and post-knee sequence prediction
- LSTM seq2seq, masking, teacher forcing, beta weighting, and extra information
- Dense-to-sequence CNN/GRU forecasting
- Physics versus data-only loss comparison
- Fixed, separated, supplied, constrained, and learned-memory physics parameters
- Additional-information, 100th-cycle, and NTK variants
- Fine tuning and surrogate/custom-LSTM modeling

## Counting rule

The 35 method groups are not claimed to be 35 fundamentally different neural networks.
They are consolidated research questions or training methods. The 11 architecture families
describe distinct model topologies. This distinction avoids inflating the count with copied
notebooks, horizon numbers, or spelling-only filename changes.

The public notebook-level index is
[`research_notebooks/README.md`](../research_notebooks/README.md).
