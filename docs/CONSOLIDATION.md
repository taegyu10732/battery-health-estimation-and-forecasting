# Consolidation principles

The private workspace remains unchanged. The public repository organizes the work by
research question and model family rather than by trial filename.

| Public area | What is consolidated |
| --- | --- |
| `notebooks/` | Four clean entry points for data inspection, clustering, SOH, and RUL |
| `research_notebooks/clustering/` | Engineered-feature MLP and raw-curve DenseNet methods |
| `research_notebooks/soh/` | MLP, CNN-Transformer, VAE, DeepHPM, knee, post-knee, and surrogate methods |
| `research_notebooks/rul/` | DenseNet, CNN-LSTM, Transformer, OP, gradient, dual-task, and fusion methods |
| `research_notebooks/sequence/` | LSTM seq2seq, masking/teacher forcing, and CNN-GRU forecasting |
| `research_notebooks/physics/` | Physics loss and fixed/supplied/constrained/memory parameter methods |
| `research_notebooks/method_development/` | Earlier normalization, ensemble, and empirical-physics development stages |

## What “consolidated” means

- Data loading exists once in `src/battery_soh/data.py`.
- Raw-cycle compatibility loading exists once in `src/battery_soh/raw_data.py`.
- Early-life and SOH feature construction exists once in `src/battery_soh/features.py`.
- Group splitting and metrics exist once in `src/battery_soh/evaluation.py`.
- Interpolation and artifact paths exist once in `preprocessing.py` and `paths.py`.
- Each public notebook owns one research question and has a configuration block near the top.
- Architecture-specific branches are not silently presented as validated results. Their
  intent remains in the experiment map and inventory, and they can be reintroduced behind
  the same shared data/split/metric interface.

This keeps the experimental history visible while preventing dozens of almost identical
files from becoming the public API of the project.
