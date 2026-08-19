# %% [markdown]
# # 03. State-of-health estimation baseline
#
# This script estimates cycle-level SOH from compact summary features. The split is made
# by cell, not by row, so a battery's early cycles cannot train a model that is evaluated on
# the same battery's later cycles.
#
# SOH is defined here as `QDischarge / 1.1 Ah`. Discharge capacity is the target and is not a
# model feature.

# %%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from battery_soh.data import load_cell_summaries
from battery_soh.evaluation import grouped_train_test_indices, regression_metrics
from battery_soh.features import SOH_FEATURE_COLUMNS, build_soh_samples

RANDOM_STATE = 42
NOMINAL_CAPACITY_AH = 1.1
SAMPLE_EVERY = 10
EXCLUDED_FEATURES = ()  # e.g. ("temperature_avg", "temperature_min", "temperature_max")
MODEL_FEATURES = tuple(name for name in SOH_FEATURE_COLUMNS if name not in EXCLUDED_FEATURES)

cells = load_cell_summaries()
samples = build_soh_samples(
    cells,
    nominal_capacity_ah=NOMINAL_CAPACITY_AH,
    sample_every=SAMPLE_EVERY,
)
print(f"Samples: {len(samples):,}; cells: {samples['cell_id'].nunique()}")
print(samples.head())

# %% [markdown]
# ## Battery-level split
#
# All preprocessing operations are inside the pipeline and are fit only on training cells.
# The charge-capacity channel is included as a simple proxy baseline; remove it if that
# measurement would not be available in the intended online setting.

# %%
train_index, test_index = grouped_train_test_indices(
    samples["cell_id"].to_numpy(),
    test_size=0.2,
    random_state=RANDOM_STATE,
)

train = samples.iloc[train_index].copy()
test = samples.iloc[test_index].copy()

assert set(train["cell_id"]).isdisjoint(test["cell_id"])
print(f"Train cells: {train['cell_id'].nunique()}; test cells: {test['cell_id'].nunique()}")

X_train = train.loc[:, MODEL_FEATURES]
y_train = train["soh"]
X_test = test.loc[:, MODEL_FEATURES]
y_test = test["soh"]

# %%
model = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=200,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    ),
)
model.fit(X_train, y_train)
prediction = model.predict(X_test)

metrics = pd.Series(regression_metrics(y_test.to_numpy(), prediction), name="SOH baseline")
print(metrics.to_frame())

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].scatter(y_test, prediction, alpha=0.35, s=14)
limits = [min(y_test.min(), prediction.min()), max(y_test.max(), prediction.max())]
axes[0].plot(limits, limits, color="black", linestyle="--", linewidth=1)
axes[0].set(xlabel="Measured SOH", ylabel="Predicted SOH", title="Held-out cells")

residual = prediction - y_test.to_numpy()
axes[1].scatter(test["cycle"], residual, alpha=0.35, s=14)
axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
axes[1].set(xlabel="Cycle", ylabel="Prediction − measurement", title="Residual by cycle")
fig.tight_layout()

# %% [markdown]
# ## Next checks
#
# Compare against a cycle-only model, repeat the group split across several seeds, and remove
# features unavailable at inference time. Specialized CNN, VAE, knee-point, and
# physics-informed models should be added only after they use this same split contract.
