# %% [markdown]
# # 04. Remaining-useful-life estimation baseline
#
# This script predicts RUL from a fixed early-life observation window. Every feature is
# computed at or before cycle 100. Batches 1 and 2 train the model; batch 3 is held out as a
# simple cross-batch test.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from battery_soh.data import load_cell_summaries
from battery_soh.evaluation import regression_metrics
from battery_soh.features import EARLY_FEATURE_COLUMNS, build_early_life_features

RANDOM_STATE = 42
OBSERVATION_CYCLE = 100
TRAIN_BATCHES = ("b1", "b2")
TEST_BATCH = "b3"
MODEL_FEATURES = EARLY_FEATURE_COLUMNS

cells = load_cell_summaries()
features = build_early_life_features(cells, observation_cycle=OBSERVATION_CYCLE)

train = features[features["batch"].isin(TRAIN_BATCHES)].copy()
test = features[features["batch"] == TEST_BATCH].copy()
print(f"Train cells: {len(train)}; held-out batch-3 cells: {len(test)}")
print(features.groupby("batch")[["cycle_life", "rul"]].describe())

# %% [markdown]
# ## Fit on log-RUL
#
# The log transform reduces the leverage of the longest-lived cells. Imputation is fit on
# the training batches only. Batch identifiers, cell identifiers, policy strings, cycle
# life, and RUL are excluded from the feature matrix.

# %%
X_train = train.loc[:, MODEL_FEATURES]
y_train = train["rul"].to_numpy()
X_test = test.loc[:, MODEL_FEATURES]
y_test = test["rul"].to_numpy()

model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=0.8,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
)
model.fit(X_train, np.log1p(y_train))
prediction = np.maximum(0, np.expm1(model.predict(X_test)))

metrics = pd.Series(regression_metrics(y_test, prediction), name="RUL baseline")
print(metrics.to_frame())

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

axes[0].scatter(y_test, prediction, alpha=0.8)
limits = [0, max(y_test.max(), prediction.max())]
axes[0].plot(limits, limits, color="black", linestyle="--", linewidth=1)
axes[0].set(
    xlabel="Measured RUL (cycles)", ylabel="Predicted RUL (cycles)", title="Held-out batch 3"
)

forest = model.named_steps["randomforestregressor"]
importance = pd.Series(forest.feature_importances_, index=MODEL_FEATURES).sort_values()
importance.plot.barh(ax=axes[1], color="tab:blue")
axes[1].set(xlabel="Random-forest importance", title="Baseline feature importance")
fig.tight_layout()

# %%
comparison = test[["cell_id", "cycle_life", "rul"]].copy()
comparison["predicted_rul"] = prediction
comparison["absolute_error"] = np.abs(comparison["rul"] - comparison["predicted_rul"])
print(comparison.sort_values("absolute_error", ascending=False).head(10))

# %% [markdown]
# ## Interpretation limits
#
# A batch holdout is stricter than a random row split but is not the exact published split.
# For a benchmark claim, reproduce the paper's corrected cell cohort and split protocol,
# then compare this summary-feature baseline with discharge-curve features under identical
# conditions.
