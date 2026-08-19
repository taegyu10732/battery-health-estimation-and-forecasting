# %% [markdown]
# # 01. Dataset overview
#
# This script inspects cell-level lifetime and per-cycle summary signals without loading
# the raw current/voltage traces. It is descriptive: no model is trained and no split is
# needed.
#
# Before running, install the project and set `BATTERY_DATA_DIR` as described in the README.
# The committed script has no outputs; execute it locally to generate figures.

# %%
import matplotlib.pyplot as plt
import pandas as pd

from battery_soh.data import load_cell_summaries, resolve_data_dir

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 20)

data_dir = resolve_data_dir()
cells = load_cell_summaries(data_dir)
print(f"Loaded {len(cells)} cell summaries from {data_dir}")

# %% [markdown]
# ## Cell-level inventory
#
# Cycle life and charge policy are descriptors. The per-cycle arrays are aligned only when
# they are converted to a tidy frame, which avoids assuming identical lengths across fields.

# %%
inventory = pd.DataFrame(
    {
        "cell_id": cell.cell_id,
        "batch": cell.batch,
        "cycle_life": cell.cycle_life,
        "recorded_cycles": len(cell.cycle),
        "charge_policy": cell.charge_policy,
    }
    for cell in cells
)

print(
    inventory.groupby("batch").agg(
        cells=("cell_id", "size"),
        median_life=("cycle_life", "median"),
        min_life=("cycle_life", "min"),
        max_life=("cycle_life", "max"),
    )
)
print(inventory.head())

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for batch, group in inventory.groupby("batch"):
    axes[0].hist(group["cycle_life"], bins=15, alpha=0.55, label=batch)
axes[0].set(xlabel="Recorded cycle life", ylabel="Cell count", title="Lifetime distribution")
axes[0].legend(title="Batch")

inventory.boxplot(column="cycle_life", by="batch", ax=axes[1], grid=False)
axes[1].set(xlabel="Batch", ylabel="Recorded cycle life", title="Lifetime by batch")
fig.suptitle("")
fig.tight_layout()

# %% [markdown]
# ## Capacity trajectories
#
# SOH is shown as discharge capacity divided by 1.1 Ah. This is a transparent convention,
# not a universal definition; change the nominal capacity when comparing with another study.

# %%
nominal_capacity_ah = 1.1
fig, ax = plt.subplots(figsize=(10, 5))

for cell in cells:
    frame = cell.to_frame()
    valid = frame["cycle"] <= cell.cycle_life
    ax.plot(
        frame.loc[valid, "cycle"],
        frame.loc[valid, "discharge_capacity"] / nominal_capacity_ah,
        alpha=0.18,
        linewidth=0.8,
        color={"b1": "tab:blue", "b2": "tab:orange", "b3": "tab:green"}[cell.batch],
    )

ax.axhline(0.8, color="black", linestyle="--", linewidth=1, label="80% SOH reference")
ax.set(xlabel="Cycle", ylabel="SOH (Qd / 1.1 Ah)", title="Discharge-capacity trajectories")
ax.legend()
fig.tight_layout()

# %% [markdown]
# ## Temperature and internal resistance
#
# The summary channels are useful for sanity checks. Outliers should be investigated before
# fitting a model; they should not be silently removed based on test performance.

# %%
summary_rows = []
for cell in cells:
    frame = cell.to_frame()
    frame = frame.loc[(frame["cycle"] >= 1) & (frame["cycle"] <= 100)]
    summary_rows.append(
        {
            "cell_id": cell.cell_id,
            "batch": cell.batch,
            "mean_temperature": frame["temperature_avg"].mean(),
            "max_temperature": frame["temperature_max"].max(),
            "mean_internal_resistance": frame["internal_resistance"].mean(),
        }
    )

summary_frame = pd.DataFrame(summary_rows)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for batch, group in summary_frame.groupby("batch"):
    axes[0].scatter(group["mean_temperature"], group["max_temperature"], label=batch, alpha=0.75)
    axes[1].hist(group["mean_internal_resistance"].dropna(), bins=15, alpha=0.55, label=batch)
axes[0].set(xlabel="Mean Tavg, cycles 1–100", ylabel="Maximum Tmax", title="Temperature summary")
axes[1].set(
    xlabel="Mean internal resistance, cycles 1–100",
    ylabel="Cell count",
    title="Internal-resistance summary",
)
axes[0].legend(title="Batch")
axes[1].legend(title="Batch")
fig.tight_layout()

# %% [markdown]
# ## Reading the plots
#
# Use this script to confirm that the three batches loaded correctly and to identify
# measurement anomalies. Any exclusion or correction should be documented before the model
# scripts are run.
