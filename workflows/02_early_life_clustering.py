# %% [markdown]
# # 02. Early-life battery clustering
#
# This script groups cells using summary statistics available through a fixed observation
# cycle. Clustering is unsupervised: cycle life is withheld until after cluster assignment
# and is used only to interpret the groups.

# %%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from battery_soh.data import load_cell_summaries
from battery_soh.features import EARLY_FEATURE_COLUMNS, build_early_life_features

RANDOM_STATE = 42
OBSERVATION_CYCLE = 100
N_CLUSTERS = 3

cells = load_cell_summaries()
features = build_early_life_features(cells, observation_cycle=OBSERVATION_CYCLE)
print(f"Eligible cells: {len(features)}")
print(features.head())

# %% [markdown]
# ## Prepare the feature matrix
#
# Imputation and scaling are fit on the feature matrix only. Identifiers, batch, charge
# policy, recorded cycle life, and RUL are not clustering inputs.

# %%
preprocess = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
X = preprocess.fit_transform(features.loc[:, EARLY_FEATURE_COLUMNS])

kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=RANDOM_STATE)
features["cluster"] = kmeans.fit_predict(X)

pca = PCA(n_components=2, random_state=RANDOM_STATE)
embedding = pca.fit_transform(X)
features["pc1"] = embedding[:, 0]
features["pc2"] = embedding[:, 1]

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

for cluster, group in features.groupby("cluster"):
    axes[0].scatter(group["pc1"], group["pc2"], label=f"cluster {cluster}", alpha=0.8)
axes[0].set(
    xlabel="Principal component 1", ylabel="Principal component 2", title="Early-life feature space"
)
axes[0].legend()

features.boxplot(column="cycle_life", by="cluster", ax=axes[1], grid=False)
axes[1].set(xlabel="Cluster", ylabel="Recorded cycle life", title="Lifetime after clustering")
fig.suptitle("")
fig.tight_layout()

# %%
cluster_summary = (
    features.groupby("cluster")
    .agg(
        cells=("cell_id", "size"),
        median_cycle_life=("cycle_life", "median"),
        mean_cycle_life=("cycle_life", "mean"),
        batches=("batch", lambda values: ", ".join(sorted(set(values)))),
    )
    .sort_values("median_cycle_life")
)
print(cluster_summary)

feature_centres = pd.DataFrame(
    preprocess.named_steps["standardscaler"].inverse_transform(kmeans.cluster_centers_),
    columns=EARLY_FEATURE_COLUMNS,
)
print(feature_centres)

# %% [markdown]
# ## Interpretation limits
#
# K-means forces spherical clusters and the sample count is small. Treat these groups as a
# visual diagnostic, not as discovered battery mechanisms. Check whether a cluster is merely
# separating batches or charge policies before assigning a physical interpretation.
