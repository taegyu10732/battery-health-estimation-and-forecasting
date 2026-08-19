# %% [markdown]
# # Consolidated research script
#
# Method group **G01**: Engineered-feature lifetime clustering. Architecture: MLP. Method tags: engineered features|lifetime classes.
#
# This copy preserves the research implementation while removing saved outputs, execution counters, pinned GPU selection, and scratch-only cells. Closely related variants are summarized in the research script index. During cleanup, 1 syntactically invalid scratch cell(s) and 1 display-only scratch cell(s) were omitted.

# %%
from battery_soh.data import resolve_data_dir
from battery_soh.paths import artifact_path
from battery_soh.raw_data import load_battery_dictionary
from scipy.stats import skew, kurtosis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
# Shared, portable raw-data loading. This may require substantial memory.
RESEARCH_BATCHES = ("b1", "b2", "b3")
bat_dict = load_battery_dictionary(batches=RESEARCH_BATCHES)

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
batch_keys = [*bat_dict.keys()]
len(batch_keys)

# %%
random.seed(319)

# %%
batch_keys = [*bat_dict.keys()]
selected_keys = []

for battery_key in batch_keys:
    if (len(bat_dict[battery_key]["summary"]["cycle"]) > 300) & (
        len(bat_dict[battery_key]["summary"]["cycle"]) < 2000
    ):
        for k in range(1, len(bat_dict[battery_key]["summary"]["QD"])):
            QD = bat_dict[battery_key]["summary"]["QD"][k]
            temp_SOH = (QD / 1.1) * 100
            if temp_SOH < 80.2:
                selected_keys.append(battery_key)
                break

# %%
selected_keys.remove("b2c9")
selected_keys.remove("b1c7")
selected_keys.remove("b2c17")
selected_keys.remove("b2c1")
selected_keys.remove("b3c46")
selected_keys.remove("b2c33")
selected_keys.remove("b2c41")
selected_keys.remove("b2c25")

# %%
random.shuffle(selected_keys)


# %%
def dataset_preprocessor(bat_key):
    features_df = pd.DataFrame()
    features_df["cell_key"] = np.array(bat_key)

    minimum_dQ_100_10 = np.zeros(len(bat_key))
    variance_dQ_100_10 = np.zeros(len(bat_key))
    skewness_dQ_100_10 = np.zeros(len(bat_key))
    kurtosis_dQ_100_10 = np.zeros(len(bat_key))

    i = 0
    for battery_key in bat_key:
        c10 = bat_dict[battery_key]["cycles"]["10"]
        c100 = bat_dict[battery_key]["cycles"]["100"]
        dQ_100_10 = c100["Qdlin"] - c10["Qdlin"]

        minimum_dQ_100_10[i] = np.log(np.abs(np.min(dQ_100_10)))
        variance_dQ_100_10[i] = np.log(np.var(dQ_100_10))
        skewness_dQ_100_10[i] = np.log(np.abs(skew(dQ_100_10)))
        kurtosis_dQ_100_10[i] = np.log(np.abs(kurtosis(dQ_100_10)))

        i += 1

    features_df["minimum_dQ_100_10"] = minimum_dQ_100_10
    features_df["variance_dQ_100_10"] = variance_dQ_100_10
    features_df["skewness_dQ_100_10"] = skewness_dQ_100_10
    features_df["kurtosis_dQ_100_10"] = kurtosis_dQ_100_10

    slope_lin_fit_2_100 = np.zeros(len(bat_key))
    intercept_lin_fit_2_100 = np.zeros(len(bat_key))
    discharge_capacity_2 = np.zeros(len(bat_key))
    diff_discharge_capacity_max_2 = np.zeros(len(bat_key))

    i = 0

    for battery_key in bat_key:
        q = bat_dict[battery_key]["summary"]["QD"][1:100].reshape(
            -1, 1
        )  # discharge cappacities; q.shape = (99, 1);
        X = cycle_numbers = bat_dict[battery_key]["summary"]["cycle"][1:100].reshape(
            -1, 1
        )  # Cylce index from 2 to 100; X.shape = (99, 1)

        linear_regressor_2_100 = LinearRegression()
        linear_regressor_2_100.fit(X, q)

        slope_lin_fit_2_100[i] = linear_regressor_2_100.coef_[0]
        intercept_lin_fit_2_100[i] = linear_regressor_2_100.intercept_
        # discharge_capacity_2[i] = q[0][0]
        # diff_discharge_capacity_max_2[i] = np.max(q) - q[0][0]
        i += 1

    features_df["slope_lin_fit_2_100"] = slope_lin_fit_2_100
    features_df["intercept_lin_fit_2_100"] = intercept_lin_fit_2_100
    # features_df["discharge_capacity_2"] = discharge_capacity_2
    # features_df["diff_discharge_capacity_max_2"] = diff_discharge_capacity_max_2

    Cycle_life = np.zeros(len(bat_key))

    i = 0

    for battery_key in bat_key:
        Cell_QD = bat_dict[battery_key]["summary"]["QD"]
        for k in range(1, len(Cell_QD)):
            QD = Cell_QD[k]
            temp_SOH = (QD / 1.1) * 100
            if temp_SOH < 80.2:
                Cycle_life[i] = k
                break
        i += 1

    features_df["Cycle_life"] = Cycle_life

    Early_age = np.zeros(len(bat_key))

    for i in range(0, len(bat_key)):
        if features_df["Cycle_life"][i] <= 600:
            Early_age[i] = 1
        else:
            Early_age[i] = 0

    features_df["Early_age"] = Early_age

    Middle_age = np.zeros(len(bat_key))

    for i in range(0, len(bat_key)):
        if (features_df["Cycle_life"][i] > 600) & (features_df["Cycle_life"][i] <= 900):
            Middle_age[i] = 1
        else:
            Middle_age[i] = 0

    features_df["Middle_age"] = Middle_age

    Old_age = np.zeros(len(bat_key))

    for i in range(0, len(bat_key)):
        if features_df["Cycle_life"][i] > 900:
            Old_age[i] = 1
        else:
            Old_age[i] = 0

    features_df["Old_age"] = Old_age

    return features_df


# %%
total_df = dataset_preprocessor(selected_keys)

# %%
train_df = total_df.iloc[:75, :].reset_index()
val_df = total_df.iloc[75:95, :].reset_index()
test_df = total_df.iloc[95:, :].reset_index()

# %%
print(len(total_df[total_df["Early_age"] == 1]))
print(len(total_df[total_df["Middle_age"] == 1]))
print(len(total_df[total_df["Old_age"] == 1]))

# %%
train_max_value = []
train_min_value = []

for i in range(2, 8):
    train_max_value.append(train_df.iloc[:, i].max())
    train_min_value.append(train_df.iloc[:, i].min())

    train_df.iloc[:, i] = (
        (1 - 1e-6)
        * (train_df.iloc[:, i] - train_df.iloc[:, i].min())
        / (train_df.iloc[:, i].max() - train_df.iloc[:, i].min())
    ) + 1e-6

# %%
for i in range(2, 8):
    val_df.iloc[:, i] = (1 - 1e-6) * (
        (val_df.iloc[:, i] - train_min_value[i - 2])
        / (train_max_value[i - 2] - train_min_value[i - 2])
    ) + 1e-6
    test_df.iloc[:, i] = (1 - 1e-6) * (
        (test_df.iloc[:, i] - train_min_value[i - 2])
        / (train_max_value[i - 2] - train_min_value[i - 2])
    ) + 1e-6

# %%
train_X = []
train_y = []

for i in range(0, len(train_df)):
    train_X.append(np.array(train_df.iloc[i, 2:8], dtype=float))
    train_y.append(np.array(train_df.iloc[i, 9:], dtype=float))

# %%
val_X = []
val_y = []

for i in range(0, len(val_df)):
    val_X.append(np.array(val_df.iloc[i, 2:8], dtype=float))
    val_y.append(np.array(val_df.iloc[i, 9:], dtype=float))

# %%
test_X = []
test_y = []

for i in range(0, len(test_df)):
    test_X.append(np.array(test_df.iloc[i, 2:8], dtype=float))
    test_y.append(np.array(test_df.iloc[i, 9:], dtype=float))

# %%
train_X = np.array(train_X)
train_y = np.array(train_y)
val_X = np.array(val_X)
val_y = np.array(val_y)
test_X = np.array(test_X)
test_y = np.array(test_y)

train_X_gpu = torch.tensor(train_X, dtype=torch.float32).to(DEVICE)
train_y_gpu = torch.tensor(train_y, dtype=torch.float32).to(DEVICE)
val_X = torch.tensor(val_X, dtype=torch.float32).to(DEVICE)
val_y = torch.tensor(val_y, dtype=torch.float32).to(DEVICE)
test_X = torch.tensor(test_X, dtype=torch.float32).to(DEVICE)
test_y = torch.tensor(test_y, dtype=torch.float32).to(DEVICE)


# %%
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


dataset = MyDataset(
    torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32)
)

data_loader = DataLoader(dataset, batch_size=5, shuffle=True)


# %%
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(6, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 50)
        self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = F.leaky_relu(x)
        x = torch.relu(self.fc2(x))
        x = F.leaky_relu(x)
        x = torch.relu(self.fc3(x))
        x = F.leaky_relu(x)
        x = torch.relu(self.fc4(x))

        return torch.softmax(self.fc5(x), dim=1)


# %%
model = MLP().to(DEVICE)
model.train()

loss_best = 2

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(2000):
    idx = 0
    for batch_X, batch_y in data_loader:
        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        output = model(batch_X)
        loss = loss_fn(output, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()

        pred_val = model(val_X)
        val_loss = loss_fn(pred_val, val_y)

        print(
            "epoch : ",
            epoch,
            "/idx : ",
            idx,
            "/loss :",
            round(float(loss), 4),
            "/val_loss :",
            round(float(val_loss), 4),
        )

        if val_loss < loss_best:
            loss_best = val_loss
            torch.save(model.state_dict(), artifact_path("clustering_nature_feature_1"))

        idx += 1

# %%
model_new = MLP().to(DEVICE)
model_new.load_state_dict(torch.load(artifact_path("clustering_nature_feature_1")))

# %%
predicted = model_new(train_X_gpu).cpu().detach().numpy()
predicted = np.argmax(predicted, axis=1)
target_data_class = np.argmax(train_y, axis=1)

accuracy = accuracy_score(target_data_class, predicted)
precision = precision_score(target_data_class, predicted, average="macro")
recall = recall_score(target_data_class, predicted, average="macro")
f1 = f1_score(target_data_class, predicted, average="macro")

print(accuracy)
print(precision)
print(recall)
print(f1)

# %%
predicted = model_new(val_X).cpu().detach().numpy()
predicted = np.argmax(predicted, axis=1)
target_data_class = np.argmax(val_y.cpu().detach().numpy(), axis=1)

accuracy = accuracy_score(target_data_class, predicted)
precision = precision_score(target_data_class, predicted, average="macro")
recall = recall_score(target_data_class, predicted, average="macro")
f1 = f1_score(target_data_class, predicted, average="macro")

print(accuracy)
print(precision)
print(recall)
print(f1)

# %%
predicted = model_new(test_X).cpu().detach().numpy()
predicted = np.argmax(predicted, axis=1)
target_data_class = np.argmax(test_y.cpu().detach().numpy(), axis=1)

accuracy = accuracy_score(target_data_class, predicted)
precision = precision_score(target_data_class, predicted, average="macro")
recall = recall_score(target_data_class, predicted, average="macro")
f1 = f1_score(target_data_class, predicted, average="macro")

print(accuracy)
print(precision)
print(recall)
print(f1)

# %%
cm = confusion_matrix(target_data_class, predicted)

# %%
cm_df = pd.DataFrame(cm, index=["Early", "Middle", "Old"], columns=["Early", "Middle", "Old"])

# %%
plt.figure(figsize=(5, 5))
sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    linewidths=0.5,
    cmap="Blues",
    cbar=False,
    annot_kws={"size": 14},
    square=True,
)
plt.ylabel("True", fontsize=15)
plt.xlabel("Predict", fontsize=15)

# %%
for one_key in total_df[total_df["Early_age"] == 1]["cell_key"]:
    plt.plot(bat_dict[one_key]["summary"]["QD"], color="green")

for one_key in total_df[total_df["Middle_age"] == 1]["cell_key"]:
    plt.plot(bat_dict[one_key]["summary"]["QD"], color="orange")

for one_key in total_df[total_df["Old_age"] == 1]["cell_key"]:
    plt.plot(bat_dict[one_key]["summary"]["QD"], color="blue")


plt.ylim([0.88, 1.1])
