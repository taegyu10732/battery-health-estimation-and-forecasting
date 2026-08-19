# %% [markdown]
# # Consolidated research script
#
# Method group **G02**: Raw-curve lifetime clustering. Architecture: DenseNet CNN. Method tags: raw curves|classification.
#
# This copy preserves the research implementation while removing saved outputs, execution counters, pinned GPU selection, and scratch-only cells. Closely related variants are summarized in the research script index. During cleanup, 0 syntactically invalid scratch cell(s) and 4 display-only scratch cell(s) were omitted.

# %%
from battery_soh.data import resolve_data_dir
from battery_soh.evaluation import MAPE, RMSPE
from battery_soh.paths import artifact_path
from battery_soh.preprocessing import interpolate_timeseries
from battery_soh.raw_data import load_battery_dictionary
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random
import scipy.io
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
len(selected_keys)

# %%
random.shuffle(selected_keys)

# %%
train_bat_key = selected_keys[:75]
val_bat_key = selected_keys[75:95]
test_bat_key = selected_keys[95:]

# %%
time_value = 5
inter_value = 100

# %%
whole_data = []
RUL_clustering = []

for battery_key in train_bat_key:
    temporal_data_CI = []
    temporal_data_CV = []
    temporal_data_CT = []
    temporal_data_DV = []

    for k in range(0, len(bat_dict[battery_key]["summary"]["QD"])):
        cycle_str = str(k)
        QD = bat_dict[battery_key]["summary"]["QD"][int(cycle_str)]
        temp_SOH = (QD / 1.1) * 100
        if temp_SOH < 80.2:
            if int(cycle_str) <= 600:
                RUL_clustering.append([1, 0, 0])
            if (int(cycle_str) > 600) & (int(cycle_str) <= 900):
                RUL_clustering.append([0, 1, 0])
            if int(cycle_str) > 900:
                RUL_clustering.append([0, 0, 1])
            break

    for cycle_num in range(0, 100):
        cycle_num = str(cycle_num)

        data_length = len(bat_dict[battery_key]["cycles"][cycle_num]["Qc"])

        for i in range(11, data_length):
            Charge_SOC = (bat_dict[battery_key]["cycles"][cycle_num]["Qc"][i] / 1.1) * 100

            if Charge_SOC > 80.1:
                for p in range(11, data_length):
                    start_time = bat_dict[battery_key]["cycles"][cycle_num]["t"][i]
                    end_time = bat_dict[battery_key]["cycles"][cycle_num]["t"][i + p]
                    total_time = end_time - start_time

                    if total_time > time_value:
                        # print(total_time)
                        final_idx = p - 1
                        break

                Distinguish_data = bat_dict[battery_key]["cycles"][cycle_num]["I"][
                    i : i + final_idx
                ]
                has_negative = np.any(Distinguish_data < 0)
                if has_negative:
                    break

                time_data = np.round(
                    bat_dict[battery_key]["cycles"][cycle_num]["t"][i : i + final_idx], 3
                )
                Current_data = np.round(
                    bat_dict[battery_key]["cycles"][cycle_num]["I"][i : i + final_idx], 3
                )
                Voltage_data = bat_dict[battery_key]["cycles"][cycle_num]["V"][i : i + final_idx]
                Temperature_data = bat_dict[battery_key]["cycles"][cycle_num]["T"][
                    i : i + final_idx
                ]

                df = pd.DataFrame(
                    {"time": time_data, "Current": Current_data, "Voltage": Voltage_data}
                )

                df = df.drop_duplicates(["time"])

                data = np.array(df.T)

                Current_data = interpolate_timeseries(data[1], inter_value)
                Voltage_data = interpolate_timeseries(data[2], inter_value)
                Temperature_data = interpolate_timeseries(Temperature_data, inter_value)

                Voltage_data = (Voltage_data - 3.37) / (3.62 - 3.37)
                Temperature_data = (Temperature_data - 27.2) / (41 - 27.2)

                temporal_data_CI.append(Current_data)
                temporal_data_CV.append(Voltage_data)
                temporal_data_CT.append(Temperature_data)

                break

        for i in range(11, len(bat_dict[battery_key]["cycles"][cycle_num]["Qd"])):
            Discharge_SOC = (bat_dict[battery_key]["cycles"][cycle_num]["Qd"][i] / 1.1) * 100
            if Discharge_SOC > 0.01:
                for p in range(data_length):
                    start_time = bat_dict[battery_key]["cycles"][cycle_num]["t"][i]
                    end_time = bat_dict[battery_key]["cycles"][cycle_num]["t"][i + p]
                    total_time = end_time - start_time
                    if total_time > time_value:
                        final_idx = p - 1
                        break

                time_data = np.round(
                    bat_dict[battery_key]["cycles"][cycle_num]["t"][i : i + final_idx], 3
                )
                Discharge_Voltage = bat_dict[battery_key]["cycles"][cycle_num]["V"][
                    i : i + final_idx
                ]

                df = pd.DataFrame({"time": time_data, "Voltage": Discharge_Voltage})

                df = df.drop_duplicates(["time"])

                data = np.array(df.T)

                Discharge_Voltage = interpolate_timeseries(data[1], inter_value)
                Discharge_Voltage = (Discharge_Voltage - 3) / (3.6 - 3)

                temporal_data_DV.append(Discharge_Voltage)

                break

    cycle_merge_data = [temporal_data_CI, temporal_data_CV, temporal_data_CT, temporal_data_DV]
    whole_data.append(cycle_merge_data)

whole_data = np.array(whole_data)
RUL_clustering = np.array(RUL_clustering)

# %%
whole_data_tensor = torch.tensor(whole_data, dtype=torch.float32).to(DEVICE)
RUL_clustering_tensor = torch.tensor(RUL_clustering, dtype=torch.float32).to(DEVICE)


# %%
class MyDataset(Dataset):
    def __init__(self, X1, y1):
        self.X1 = X1
        self.y1 = y1

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        return self.X1[idx], self.y1[idx]


dataset = MyDataset(
    torch.tensor(whole_data, dtype=torch.float32), torch.tensor(RUL_clustering, dtype=torch.float32)
)

data_loader = DataLoader(dataset, batch_size=5, shuffle=True)

# %%
whole_val_data = []
RUL_clustering_val = []

for battery_key in val_bat_key:
    temporal_data_CI = []
    temporal_data_CV = []
    temporal_data_CT = []
    temporal_data_DV = []

    for i in range(1, len(bat_dict[battery_key]["summary"]["QD"])):
        cycle_str = str(i)
        QD = bat_dict[battery_key]["summary"]["QD"][int(cycle_str)]
        temp_SOH = (QD / 1.1) * 100
        if temp_SOH < 80.2:
            if int(cycle_str) <= 600:
                RUL_clustering_val.append([1, 0, 0])
            if (int(cycle_str) > 600) & (int(cycle_str) <= 900):
                RUL_clustering_val.append([0, 1, 0])
            if int(cycle_str) > 900:
                RUL_clustering_val.append([0, 0, 1])
            break

    for cycle_num in range(0, 100):
        cycle_num = str(cycle_num)

        data_length = len(bat_dict[battery_key]["cycles"][cycle_num]["Qc"])

        for i in range(11, data_length):
            Charge_SOC = (bat_dict[battery_key]["cycles"][cycle_num]["Qc"][i] / 1.1) * 100
            if Charge_SOC > 80.1:
                for p in range(11, data_length):
                    start_time = bat_dict[battery_key]["cycles"][cycle_num]["t"][i]
                    end_time = bat_dict[battery_key]["cycles"][cycle_num]["t"][i + p]
                    total_time = end_time - start_time
                    if total_time > time_value:
                        final_idx = p - 1
                        break

                Distinguish_data = bat_dict[battery_key]["cycles"][cycle_num]["I"][
                    i : i + final_idx
                ]
                has_negative = np.any(Distinguish_data < 0)
                if has_negative:
                    break

                time_data = np.round(
                    bat_dict[battery_key]["cycles"][cycle_num]["t"][i : i + final_idx], 3
                )
                Current_data = np.round(
                    bat_dict[battery_key]["cycles"][cycle_num]["I"][i : i + final_idx], 3
                )
                Voltage_data = bat_dict[battery_key]["cycles"][cycle_num]["V"][i : i + final_idx]
                Temperature_data = bat_dict[battery_key]["cycles"][cycle_num]["T"][
                    i : i + final_idx
                ]

                df = pd.DataFrame(
                    {"time": time_data, "Current": Current_data, "Voltage": Voltage_data}
                )

                df = df.drop_duplicates(["time"])

                data = np.array(df.T)

                Current_data = interpolate_timeseries(data[1], inter_value)
                Voltage_data = interpolate_timeseries(data[2], inter_value)
                Temperature_data = interpolate_timeseries(Temperature_data, inter_value)

                Voltage_data = (Voltage_data - 3.37) / (3.62 - 3.37)
                Temperature_data = (Temperature_data - 27.2) / (41 - 27.2)

                temporal_data_CI.append(Current_data)
                temporal_data_CV.append(Voltage_data)
                temporal_data_CT.append(Temperature_data)

                break

        for i in range(11, len(bat_dict[battery_key]["cycles"][cycle_num]["Qd"])):
            Discharge_SOC = (bat_dict[battery_key]["cycles"][cycle_num]["Qd"][i] / 1.1) * 100
            if Discharge_SOC > 0.01:
                for p in range(data_length):
                    start_time = bat_dict[battery_key]["cycles"][cycle_num]["t"][i]
                    end_time = bat_dict[battery_key]["cycles"][cycle_num]["t"][i + p]
                    total_time = end_time - start_time
                    if total_time > time_value:
                        final_idx = p - 1
                        break

                time_data = np.round(
                    bat_dict[battery_key]["cycles"][cycle_num]["t"][i : i + final_idx], 3
                )
                Discharge_Voltage = bat_dict[battery_key]["cycles"][cycle_num]["V"][
                    i : i + final_idx
                ]

                df = pd.DataFrame({"time": time_data, "Voltage": Discharge_Voltage})

                df = df.drop_duplicates(["time"])

                data = np.array(df.T)

                Discharge_Voltage = interpolate_timeseries(data[1], inter_value)
                Discharge_Voltage = (Discharge_Voltage - 3) / (3.6 - 3)

                temporal_data_DV.append(Discharge_Voltage)

                break

    cycle_merge_data = [temporal_data_CI, temporal_data_CV, temporal_data_CT, temporal_data_DV]
    whole_val_data.append(cycle_merge_data)

whole_val_data = np.array(whole_val_data)
RUL_clustering_val = np.array(RUL_clustering_val)
whole_val_data = torch.tensor(whole_val_data, dtype=torch.float32).to(DEVICE)
RUL_clustering_val = torch.tensor(RUL_clustering_val, dtype=torch.float32).to(DEVICE)

# %%
whole_test_data = []
RUL_clustering_test = []

for battery_key in test_bat_key:
    temporal_data_CI = []
    temporal_data_CV = []
    temporal_data_CT = []
    temporal_data_DV = []

    for i in range(1, len(bat_dict[battery_key]["summary"]["QD"])):
        cycle_str = str(i)
        QD = bat_dict[battery_key]["summary"]["QD"][int(cycle_str)]
        temp_SOH = (QD / 1.1) * 100
        if temp_SOH < 80.2:
            if int(cycle_str) <= 600:
                RUL_clustering_test.append([1, 0, 0])
            if (int(cycle_str) > 600) & (int(cycle_str) <= 900):
                RUL_clustering_test.append([0, 1, 0])
            if int(cycle_str) > 900:
                RUL_clustering_test.append([0, 0, 1])
            break

    for cycle_num in range(0, 100):
        cycle_num = str(cycle_num)

        data_length = len(bat_dict[battery_key]["cycles"][cycle_num]["Qc"])

        for i in range(11, data_length):
            Charge_SOC = (bat_dict[battery_key]["cycles"][cycle_num]["Qc"][i] / 1.1) * 100
            if Charge_SOC > 80.1:
                for p in range(11, data_length):
                    start_time = bat_dict[battery_key]["cycles"][cycle_num]["t"][i]
                    end_time = bat_dict[battery_key]["cycles"][cycle_num]["t"][i + p]
                    total_time = end_time - start_time
                    if total_time > time_value:
                        final_idx = p - 1
                        break

                Distinguish_data = bat_dict[battery_key]["cycles"][cycle_num]["I"][
                    i : i + final_idx
                ]
                has_negative = np.any(Distinguish_data < 0)
                if has_negative:
                    break

                time_data = np.round(
                    bat_dict[battery_key]["cycles"][cycle_num]["t"][i : i + final_idx], 3
                )
                Current_data = np.round(
                    bat_dict[battery_key]["cycles"][cycle_num]["I"][i : i + final_idx], 3
                )
                Voltage_data = bat_dict[battery_key]["cycles"][cycle_num]["V"][i : i + final_idx]
                Temperature_data = bat_dict[battery_key]["cycles"][cycle_num]["T"][
                    i : i + final_idx
                ]

                df = pd.DataFrame(
                    {"time": time_data, "Current": Current_data, "Voltage": Voltage_data}
                )

                df = df.drop_duplicates(["time"])

                data = np.array(df.T)

                Current_data = interpolate_timeseries(data[1], inter_value)
                Voltage_data = interpolate_timeseries(data[2], inter_value)
                Temperature_data = interpolate_timeseries(Temperature_data, inter_value)

                Voltage_data = (Voltage_data - 3.37) / (3.62 - 3.37)
                Temperature_data = (Temperature_data - 27.2) / (41 - 27.2)

                temporal_data_CI.append(Current_data)
                temporal_data_CV.append(Voltage_data)
                temporal_data_CT.append(Temperature_data)

                break

        for i in range(11, len(bat_dict[battery_key]["cycles"][cycle_num]["Qd"])):
            Discharge_SOC = (bat_dict[battery_key]["cycles"][cycle_num]["Qd"][i] / 1.1) * 100
            if Discharge_SOC > 0.01:
                for p in range(data_length):
                    start_time = bat_dict[battery_key]["cycles"][cycle_num]["t"][i]
                    end_time = bat_dict[battery_key]["cycles"][cycle_num]["t"][i + p]
                    total_time = end_time - start_time
                    if total_time > time_value:
                        final_idx = p - 1
                        break

                time_data = np.round(
                    bat_dict[battery_key]["cycles"][cycle_num]["t"][i : i + final_idx], 3
                )
                Discharge_Voltage = bat_dict[battery_key]["cycles"][cycle_num]["V"][
                    i : i + final_idx
                ]

                df = pd.DataFrame({"time": time_data, "Voltage": Discharge_Voltage})

                df = df.drop_duplicates(["time"])

                data = np.array(df.T)

                Discharge_Voltage = interpolate_timeseries(data[1], inter_value)
                Discharge_Voltage = (Discharge_Voltage - 3) / (3.6 - 3)

                temporal_data_DV.append(Discharge_Voltage)

                break

    cycle_merge_data = [temporal_data_CI, temporal_data_CV, temporal_data_CT, temporal_data_DV]
    whole_test_data.append(cycle_merge_data)

whole_test_data = np.array(whole_test_data)
RUL_clustering_test = np.array(RUL_clustering_test)
whole_test_data = torch.tensor(whole_test_data, dtype=torch.float32).to(DEVICE)
RUL_clustering_test = torch.tensor(RUL_clustering_test, dtype=torch.float32).to(DEVICE)


# %%
class BottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inner_channels = 4 * growth_rate

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, inner_channels, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.LeakyReLU(),
            nn.Conv2d(inner_channels, growth_rate, 3, stride=1, padding=1, bias=False),
        )

        self.shortcut = nn.Sequential()

    def forward(self, x):
        return torch.cat([self.shortcut(x), self.residual(x)], 1)


# %%
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(2, stride=2),
        )

    def forward(self, x):
        return self.down_sample(x)


# %%
class DenseNet(nn.Module):
    def __init__(self, nblocks, growth_rate=12, reduction=0.5, init_weights=False):
        super().__init__()

        self.growth_rate = growth_rate
        inner_channels = 2 * growth_rate  # output channels of conv1 before entering Dense Block

        self.conv1 = nn.Sequential(
            nn.Conv2d(4, inner_channels, 7, stride=2, padding=3), nn.MaxPool2d(3, 2, padding=1)
        )

        self.features = nn.Sequential()

        for i in range(len(nblocks) - 1):
            self.features.add_module(
                "dense_block_{}".format(i), self._make_dense_block(nblocks[i], inner_channels)
            )
            inner_channels += growth_rate * nblocks[i]
            out_channels = int(reduction * inner_channels)
            self.features.add_module(
                "transition_layer_{}".format(i), Transition(inner_channels, out_channels)
            )
            inner_channels = out_channels

        self.features.add_module(
            "dense_block_{}".format(len(nblocks) - 1),
            self._make_dense_block(nblocks[len(nblocks) - 1], inner_channels),
        )
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module("bn", nn.BatchNorm2d(inner_channels))
        self.features.add_module("relu", nn.ReLU())

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(inner_channels, 50)
        self.linear_f = nn.Linear(50, 3)

        # weight initialization
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = F.leaky_relu(x)
        x = self.linear_f(x)
        return x.squeeze(1)

    def _make_dense_block(self, nblock, inner_channels):
        dense_block = nn.Sequential()
        for i in range(nblock):
            dense_block.add_module(
                "bottle_neck_layer_{}".format(i), BottleNeck(inner_channels, self.growth_rate)
            )
            inner_channels += self.growth_rate
        return dense_block

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def DenseNet_121():
    return DenseNet([6, 12, 24, 6])


# %%
model = DenseNet_121().to(DEVICE)
model = model.train()

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

loss_best = 1000

for epoch in range(0, 500):
    idx = 0
    for batch_X1, batch_y1 in data_loader:
        batch_X1 = batch_X1.to(DEVICE)
        batch_y1 = batch_y1.to(DEVICE)
        model.train()
        RUL_pred = model(batch_X1)
        loss = loss_fn(RUL_pred, batch_y1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()

        RUL_pred_val = model(whole_val_data)

        # RUL_pred_val = RUL_pred_val.cpu().detach().numpy()

        RUL_val_loss = loss_fn(RUL_pred_val, RUL_clustering_val)

        print(
            "epoch : ",
            epoch,
            "/idx : ",
            idx,
            "/loss :",
            round(float(loss), 4),
            "/val_loss :",
            round(float(RUL_val_loss), 4),
        )

        if RUL_val_loss < loss_best:
            loss_best = RUL_val_loss
            torch.save(model.state_dict(), artifact_path("clustering_raw_data_3"))

        idx += 1

# %%
model_new = DenseNet_121().to(DEVICE)
model_new.load_state_dict(torch.load(artifact_path("clustering_raw_data_2")))

# %%
predicted = model_new(whole_data_tensor).cpu().detach().numpy()
predicted = np.argmax(predicted, axis=1)
target_data_class = np.argmax(RUL_clustering, axis=1)

accuracy = accuracy_score(target_data_class, predicted)
precision = precision_score(target_data_class, predicted, average="macro")
recall = recall_score(target_data_class, predicted, average="macro")
f1 = f1_score(target_data_class, predicted, average="macro")

print(accuracy)
print(precision)
print(recall)
print(f1)

# %%
predicted = model_new(whole_val_data).cpu().detach().numpy()
predicted = np.argmax(predicted, axis=1)
target_data_class = np.argmax(RUL_clustering_val.cpu(), axis=1)

accuracy = accuracy_score(target_data_class, predicted)
precision = precision_score(target_data_class, predicted, average="macro")
recall = recall_score(target_data_class, predicted, average="macro")
f1 = f1_score(target_data_class, predicted, average="macro")

print(accuracy)
print(precision)
print(recall)
print(f1)

# %%
predicted = model_new(whole_test_data).cpu().detach().numpy()
predicted = np.argmax(predicted, axis=1)
target_data_class = np.argmax(RUL_clustering_test.cpu(), axis=1)

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
one_key = "b3c39"
plt.plot(bat_dict[one_key]["summary"]["QD"], color="green")
len(bat_dict[one_key]["summary"]["QD"])

# %%
sns.stripplot([1, 2, 3], orient="h")
sns.stripplot([1, 2, 3], orient="h")
sns.stripplot([1, 2, 3], orient="h")
