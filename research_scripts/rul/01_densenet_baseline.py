# %% [markdown]
# # Consolidated research script
#
# Method group **G12**: DenseNet RUL baseline. Architecture: DenseNet CNN. Method tags: RUL regression.
#
# This copy preserves the research implementation while removing saved outputs, execution counters, pinned GPU selection, and scratch-only cells. Closely related variants are summarized in the research script index. During cleanup, 0 syntactically invalid scratch cell(s) and 3 display-only scratch cell(s) were omitted.

# %%
from battery_soh.data import resolve_data_dir
from battery_soh.evaluation import MAPE, RMSPE
from battery_soh.paths import artifact_path
from battery_soh.preprocessing import interpolate_timeseries
from battery_soh.raw_data import load_battery_dictionary
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
batch_keys = [*bat_dict.keys()]
selected_keys = []

for battery_key in batch_keys:
    if (len(bat_dict[battery_key]["summary"]["cycle"]) > 250) & (
        len(bat_dict[battery_key]["summary"]["cycle"]) < 1200
    ):
        for k in range(1, len(bat_dict[battery_key]["summary"]["QD"])):
            QD = bat_dict[battery_key]["summary"]["QD"][k]
            temp_SOH = (QD / 1.1) * 100
            if temp_SOH < 80.2:
                selected_keys.append(battery_key)
                break

# %%
len(selected_keys)

# %%
random.shuffle(selected_keys)

# %%
train_bat_key = selected_keys[:90]
val_bat_key = selected_keys[90:100]
test_bat_key = selected_keys[100:]

# %%
time_value = 5
inter_value = 100

# %%
whole_data = []
RUL_true = []

for battery_key in train_bat_key:
    temporal_data_CI = []
    temporal_data_CV = []
    temporal_data_CT = []
    temporal_data_DV = []

    for k in range(1, len(bat_dict[battery_key]["summary"]["QD"])):
        cycle_str = str(k)
        QD = bat_dict[battery_key]["summary"]["QD"][int(cycle_str)]
        temp_SOH = (QD / 1.1) * 100
        if temp_SOH < 80.2:
            RUL_true.append(int(cycle_str))
            break

    for cycle_num in range(1, 102):
        cycle_num = str(cycle_num)

        data_length = len(bat_dict[battery_key]["cycles"][cycle_num]["Qc"])

        if cycle_num != "1":
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
                    Voltage_data = bat_dict[battery_key]["cycles"][cycle_num]["V"][
                        i : i + final_idx
                    ]
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
RUL_true = np.array(RUL_true)

# %%
RUL_true = RUL_true / 1200

# %%
whole_data_tensor = torch.tensor(whole_data, dtype=torch.float32).to(DEVICE)
RUL_true_tensor = torch.tensor(RUL_true, dtype=torch.float32).to(DEVICE)


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
    torch.tensor(whole_data, dtype=torch.float32), torch.tensor(RUL_true, dtype=torch.float32)
)

data_loader = DataLoader(dataset, batch_size=5, shuffle=True)

# %%
whole_val_data = []
RUL_val = []

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
            RUL_val.append(int(cycle_str))
            break

    for cycle_num in range(1, 102):
        cycle_num = str(cycle_num)

        data_length = len(bat_dict[battery_key]["cycles"][cycle_num]["Qc"])

        if cycle_num != "1":
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
                    Voltage_data = bat_dict[battery_key]["cycles"][cycle_num]["V"][
                        i : i + final_idx
                    ]
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
RUL_val = np.array(RUL_val)
whole_val_data = torch.tensor(whole_val_data, dtype=torch.float32).to(DEVICE)
# RUL_val = torch.tensor(RUL_val, dtype=torch.float32)

# %%
RUL_val = RUL_val / 1200

# %%
whole_test_data = []
RUL_test = []

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
            RUL_test.append(int(cycle_str))
            break

    for cycle_num in range(1, 102):
        cycle_num = str(cycle_num)

        data_length = len(bat_dict[battery_key]["cycles"][cycle_num]["Qc"])

        if cycle_num != "1":
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
                    Voltage_data = bat_dict[battery_key]["cycles"][cycle_num]["V"][
                        i : i + final_idx
                    ]
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
RUL_test = np.array(RUL_test)
whole_test_data = torch.tensor(whole_test_data, dtype=torch.float32).to(DEVICE)
# RUL_val = torch.tensor(RUL_val, dtype=torch.float32)


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
        self.linear_f = nn.Linear(50, 1)

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
loss_fn = torch.nn.MSELoss()

loss_best = 1000

for epoch in range(0, 500):
    idx = 0
    for batch_X1, batch_y1 in data_loader:
        batch_X1 = batch_X1.to(DEVICE)
        batch_y1 = batch_y1.to(DEVICE)
        model.train()
        RUL_pred = model(batch_X1)
        loss = torch.sqrt(loss_fn(RUL_pred, batch_y1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()

        RUL_pred_val = model(whole_val_data)

        RUL_pred_val = RUL_pred_val.cpu().detach().numpy()

        RUL_val_loss = MAPE(np.array(RUL_val), RUL_pred_val)

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
            torch.save(model.state_dict(), artifact_path("RUL_prediction_3"))

        idx += 1

# %%
RUL_true_origin = np.array(RUL_true * 1200)

# %%
model_new = DenseNet_121().to(DEVICE)
model_new.load_state_dict(torch.load(artifact_path("RUL_prediction_3")))
model_new.eval()

RUL_train_pred = model_new(whole_data_tensor).cpu().detach().numpy()
RUL_train_pred = RUL_train_pred * 1200
RUL_val_pred = model_new(whole_val_data).cpu().detach().numpy()
RUL_val_pred = RUL_val_pred * 1200
RUL_test_pred = model_new(whole_test_data).cpu().detach().numpy()
RUL_test_pred = RUL_test_pred * 1200

# %%
print(MAPE(RUL_true * 1200, RUL_train_pred))
print(MAPE(RUL_val * 1200, RUL_val_pred))
print(MAPE(RUL_test, RUL_test_pred))

# %%
P = np.linspace(1, 1200, 1200)

# %%
fig = plt.figure(figsize=(5, 5))
plt.plot(P, P, color="black")

plt.scatter(RUL_train_pred, RUL_true * 1200, color="blue", label="train")
plt.scatter(RUL_val_pred, RUL_val * 1200, color="green", label="val")
plt.scatter(RUL_test_pred, RUL_test, color="red", label="test")
plt.xlabel("Real EOL", fontsize=15)
plt.ylabel("Predicted EOL", fontsize=15)
plt.legend()
