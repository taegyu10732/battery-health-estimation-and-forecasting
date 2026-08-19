# %% [markdown]
# # Consolidated research script
#
# Method group **G26**: DenseNet physics-loss comparison. Architecture: DenseNet CNN. Method tags: physics loss|data-only ablation.
#
# This copy preserves the research implementation while removing saved outputs, execution counters, pinned GPU selection, and scratch-only cells. Closely related variants are summarized in the research script index. During cleanup, 0 syntactically invalid scratch cell(s) and 1 display-only scratch cell(s) were omitted.

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
random.seed(19)

# %%
batch_keys = [*bat_dict.keys()]
selected_keys = []

for battery_key in batch_keys:
    if (len(bat_dict[battery_key]["summary"]["cycle"]) > 250) & (
        len(bat_dict[battery_key]["summary"]["cycle"]) < 1500
    ):
        for k in range(1, len(bat_dict[battery_key]["summary"]["QD"])):
            QD = bat_dict[battery_key]["summary"]["QD"][k]
            temp_SOH = (QD / 1.1) * 100
            if temp_SOH < 80.2:
                selected_keys.append(battery_key)
                break

# %%
selected_keys.remove("b1c7")
selected_keys.remove("b2c20")
selected_keys.remove("b2c43")
selected_keys.remove("b2c25")
selected_keys.remove("b2c22")
selected_keys.remove("b2c38")
selected_keys.remove("b2c16")
selected_keys.remove("b2c6")
selected_keys.remove("b2c27")
selected_keys.remove("b2c13")
selected_keys.remove("b2c3")
selected_keys.remove("b2c1")
selected_keys.remove("b2c2")
selected_keys.remove("b2c15")

# %%
len(selected_keys)

# %%
random.shuffle(selected_keys)

# %%
train_bat_key = selected_keys[:80]
val_bat_key = selected_keys[80:90]
test_bat_key = selected_keys[90:]

# %%
len(train_bat_key)

# %%
time_value = 5
inter_value = 100

# %%
whole_data = []
RUL_true = []
Cap_true = []

for battery_key in train_bat_key:
    temporal_data_CI = []
    temporal_data_CV = []
    temporal_data_CT = []
    temporal_data_DV = []

    indi_94 = True
    indi_92 = True
    indi_90 = True
    indi_88 = True
    indi_86 = True
    indi_84 = True
    indi_82 = True
    indi_80 = True

    cycle_list = []
    cap_list = [bat_dict[battery_key]["summary"]["QD"][0]]

    for k in range(0, len(bat_dict[battery_key]["summary"]["QD"])):
        cycle_str = int(k)
        QD = bat_dict[battery_key]["summary"]["QD"][int(cycle_str)]
        temp_SOH = (QD / 1.1) * 100
        cycle_str = cycle_str + 1

        if (temp_SOH < 94.1) & indi_94:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_94 = False
        if (temp_SOH < 92.1) & indi_92:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_92 = False
        if (temp_SOH < 90.1) & indi_90:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_90 = False
        if (temp_SOH < 88.1) & indi_88:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_88 = False
        if (temp_SOH < 86.2) & indi_86:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_86 = False
        if (temp_SOH < 84.2) & indi_84:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_84 = False
        if (temp_SOH < 82.2) & indi_82:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_82 = False
        if (temp_SOH < 80.2) & indi_80:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_80 = False

    RUL_true.append(cycle_list)
    Cap_true.append(cap_list)

    for cycle_num in range(0, 101):
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
Cap_true = np.array(Cap_true)

# %%
RUL_true = RUL_true / 1400

# %%
whole_data_tensor = torch.tensor(whole_data, dtype=torch.float32).to(DEVICE)
RUL_true_tensor = torch.tensor(RUL_true, dtype=torch.float32).to(DEVICE)
Cap_true_tensor = torch.tensor(Cap_true, dtype=torch.float32).to(DEVICE)


# %%
class MyDataset(Dataset):
    def __init__(self, X1, y1, y2):
        self.X1 = X1
        self.y1 = y1
        self.y2 = y2

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        return self.X1[idx], self.y1[idx], self.y2[idx]


dataset = MyDataset(
    torch.tensor(whole_data, dtype=torch.float32),
    torch.tensor(RUL_true, dtype=torch.float32),
    torch.tensor(Cap_true, dtype=torch.float32),
)

data_loader = DataLoader(dataset, batch_size=20, shuffle=True)

# %%
whole_val_data = []
RUL_val = []
Cap_val = []

for battery_key in val_bat_key:
    temporal_data_CI = []
    temporal_data_CV = []
    temporal_data_CT = []
    temporal_data_DV = []

    indi_94 = True
    indi_92 = True
    indi_90 = True
    indi_88 = True
    indi_86 = True
    indi_84 = True
    indi_82 = True
    indi_80 = True

    cycle_list = []
    cap_list = [bat_dict[battery_key]["summary"]["QD"][0]]

    for k in range(0, len(bat_dict[battery_key]["summary"]["QD"])):
        cycle_str = int(k)
        QD = bat_dict[battery_key]["summary"]["QD"][int(cycle_str)]
        temp_SOH = (QD / 1.1) * 100
        cycle_str = cycle_str + 1

        if (temp_SOH < 94.1) & indi_94:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_94 = False
        if (temp_SOH < 92.1) & indi_92:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_92 = False
        if (temp_SOH < 90.1) & indi_90:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_90 = False
        if (temp_SOH < 88.1) & indi_88:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_88 = False
        if (temp_SOH < 86.2) & indi_86:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_86 = False
        if (temp_SOH < 84.2) & indi_84:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_84 = False
        if (temp_SOH < 82.2) & indi_82:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_82 = False
        if (temp_SOH < 80.2) & indi_80:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_80 = False

    RUL_val.append(cycle_list)
    Cap_val.append(cap_list)

    for cycle_num in range(0, 101):
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
Cap_val = np.array(Cap_val)
whole_val_data = torch.tensor(whole_val_data, dtype=torch.float32).to(DEVICE)
RUL_val = torch.tensor(RUL_val, dtype=torch.float32).to(DEVICE)
Cap_val = torch.tensor(Cap_val, dtype=torch.float32).to(DEVICE)

# %%
RUL_val = RUL_val / 1400

# %%
whole_test_data = []
RUL_test = []
Cap_test = []

for battery_key in test_bat_key:
    temporal_data_CI = []
    temporal_data_CV = []
    temporal_data_CT = []
    temporal_data_DV = []

    indi_94 = True
    indi_92 = True
    indi_90 = True
    indi_88 = True
    indi_86 = True
    indi_84 = True
    indi_82 = True
    indi_80 = True

    cycle_list = []
    cap_list = [bat_dict[battery_key]["summary"]["QD"][0]]

    for k in range(0, len(bat_dict[battery_key]["summary"]["QD"])):
        cycle_str = int(k)
        QD = bat_dict[battery_key]["summary"]["QD"][int(cycle_str)]
        temp_SOH = (QD / 1.1) * 100
        cycle_str = cycle_str + 1

        if (temp_SOH < 94.1) & indi_94:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_94 = False
        if (temp_SOH < 92.1) & indi_92:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_92 = False
        if (temp_SOH < 90.1) & indi_90:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_90 = False
        if (temp_SOH < 88.1) & indi_88:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_88 = False
        if (temp_SOH < 86.2) & indi_86:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_86 = False
        if (temp_SOH < 84.2) & indi_84:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_84 = False
        if (temp_SOH < 82.2) & indi_82:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_82 = False
        if (temp_SOH < 80.2) & indi_80:
            cycle_list.append(int(cycle_str))
            cap_list.append(QD)
            indi_80 = False

    RUL_test.append(cycle_list)
    Cap_test.append(cap_list)

    for cycle_num in range(0, 101):
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
Cap_test = np.array(Cap_test)
whole_test_data = torch.tensor(whole_test_data, dtype=torch.float32).to(DEVICE)
# RUL_test = torch.tensor(RUL_, dtype=torch.float32)


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
        self.linear_cycle = nn.Linear(50, 8)
        self.linear_param = nn.Linear(50, 4)

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
        cycle = self.linear_cycle(x)
        param = self.linear_param(x)

        return cycle.squeeze(1), param.squeeze(1)

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
def mape_loss(preds, target):
    epsilon = 1e-8  # Small value to avoid division by zero
    return torch.mean(torch.abs((preds - target) / (target + epsilon))) * 100


# %%
model = DenseNet_121().to(DEVICE)
model = model.train()

lam_1 = torch.tensor([0.01]).to(DEVICE)
lam_2 = torch.tensor([1.0]).to(DEVICE)

lam_1 = torch.nn.Parameter(lam_1)
lam_2 = torch.nn.Parameter(lam_2)

model.register_parameter("lam_1", lam_1)
model.register_parameter("lam_2", lam_2)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss()

loss_best = 1000

for epoch in range(0, 500):
    # print(lam_1, lam_2)
    idx = 0
    for batch_X1, batch_y1, batch_y2 in data_loader:
        batch_X1 = batch_X1.to(DEVICE)
        batch_y1 = batch_y1.to(DEVICE)
        batch_y2 = batch_y2.to(DEVICE)
        model.train()
        Cycle_pred, Param_pred = model(batch_X1)
        cycle_loss = torch.sqrt(mape_loss(Cycle_pred, batch_y1))
        Cycle_integrated = torch.cat(
            ((torch.ones(Cycle_pred.shape[0], 1) / 1400).to(DEVICE), Cycle_pred), dim=1
        )
        par_a = Param_pred[:, 0:1]
        par_b = Param_pred[:, 1:2]
        par_c = Param_pred[:, 2:3]
        par_d = Param_pred[:, 3:4]
        # physics_term = par_a*torch.exp(Cycle_integrated * par_b) + par_c*torch.exp(Cycle_integrated * par_d)
        physics_term = (
            par_a * (Cycle_integrated**3)
            + par_b * (Cycle_integrated**2)
            + par_c * (Cycle_integrated**2)
            + par_d
        )
        physics_loss = torch.sqrt(loss_fn(physics_term, batch_y2))

        loss = lam_1 * cycle_loss + lam_2 * physics_loss - torch.log(lam_1 * lam_2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print('epoch : ', epoch ,'/idx : ', idx , '/Cycle_loss :', round(float(cycle_loss),4), '/physics_loss :', round(float(physics_loss),4))

        model.eval()

        RUL_pred_val, _ = model(whole_val_data)

        RUL_pred_val = RUL_pred_val.cpu().detach().numpy()

        RUL_val_loss = MAPE(np.array(RUL_val.cpu()), RUL_pred_val)

        # print('epoch : ', epoch ,'/val_loss :', round(float(RUL_val_loss),4), '/Cycle_loss :', round(float(cycle_loss),4), '/physics_loss :', round(float(physics_loss),4), )

        if RUL_val_loss < loss_best:
            loss_best = RUL_val_loss
            torch.save(model.state_dict(), artifact_path("physics_prediction_4"))
            print(
                "epoch : ",
                epoch,
                "/val_loss :",
                round(float(RUL_val_loss), 4),
                "/Cycle_loss :",
                round(float(cycle_loss), 4),
                "/physics_loss :",
                round(float(physics_loss), 4),
            )

        idx += 1

# %%
model_new = DenseNet_121().to(DEVICE)
model_new.register_parameter("lam_1", lam_1)
model_new.register_parameter("lam_2", lam_2)
model_new.load_state_dict(torch.load(artifact_path("physics_prediction_4")))
model_new.eval()

RUL_train_pred, _ = model_new(whole_data_tensor)
RUL_train_pred = RUL_train_pred.cpu().detach().numpy()
RUL_train_pred = RUL_train_pred * 1400
RUL_val_pred, _ = model_new(whole_val_data)
RUL_val_pred = RUL_val_pred.cpu().detach().numpy()
RUL_val_pred = RUL_val_pred * 1400
RUL_test_pred, _ = model_new(whole_test_data)
RUL_test_pred = RUL_test_pred.cpu().detach().numpy()
RUL_test_pred = RUL_test_pred * 1400

# %%
print(MAPE(RUL_true * 1400, RUL_train_pred))
print(MAPE(RUL_val.cpu().detach().numpy() * 1400, RUL_val_pred))
print(MAPE(RUL_test, RUL_test_pred))

# %%
for i in range(0, len(RUL_test)):
    A = MAPE(RUL_test[i], RUL_test_pred[i])
    if A > 10:
        print(i)

# %%
print(MAPE(RUL_true[:, -1] * 1400, RUL_train_pred[:, -1]))
print(MAPE(RUL_val[:, -1].cpu().detach().numpy() * 1400, RUL_val_pred[:, -1]))
print(MAPE(RUL_test[:, -1], RUL_test_pred[:, -1]))

# %%
P = np.linspace(1, 1400, 1400)

# %%
fig = plt.figure(figsize=(5, 5))
plt.plot(P, P, color="black")

plt.scatter(RUL_train_pred[:, -1], RUL_true[:, -1] * 1400, color="blue", label="train")
plt.scatter(
    RUL_val_pred[:, -1], RUL_val[:, -1].cpu().detach().numpy() * 1400, color="green", label="val"
)
plt.scatter(RUL_test_pred[:, -1], RUL_test[:, -1], color="red", label="test")
plt.xlabel("Real EOL", fontsize=15)
plt.ylabel("Predicted EOL", fontsize=15)
plt.legend()

# %%
fig = plt.figure(figsize=(5, 5))
plt.plot(P, P, color="black")

plt.scatter(RUL_train_pred, RUL_true * 1400, color="blue", label="train")
plt.scatter(RUL_val_pred, RUL_val.cpu().detach().numpy() * 1400, color="green", label="val")
plt.scatter(RUL_test_pred, RUL_test, color="red", label="test")
plt.xlabel("Real EOL", fontsize=15)
plt.ylabel("Predicted EOL", fontsize=15)
plt.legend()

# %%
index = 9

plt.plot(np.append(index, RUL_test_pred[index]), Cap_test[index])
plt.scatter(np.append(0, RUL_test_pred[index]), Cap_test[index])
plt.plot(np.append(0, RUL_test[index]), Cap_test[index])
plt.scatter(np.append(0, RUL_test[index]), Cap_test[index])
data_length = len(bat_dict[test_bat_key[index]]["summary"]["QD"])
X = np.linspace(1, data_length, data_length, dtype=int)
plt.plot(X, bat_dict[test_bat_key[index]]["summary"]["QD"])

# fitted_function = test_param_pred[index][0] * (X**3) + test_param_pred[index][1] * (X**2) + test_param_pred[index][2] * X + test_param_pred[index][3]
# plt.plot(X, fitted_function)

plt.ylim([0.87, 1.1])
