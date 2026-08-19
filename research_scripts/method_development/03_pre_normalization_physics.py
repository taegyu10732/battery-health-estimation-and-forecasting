# %% [markdown]
# # Consolidated research script
#
# Method group **G34**: Pre-normalization physics estimator. Architecture: CNN-Transformer encoder. Method tags: physics parameters|180-cycle.
#
# This copy preserves the research implementation while removing saved outputs, execution counters, pinned GPU selection, and scratch-only cells. Closely related variants are summarized in the research script index. During cleanup, 0 syntactically invalid scratch cell(s) and 1 display-only scratch cell(s) were omitted.

# %%
from battery_soh.data import resolve_data_dir
from battery_soh.evaluation import MAPE
from battery_soh.paths import artifact_path
from battery_soh.preprocessing import interpolate_timeseries
from battery_soh.raw_data import load_battery_dictionary
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

# %%
# Shared, portable raw-data loading. This may require substantial memory.
RESEARCH_BATCHES = ("b1", "b2")
bat_dict = load_battery_dictionary(batches=RESEARCH_BATCHES)

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
train_bat_key = [
    "b1c10",
    "b1c20",
    "b1c14",
    "b1c22",
    "b1c16",
    "b1c23",
    "b1c24",
    "b1c18",
    "b1c11",
    "b1c32",
    "b1c26",
    "b1c33",
    "b1c34",
    "b1c27",
    "b1c36",
    "b1c40",
    "b1c37",
    "b1c38",
    "b1c41",
    "b1c42",
    "b1c43",
    "b1c44",
    "b1c46",
    "b2c10",
    "b2c14",
    "b2c31",
    "b2c32",
    "b2c37",
    "b2c39",
    "b2c41",
    "b2c44",
    "b2c45",
    "b2c46",
    "b2c47",
]
val_bat_key = ["b1c28", "b2c34", "b1c15", "b2c19"]
test_bat_key = [
    "b1c9",
    "b1c12",
    "b1c21",
    "b1c29",
    "b1c39",
    "b1c45",
    "b2c18",
    "b2c30",
    "b2c42",
    "b2c21",
]

# %%
time_value = 3
inter_value = 180

# %%
whole_data = []
Cycle_true = []
SOH_true = []

for battery_key in train_bat_key:
    for cycle_num in range(1, len(bat_dict[battery_key]["summary"]["cycle"])):
        temporal_data = []
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
                    Current_data = bat_dict[battery_key]["cycles"][cycle_num]["I"][
                        i : i + final_idx
                    ]
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
                    Voltage_data = interpolate_timeseries(data[2], inter_value).reshape(-1, 1)
                    Temperature_data = interpolate_timeseries(
                        Temperature_data, inter_value
                    ).reshape(-1, 1)

                    scaler = MinMaxScaler((1e-6, 1))
                    Voltage_data = scaler.fit_transform(Voltage_data).reshape(-1)
                    Temperature_data = scaler.fit_transform(Temperature_data).reshape(-1)

                    temporal_data.append(Current_data)
                    temporal_data.append(Voltage_data)
                    temporal_data.append(Temperature_data)

                    # if (end_time - start_time) < 5.1:
                    # print(battery_key + ' '+ str(cycle_num))
                    # print('Charge_time : ' + str(end_time - start_time))
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
                    scaler = MinMaxScaler((1e-6, 1))
                    Discharge_Voltage = interpolate_timeseries(data[1], inter_value).reshape(-1, 1)
                    Discharge_Voltage = scaler.fit_transform(Discharge_Voltage).reshape(-1)

                    temporal_data.append(Discharge_Voltage)

                    break
            max_cycle = int(bat_dict[battery_key]["summary"]["cycle"][-1])
            Cycle_true.append(bat_dict[battery_key]["summary"]["cycle"][int(cycle_num)] / max_cycle)
            SOH_true.append(bat_dict[battery_key]["summary"]["QD"][int(cycle_num)])
            whole_data.append(np.array(temporal_data))

for i in range(len(whole_data) - 1, 0, -1):
    if whole_data[i].shape != (4, inter_value):
        del whole_data[i]
        del Cycle_true[i]
        del SOH_true[i]

whole_data = np.array(whole_data)
Cycle_true = np.array(Cycle_true)
SOH_true = np.array(SOH_true)

whole_data.shape

# %%
minus_data = []


for battery_key in train_bat_key:
    first_data = []

    for cycle_num in range(1, len(bat_dict[battery_key]["summary"]["cycle"])):
        temporal_data = []
        cycle_num = str(cycle_num)

        data_length = len(bat_dict[battery_key]["cycles"][cycle_num]["Qc"])

        if cycle_num == "1":
            for i in range(11, len(bat_dict[battery_key]["cycles"][cycle_num]["Qc"])):
                Charge_SOC = (bat_dict[battery_key]["cycles"][cycle_num]["Qc"][i] / 1.1) * 100
                if Charge_SOC > 80.1:
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
                    Current_data = bat_dict[battery_key]["cycles"][cycle_num]["I"][
                        i : i + final_idx
                    ]
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

                    first_data.append(Current_data)
                    first_data.append(Voltage_data)
                    first_data.append(Temperature_data)
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

                    first_data.append(Discharge_Voltage)

                    break

        else:
            for i in range(11, len(bat_dict[battery_key]["cycles"][cycle_num]["Qc"])):
                Charge_SOC = (bat_dict[battery_key]["cycles"][cycle_num]["Qc"][i] / 1.1) * 100
                if Charge_SOC > 80.1:
                    for p in range(data_length):
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
                    Current_data = bat_dict[battery_key]["cycles"][cycle_num]["I"][
                        i : i + final_idx
                    ]
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

                    Current_data = interpolate_timeseries(data[1], inter_value) - first_data[0]
                    Voltage_data = interpolate_timeseries(data[2], inter_value) - first_data[1]
                    Temperature_data = (
                        interpolate_timeseries(Temperature_data, inter_value) - first_data[2]
                    )

                    temporal_data.append(Current_data)
                    temporal_data.append(Voltage_data)
                    temporal_data.append(Temperature_data)
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
                    Discharge_Voltage = interpolate_timeseries(data[1], inter_value) - first_data[3]

                    temporal_data.append(Discharge_Voltage)

                    break
            minus_data.append(np.array(temporal_data))

for i in range(len(minus_data) - 1, 0, -1):
    if minus_data[i].shape != (4, inter_value):
        del minus_data[i]

minus_data = np.array(minus_data)

minus_data.shape

# %%
whole_data = np.reshape(
    whole_data, (whole_data.shape[0], 1, whole_data.shape[1], whole_data.shape[2])
)
minus_data = np.reshape(
    minus_data, (minus_data.shape[0], 1, minus_data.shape[1], minus_data.shape[2])
)
print(whole_data.shape)
print(minus_data.shape)


# %%
class MyDataset(Dataset):
    def __init__(self, X1, X2, y1, y2):
        self.X1 = X1
        self.X2 = X2
        self.y1 = y1
        self.y2 = y2

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y1[idx], self.y2[idx]


dataset = MyDataset(
    torch.tensor(whole_data, dtype=torch.float32),
    torch.tensor(minus_data, dtype=torch.float32),
    torch.tensor(SOH_true, dtype=torch.float32),
    torch.tensor(Cycle_true, dtype=torch.float32),
)

data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# %%
###############################
#####BASE MODEL ###############
###############################


class SOH_model(nn.Module):
    def __init__(
        self, in_features, time_length, first_head, first_layer, second_head, second_layer
    ):
        super().__init__()

        self.in_features = in_features
        self.time_length = time_length
        self.first_head = first_head
        self.first_layer = first_layer
        self.second_head = second_head
        self.second_layer = second_layer

        self.linear_f = nn.Linear(self.time_length, 100)

        self.conv_1 = nn.Conv2d(1, 10, kernel_size=(1, 10), padding="same")
        self.conv_2 = nn.Conv2d(10, 10, kernel_size=(1, 10), padding="same")
        self.conv_3 = nn.Conv2d(10, 20, kernel_size=(1, 10), padding="same")
        self.conv_4 = nn.Conv2d(20, 10, kernel_size=(1, 5), padding="same")
        self.conv_5 = nn.Conv2d(10, 1, kernel_size=(1, 5), padding="same")
        self.conv_6 = nn.Conv2d(1, 1, kernel_size=(4, 4), padding="same")

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=100, nhead=self.first_head, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=self.first_layer
        )

        self.linear1 = nn.Linear(100 * self.in_features, 10)

        # --------------------------------------------------------

        self.linear_f_prime = nn.Linear(self.time_length, 100)

        self.conv_1_prime = nn.Conv2d(1, 10, kernel_size=(1, 10), padding="same")
        self.conv_2_prime = nn.Conv2d(10, 10, kernel_size=(1, 10), padding="same")
        self.conv_3_prime = nn.Conv2d(10, 20, kernel_size=(1, 10), padding="same")
        self.conv_4_prime = nn.Conv2d(20, 10, kernel_size=(1, 5), padding="same")
        self.conv_5_prime = nn.Conv2d(10, 1, kernel_size=(1, 5), padding="same")
        self.conv_6_prime = nn.Conv2d(
            1, 1, kernel_size=(self.in_features, self.in_features), padding="same"
        )

        self.encoder_layer_prime = nn.TransformerEncoderLayer(
            d_model=100, nhead=self.second_head, batch_first=True
        )
        self.encoder_prime = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer_prime, num_layers=self.second_layer
        )

        self.linear1_prime = nn.Linear(100 * self.in_features, 10)

        self.final_reggression1 = nn.Linear(20, 2)

    def forward(self, x, p):
        x = self.linear_f(x)
        # print(x.shape)
        x1 = x
        x = self.conv_1(x)
        x = F.leaky_relu(x)
        x = self.conv_2(x)
        x = F.leaky_relu(x)
        x = self.conv_3(x)
        x = F.leaky_relu(x)
        x = self.conv_4(x)
        x = F.leaky_relu(x)
        x = self.conv_5(x)
        x = F.leaky_relu(x)
        x = x + x1
        x = self.conv_6(x)
        x = F.leaky_relu(x)

        x = x.squeeze(1)

        x = self.encoder(x)

        x = x.reshape(-1, 400)

        x = self.linear1(x)

        # -------------------------------------------------------

        p = self.linear_f(p)
        # print(x.shape)
        p1 = p
        p = self.conv_1_prime(p)
        p = F.leaky_relu(p)
        p = self.conv_2_prime(p)
        p = F.leaky_relu(p)
        p = self.conv_3_prime(p)
        p = F.leaky_relu(p)
        p = self.conv_4_prime(p)
        p = F.leaky_relu(p)
        p = self.conv_5_prime(p)
        p = F.leaky_relu(p)
        p = p + p1
        p = self.conv_6_prime(p)
        p = F.leaky_relu(p)

        p = p.squeeze(1)

        p = self.encoder_prime(p)
        # print(x[0].squeeze(0).shape)

        p = p.reshape(-1, 400)

        p = self.linear1_prime(p)

        # ---------------------------------------------------------------

        final = torch.cat((x, p), 1)

        final = self.final_reggression1(final)

        SOH_final = final[:, 0:1]
        Cycle_final = final[:, 1:2]

        return SOH_final.squeeze(1), Cycle_final.squeeze(1)


# %%
whole_val_data = []
Cycle_val = []
SOH_val = []

for battery_key in val_bat_key:
    for cycle_num in range(2, len(bat_dict[battery_key]["summary"]["cycle"])):
        temporal_data = []
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
                    Current_data = bat_dict[battery_key]["cycles"][cycle_num]["I"][
                        i : i + final_idx
                    ]
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
                    Voltage_data = interpolate_timeseries(data[2], inter_value).reshape(-1, 1)
                    Temperature_data = interpolate_timeseries(
                        Temperature_data, inter_value
                    ).reshape(-1, 1)

                    scaler = MinMaxScaler((1e-6, 1))
                    Voltage_data = scaler.fit_transform(Voltage_data).reshape(-1)
                    Temperature_data = scaler.fit_transform(Temperature_data).reshape(-1)

                    temporal_data.append(Current_data)
                    temporal_data.append(Voltage_data)
                    temporal_data.append(Temperature_data)

                    # if (end_time - start_time) < 5.1:
                    # print(battery_key + ' '+ str(cycle_num))
                    # print('Charge_time : ' + str(end_time - start_time))
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
                    scaler = MinMaxScaler((1e-6, 1))
                    Discharge_Voltage = interpolate_timeseries(data[1], inter_value).reshape(-1, 1)
                    Discharge_Voltage = scaler.fit_transform(Discharge_Voltage).reshape(-1)

                    temporal_data.append(Discharge_Voltage)

                    break
            max_cycle = int(bat_dict[battery_key]["summary"]["cycle"][-1])
            Cycle_val.append(bat_dict[battery_key]["summary"]["cycle"][int(cycle_num)] / max_cycle)
            SOH_val.append(bat_dict[battery_key]["summary"]["QD"][int(cycle_num)])
            whole_val_data.append(np.array(temporal_data))

for i in range(len(whole_val_data) - 1, 0, -1):
    if whole_val_data[i].shape != (4, inter_value):
        del whole_val_data[i]
        del Cycle_val[i]
        del SOH_val[i]

whole_val_data = np.array(whole_val_data)
Cycle_val = np.array(Cycle_val)
SOH_val = np.array(SOH_val)

whole_val_data = np.reshape(
    whole_val_data, (whole_val_data.shape[0], 1, whole_val_data.shape[1], whole_val_data.shape[2])
)

whole_val_data = torch.tensor(whole_val_data, dtype=torch.float32)

# %%
minus_val_data = []

for battery_key in val_bat_key:
    first_data = []

    for cycle_num in range(1, len(bat_dict[battery_key]["summary"]["cycle"])):
        temporal_data = []
        cycle_num = str(cycle_num)

        data_length = len(bat_dict[battery_key]["cycles"][cycle_num]["Qc"])

        if cycle_num == "1":
            for i in range(11, len(bat_dict[battery_key]["cycles"][cycle_num]["Qc"])):
                Charge_SOC = (bat_dict[battery_key]["cycles"][cycle_num]["Qc"][i] / 1.1) * 100
                if Charge_SOC > 80.1:
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
                    Current_data = bat_dict[battery_key]["cycles"][cycle_num]["I"][
                        i : i + final_idx
                    ]
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

                    first_data.append(Current_data)
                    first_data.append(Voltage_data)
                    first_data.append(Temperature_data)
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

                    first_data.append(Discharge_Voltage)

                    break

        elif int(cycle_num) >= 2:
            for i in range(11, len(bat_dict[battery_key]["cycles"][cycle_num]["Qc"])):
                Charge_SOC = (bat_dict[battery_key]["cycles"][cycle_num]["Qc"][i] / 1.1) * 100
                if Charge_SOC > 80.1:
                    for p in range(data_length):
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
                    Current_data = bat_dict[battery_key]["cycles"][cycle_num]["I"][
                        i : i + final_idx
                    ]
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

                    Current_data = interpolate_timeseries(data[1], inter_value) - first_data[0]
                    Voltage_data = interpolate_timeseries(data[2], inter_value) - first_data[1]
                    Temperature_data = (
                        interpolate_timeseries(Temperature_data, inter_value) - first_data[2]
                    )

                    temporal_data.append(Current_data)
                    temporal_data.append(Voltage_data)
                    temporal_data.append(Temperature_data)
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
                    Discharge_Voltage = interpolate_timeseries(data[1], inter_value) - first_data[3]

                    temporal_data.append(Discharge_Voltage)

                    break

            minus_val_data.append(np.array(temporal_data))

for i in range(len(minus_val_data) - 1, 0, -1):
    if minus_val_data[i].shape != (4, inter_value):
        del minus_val_data[i]

minus_val_data = np.array(minus_val_data)


minus_val_data = np.reshape(
    minus_val_data, (minus_val_data.shape[0], 1, minus_val_data.shape[1], minus_val_data.shape[2])
)

minus_val_data = torch.tensor(minus_val_data, dtype=torch.float32)

# %%
print(minus_val_data.shape)
print(whole_val_data.shape)
whole_val_data = whole_val_data.to(DEVICE)
minus_val_data = minus_val_data.to(DEVICE)

# %%
global global_best

global_best = 10000


# %%
def objective(trial):
    # Define the hyperparameters to optimize
    first_head = trial.suggest_categorical("first_head", [1, 2, 4, 5, 10, 20])
    first_layer = trial.suggest_int("first_layer", 2, 10)
    second_head = trial.suggest_categorical("second_head", [1, 2, 4, 5, 10, 20])
    second_layer = trial.suggest_int("second_layer", 2, 10)

    model = SOH(4, inter_value, first_head, first_layer, second_head, second_layer).to(DEVICE)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    val_loss_best = 10000

    for epoch in range(0, 50):
        idx = 0
        for batch_X1, batch_X2, batch_y in data_loader:
            model.train()
            batch_X1 = batch_X1.to(DEVICE)
            batch_X2 = batch_X2.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            predictions = model(batch_X1, batch_X2)
            loss = torch.sqrt(loss_fn(predictions, batch_y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if torch.isnan(loss):
                raise optuna.TrialPruned()

            model.eval()

            predict = model(whole_val_data, minus_val_data).cpu().detach().numpy()
            real_y = np.array(y_val)

            val_loss = mean_squared_error(real_y, predict) ** 0.5

            if val_loss < val_loss_best:
                val_loss_best = val_loss

    if not np.isnan(val_loss_best):
        return val_loss_best
    else:
        return float("inf")


# %%
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# %%
model = SOH_model(4, inter_value, 4, 2, 1, 9).to(DEVICE)
model = model.train()

par_a = torch.tensor([0.0]).to(DEVICE)
par_b = torch.tensor([0.0]).to(DEVICE)
par_c = torch.tensor([0.0]).to(DEVICE)
par_d = torch.tensor([0.0]).to(DEVICE)


par_a = torch.nn.Parameter(par_a)
par_b = torch.nn.Parameter(par_b)
par_c = torch.nn.Parameter(par_c)
par_d = torch.nn.Parameter(par_d)

model.register_parameter("par_a", par_a)
model.register_parameter("par_b", par_b)
model.register_parameter("par_c", par_c)
model.register_parameter("par_d", par_d)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss()

SOH_loss_best = 1000
Cycle_loss_best = 1000


for epoch in range(0, 300):
    idx = 0
    print(
        round(float(par_a), 4),
        round(float(par_b), 4),
        round(float(par_c), 4),
        round(float(par_d), 4),
    )
    for batch_X1, batch_X2, batch_y1, batch_y2 in data_loader:
        model.train()
        batch_X1 = batch_X1.to(DEVICE)
        batch_X2 = batch_X2.to(DEVICE)
        batch_y1 = batch_y1.to(DEVICE)
        batch_y2 = batch_y2.to(DEVICE)
        SOH_pred, Cycle_pred = model(batch_X1, batch_X2)
        l1 = torch.sqrt(loss_fn(SOH_pred, batch_y1))
        l2 = torch.sqrt(loss_fn(Cycle_pred, batch_y2))
        physics = (
            SOH_pred - par_a * torch.exp(par_b * Cycle_pred) - par_c * torch.exp(par_d * Cycle_pred)
        )

        l3 = torch.sqrt(loss_fn(physics, torch.zeros(len(SOH_pred)).to(DEVICE)))

        loss = l1 * 10 + l2 + l3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()

        SOH_pred_val, Cycle_pred_val = model(whole_val_data, minus_val_data)

        SOH_pred_val = SOH_pred_val.cpu().detach().numpy()
        Cycle_pred_val = Cycle_pred_val.cpu().detach().numpy()

        SOH_val_loss = mean_squared_error(SOH_val, SOH_pred_val) ** 0.5
        Cycle_val_loss = mean_squared_error(Cycle_val, Cycle_pred_val) ** 0.5

        print(
            "epoch : ",
            epoch,
            "/idx : ",
            idx,
            "/l1_loss :",
            round(float(l1), 4),
            "/l2_loss :",
            round(float(l2), 4),
            "/l3_loss :",
            round(float(l3), 4),
            "/SOH_val_loss : ",
            round(SOH_val_loss, 4),
            "/Cycle_val_loss : ",
            round(Cycle_val_loss, 4),
        )

        if SOH_val_loss < SOH_loss_best:
            SOH_loss_best = SOH_val_loss
            torch.save(model.state_dict(), artifact_path("SOH_Cycle_physics_estimation_180"))

        if Cycle_val_loss < Cycle_loss_best:
            Cycle_loss_best = Cycle_val_loss
        idx += 1

# %%
rmse_list = []
mape_list = []

mode = "Cycle"

fig = plt.figure(figsize=(15, 15))

for k in range(len(test_bat_key) - 1):
    battery_list = [test_bat_key[k]]

    model_new = SOH_model(4, inter_value, 4, 2, 1, 9).to(DEVICE)
    par_a = torch.tensor([0.0]).to(DEVICE)
    par_b = torch.tensor([0.0]).to(DEVICE)
    par_c = torch.tensor([0.0]).to(DEVICE)
    par_d = torch.tensor([0.0]).to(DEVICE)

    par_a = torch.nn.Parameter(par_a)
    par_b = torch.nn.Parameter(par_b)
    par_c = torch.nn.Parameter(par_c)
    par_d = torch.nn.Parameter(par_d)

    model_new.register_parameter("par_a", par_a)
    model_new.register_parameter("par_b", par_b)
    model_new.register_parameter("par_c", par_c)
    model_new.register_parameter("par_d", par_d)

    model_new.load_state_dict(torch.load(artifact_path("SOH_Cycle_physics_estimation_180")))
    model_new.eval()

    whole_test_data = []
    Cycle_test = []
    SOH_test = []

    for battery_key in battery_list:
        for cycle_num in range(1, len(bat_dict[battery_key]["summary"]["cycle"])):
            temporal_data = []
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
                        Current_data = bat_dict[battery_key]["cycles"][cycle_num]["I"][
                            i : i + final_idx
                        ]
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
                        Voltage_data = interpolate_timeseries(data[2], inter_value).reshape(-1, 1)
                        Temperature_data = interpolate_timeseries(
                            Temperature_data, inter_value
                        ).reshape(-1, 1)

                        scaler = MinMaxScaler((1e-6, 1))
                        Voltage_data = scaler.fit_transform(Voltage_data).reshape(-1)
                        Temperature_data = scaler.fit_transform(Temperature_data).reshape(-1)

                        temporal_data.append(Current_data)
                        temporal_data.append(Voltage_data)
                        temporal_data.append(Temperature_data)

                        # if (end_time - start_time) < 5.1:
                        # print(battery_key + ' '+ str(cycle_num))
                        # print('Charge_time : ' + str(end_time - start_time))
                        break

                for i in range(11, len(bat_dict[battery_key]["cycles"][cycle_num]["Qd"])):
                    Discharge_SOC = (
                        bat_dict[battery_key]["cycles"][cycle_num]["Qd"][i] / 1.1
                    ) * 100
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
                        scaler = MinMaxScaler((1e-6, 1))
                        Discharge_Voltage = interpolate_timeseries(data[1], inter_value).reshape(
                            -1, 1
                        )
                        Discharge_Voltage = scaler.fit_transform(Discharge_Voltage).reshape(-1)

                        temporal_data.append(Discharge_Voltage)

                        break
                max_cycle = int(bat_dict[battery_key]["summary"]["cycle"][-1])
                Cycle_test.append(
                    bat_dict[battery_key]["summary"]["cycle"][int(cycle_num)] / max_cycle
                )
                SOH_test.append(bat_dict[battery_key]["summary"]["QD"][int(cycle_num)])
                whole_test_data.append(np.array(temporal_data))

    for i in range(len(whole_test_data) - 1, 0, -1):
        if whole_test_data[i].shape != (4, inter_value):
            del whole_test_data[i]
            del Cycle_test[i]
            del SOH_test[i]

    whole_test_data = np.array(whole_test_data)
    Cycle_test = np.array(Cycle_test)
    SOH_test = np.array(SOH_test)

    whole_test_data = np.reshape(
        whole_test_data,
        (whole_test_data.shape[0], 1, whole_test_data.shape[1], whole_test_data.shape[2]),
    )

    whole_test_data = torch.tensor(whole_test_data, dtype=torch.float32)

    test_data = []

    for battery_key in battery_list:
        first_data = []

        for cycle_num in range(1, len(bat_dict[battery_key]["summary"]["cycle"])):
            temporal_data = []
            cycle_num = str(cycle_num)

            data_length = len(bat_dict[battery_key]["cycles"][cycle_num]["Qc"])

            if cycle_num == "1":
                for i in range(11, len(bat_dict[battery_key]["cycles"][cycle_num]["Qc"])):
                    Charge_SOC = (bat_dict[battery_key]["cycles"][cycle_num]["Qc"][i] / 1.1) * 100
                    if Charge_SOC > 80.1:
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
                        Current_data = bat_dict[battery_key]["cycles"][cycle_num]["I"][
                            i : i + final_idx
                        ]
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

                        first_data.append(Current_data)
                        first_data.append(Voltage_data)
                        first_data.append(Temperature_data)
                        break

                for i in range(11, len(bat_dict[battery_key]["cycles"][cycle_num]["Qd"])):
                    Discharge_SOC = (
                        bat_dict[battery_key]["cycles"][cycle_num]["Qd"][i] / 1.1
                    ) * 100
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

                        first_data.append(Discharge_Voltage)

                        break

            else:
                for i in range(11, len(bat_dict[battery_key]["cycles"][cycle_num]["Qc"])):
                    Charge_SOC = (bat_dict[battery_key]["cycles"][cycle_num]["Qc"][i] / 1.1) * 100
                    if Charge_SOC > 80.1:
                        for p in range(data_length):
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
                        Current_data = bat_dict[battery_key]["cycles"][cycle_num]["I"][
                            i : i + final_idx
                        ]
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

                        Current_data = interpolate_timeseries(data[1], inter_value) - first_data[0]
                        Voltage_data = interpolate_timeseries(data[2], inter_value) - first_data[1]
                        Temperature_data = (
                            interpolate_timeseries(Temperature_data, inter_value) - first_data[2]
                        )

                        temporal_data.append(Current_data)
                        temporal_data.append(Voltage_data)
                        temporal_data.append(Temperature_data)
                        break

                for i in range(11, len(bat_dict[battery_key]["cycles"][cycle_num]["Qd"])):
                    Discharge_SOC = (
                        bat_dict[battery_key]["cycles"][cycle_num]["Qd"][i] / 1.1
                    ) * 100
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
                        Discharge_Voltage = (
                            interpolate_timeseries(data[1], inter_value) - first_data[3]
                        )

                        temporal_data.append(Discharge_Voltage)

                        break

                test_data.append(np.array(temporal_data))

    for i in range(len(test_data) - 1, 0, -1):
        if test_data[i].shape != (4, inter_value):
            del test_data[i]

    test_data = np.array(test_data)

    test_data = np.reshape(
        test_data, (test_data.shape[0], 1, test_data.shape[1], test_data.shape[2])
    )

    test_data = torch.tensor(test_data, dtype=torch.float32)

    model_new.eval()

    whole_test_data = whole_test_data.to(DEVICE)
    test_data = test_data.to(DEVICE)

    SOH_test_pred, Cycle_test_pred = model_new(whole_test_data, test_data)

    SOH_test_pred = SOH_test_pred.cpu().detach().numpy()
    Cycle_test_pred = Cycle_test_pred.cpu().detach().numpy()

    SOH_test = np.array(SOH_test)
    Cycle_test = np.array(Cycle_test)

    if battery_key == "b2c42":
        SOH_test_pred = np.delete(SOH_test_pred, [248])
        Cycle_test_pred = np.delete(Cycle_test_pred, [248])
        Cycle_test = np.delete(Cycle_test, [248])
        SOH_test = np.delete(SOH_test, [248])

    plot_num = 331 + k

    if mode == "SOH":
        soh_rmse = mean_squared_error(SOH_test, SOH_test_pred) ** 0.5
        soh_mape = MAPE(np.array(SOH_test), np.array(SOH_test_pred))

        rmse_list.append(soh_rmse)
        mape_list.append(soh_mape)

        plt.subplot(plot_num)
        plt.plot(SOH_test_pred, color="r", label="Estimated SOH")
        plt.plot(SOH_test, color="b", label="real SOH")
        plt.text(1, 0.95, "RMSE : " + str(round(soh_rmse, 5)), fontsize=20)

        plt.title(battery_list[0])
        plt.ylabel("Capacity (AH)", fontsize=15)
        plt.xlabel("cycle", fontsize=15)
        plt.legend()

    if mode == "Cycle":
        cycle_rmse = mean_squared_error(Cycle_test, Cycle_test_pred) ** 0.5
        cycle_mape = MAPE(np.array(Cycle_test), np.array(Cycle_test_pred))

        rmse_list.append(cycle_rmse)
        mape_list.append(cycle_mape)

        plt.subplot(plot_num)
        plt.plot(Cycle_test_pred, color="r", label="predict cycle")
        plt.plot(Cycle_test, color="b", label="real cycle")
        plt.text(1, 0.7, "RMSE : " + str(round(cycle_rmse, 5)), fontsize=20)

        plt.title(battery_list[0])
        plt.ylabel("Normalized Cycle", fontsize=15)
        plt.xlabel("cycle", fontsize=15)
        plt.legend()


fig.tight_layout(pad=3.0)
plt.show()

# %%
fig, ax = plt.subplots()

rmse_arr = np.array(rmse_list)
rmse_avg = np.mean(rmse_arr)

mape_arr = np.array(mape_list)
mape_avg = np.mean(mape_arr)

ax.boxplot([rmse_arr, mape_arr])
plt.xticks([1, 2], ["RMSE(%)", "MAPE"])
plt.text(0.6, 0.5, "[RMSE_AVG]\n   " + str(round(rmse_avg, 3)) + "%")
plt.text(1.6, 0.5, "[MAPE_AVG]\n   " + str(round(mape_avg, 3)) + "%")

plt.show

# %%
rmse_list = []
mape_list = []

fig = plt.figure(figsize=(20, 12))

for k in range(len(test_bat_key) - 1):
    battery_list = [test_bat_key[k]]

    model_new = SOH(4, inter_value, 1, 3, 20, 8).to(DEVICE)
    model_new.load_state_dict(torch.load(artifact_path("ensemble_transformer_270")))
    model_new.eval()

    whole_test_data = []
    y_test = []

    for battery_key in battery_list:
        for cycle_num in range(1, len(bat_dict[battery_key]["summary"]["cycle"])):
            temporal_data = []
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
                        Current_data = bat_dict[battery_key]["cycles"][cycle_num]["I"][
                            i : i + final_idx
                        ]
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
                        Voltage_data = interpolate_timeseries(data[2], inter_value).reshape(-1, 1)
                        Temperature_data = interpolate_timeseries(
                            Temperature_data, inter_value
                        ).reshape(-1, 1)

                        scaler = MinMaxScaler((1e-6, 1))
                        Voltage_data = scaler.fit_transform(Voltage_data).reshape(-1)
                        Temperature_data = scaler.fit_transform(Temperature_data).reshape(-1)

                        temporal_data.append(Current_data)
                        temporal_data.append(Voltage_data)
                        temporal_data.append(Temperature_data)

                        # if (end_time - start_time) < 5.1:
                        # print(battery_key + ' '+ str(cycle_num))
                        # print('Charge_time : ' + str(end_time - start_time))
                        break

                for i in range(11, len(bat_dict[battery_key]["cycles"][cycle_num]["Qd"])):
                    Discharge_SOC = (
                        bat_dict[battery_key]["cycles"][cycle_num]["Qd"][i] / 1.1
                    ) * 100
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
                        scaler = MinMaxScaler((1e-6, 1))
                        Discharge_Voltage = interpolate_timeseries(data[1], inter_value).reshape(
                            -1, 1
                        )
                        Discharge_Voltage = scaler.fit_transform(Discharge_Voltage).reshape(-1)

                        temporal_data.append(Discharge_Voltage)

                        break
                y_test.append(bat_dict[battery_key]["summary"]["QD"][int(cycle_num)])
                whole_test_data.append(np.array(temporal_data))

    for i in range(len(whole_test_data) - 1, 0, -1):
        if whole_test_data[i].shape != (4, inter_value):
            del whole_test_data[i]
            del y_test[i]

    whole_test_data = np.array(whole_test_data)
    y_test = np.array(y_test)

    whole_test_data = np.reshape(
        whole_test_data,
        (whole_test_data.shape[0], 1, whole_test_data.shape[1], whole_test_data.shape[2]),
    )

    whole_test_data = torch.tensor(whole_test_data, dtype=torch.float32)

    test_data = []
    y_test = []

    for battery_key in battery_list:
        first_data = []

        for cycle_num in range(1, len(bat_dict[battery_key]["summary"]["cycle"])):
            temporal_data = []
            cycle_num = str(cycle_num)

            data_length = len(bat_dict[battery_key]["cycles"][cycle_num]["Qc"])

            if cycle_num == "1":
                for i in range(11, len(bat_dict[battery_key]["cycles"][cycle_num]["Qc"])):
                    Charge_SOC = (bat_dict[battery_key]["cycles"][cycle_num]["Qc"][i] / 1.1) * 100
                    if Charge_SOC > 80.1:
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
                        Current_data = bat_dict[battery_key]["cycles"][cycle_num]["I"][
                            i : i + final_idx
                        ]
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

                        first_data.append(Current_data)
                        first_data.append(Voltage_data)
                        first_data.append(Temperature_data)
                        break

                for i in range(11, len(bat_dict[battery_key]["cycles"][cycle_num]["Qd"])):
                    Discharge_SOC = (
                        bat_dict[battery_key]["cycles"][cycle_num]["Qd"][i] / 1.1
                    ) * 100
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

                        first_data.append(Discharge_Voltage)

                        break

            else:
                for i in range(11, len(bat_dict[battery_key]["cycles"][cycle_num]["Qc"])):
                    Charge_SOC = (bat_dict[battery_key]["cycles"][cycle_num]["Qc"][i] / 1.1) * 100
                    if Charge_SOC > 80.1:
                        for p in range(data_length):
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
                        Current_data = bat_dict[battery_key]["cycles"][cycle_num]["I"][
                            i : i + final_idx
                        ]
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

                        Current_data = interpolate_timeseries(data[1], inter_value) - first_data[0]
                        Voltage_data = interpolate_timeseries(data[2], inter_value) - first_data[1]
                        Temperature_data = (
                            interpolate_timeseries(Temperature_data, inter_value) - first_data[2]
                        )

                        temporal_data.append(Current_data)
                        temporal_data.append(Voltage_data)
                        temporal_data.append(Temperature_data)
                        break

                for i in range(11, len(bat_dict[battery_key]["cycles"][cycle_num]["Qd"])):
                    Discharge_SOC = (
                        bat_dict[battery_key]["cycles"][cycle_num]["Qd"][i] / 1.1
                    ) * 100
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
                        Discharge_Voltage = (
                            interpolate_timeseries(data[1], inter_value) - first_data[3]
                        )

                        temporal_data.append(Discharge_Voltage)

                        break

                test_data.append(np.array(temporal_data))
                y_test.append(bat_dict[battery_key]["summary"]["QD"][int(cycle_num)])

    for i in range(len(test_data) - 1, 0, -1):
        if test_data[i].shape != (4, inter_value):
            del test_data[i]
            del y_test[i]

    test_data = np.array(test_data)
    y_test = np.array(y_test)

    test_data = np.reshape(
        test_data, (test_data.shape[0], 1, test_data.shape[1], test_data.shape[2])
    )

    test_data = torch.tensor(test_data, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    # FIne tuning

    for param in model_new.parameters():
        param.requires_grad = False

    for name, param in model_new.named_parameters():
        if name in ["final_reggression1.weight", "final_reggression1.bias"]:
            param.requires_grad = True

    fine_tune_rawdata = whole_test_data[0:5]
    fine_tune_deltadata = test_data[0:5]
    fine_y_data = y_test[0:5]

    dataset = MyDataset(fine_tune_rawdata, fine_tune_deltadata, fine_y_data)

    fine_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.SGD(
        [param for param in model_new.parameters() if param.requires_grad == True],
        lr=0.00001,
        momentum=0.2,
    )

    for epoch in range(0, 2):
        idx = 0
        for batch_X1, batch_X2, batch_y in fine_loader:
            batch_X1 = batch_X1.to(DEVICE)
            batch_X2 = batch_X2.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            predictions = model_new(batch_X1, batch_X2)
            loss = torch.sqrt(loss_fn(predictions, batch_y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print('epoch : ', idx , ' train_loss :', loss )

            idx += 1

    model_new.eval()

    whole_test_data = whole_test_data.to(DEVICE)
    test_data = test_data.to(DEVICE)

    predict = model_new(whole_test_data, test_data).cpu().detach().numpy()
    real_y = np.array(y_test)

    if battery_key == "b2c42":
        predict = np.delete(predict, [248])
        real_y = np.delete(real_y, [248])

    rmse = mean_squared_error(real_y, predict) ** 0.5
    mape = MAPE(np.array(real_y), np.array(predict))

    rmse_list.append(rmse)
    mape_list.append(mape)

    plot_num = 331 + k

    plt.subplot(plot_num)
    plt.plot(predict, color="r")
    plt.plot(real_y, color="b")
    plt.text(
        1, 0.95, "RMSE : " + str(round(rmse, 5)) + "\nMAPE : " + str(round(mape, 5)), fontsize=20
    )

    plt.title(battery_list[0])
    plt.ylabel("Capacity(Ah)")
    plt.xlabel("cycle")


fig.tight_layout(pad=3.0)
plt.show()

# %%
fig, ax = plt.subplots()

rmse_arr = (np.array(rmse_list) / 1.1) * 100
rmse_avg = np.mean(rmse_arr)

mape_arr = np.array(mape_list)
mape_avg = np.mean(mape_arr)

ax.boxplot([rmse_arr, mape_arr])
plt.xticks([1, 2], ["RMSE(%)", "MAPE"])
plt.text(0.6, 0.5, "[RMSE_AVG]\n   " + str(round(rmse_avg, 3)) + "%")
plt.text(1.6, 0.5, "[MAPE_AVG]\n   " + str(round(mape_avg, 3)) + "%")

plt.show
