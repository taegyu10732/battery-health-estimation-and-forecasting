# %% [markdown]
# # Consolidated research script
#
# Method group **G06**: Variational SOH model. Architecture: VAE CNN-Transformer. Method tags: VAE|reconstruction.
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
import matplotlib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
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
val_bat_key = ["b1c28", "b2c34", "b1c15", "b2c19", "b2c21"]
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
time_value = 5
inter_value = 300

# %%
whole_data = []
SOH_true = []
HI_true = []
Cycle_index = 0

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

                    Discharge_Voltage = interpolate_timeseries(data[1], inter_value)
                    Discharge_Voltage = (Discharge_Voltage - 3) / (3.6 - 3)

                    temporal_data.append(Discharge_Voltage)

                    break

            if np.array(temporal_data).shape == (4, inter_value):
                SOH_true.append(bat_dict[battery_key]["summary"]["QD"][int(cycle_num)])
                HI_true.append(
                    (bat_dict[battery_key]["summary"]["QD"][int(cycle_num)] - 0.88) / 0.22
                )
                whole_data.append(np.array(temporal_data))


whole_data = np.array(whole_data)
SOH_true = np.array(SOH_true)
HI_true = np.array(HI_true)

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
    torch.tensor(HI_true, dtype=torch.float32),
)

data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

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

        self.linear1 = nn.Linear(100 * self.in_features, 20)

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

        self.linear1_prime = nn.Linear(100 * self.in_features, 20)

        self.final_reggression_1 = nn.Linear(40, 20)

        self.linear_mu = nn.Linear(20, 2)
        self.linear_sigma = nn.Linear(20, 2)

        # self.z1_reg_1 = nn.Linear(1, 20)
        # self.z1_reg_2 = nn.Linear(20, 20)
        # self.z1_reg_3 = nn.Linear(20, 20)
        # self.z2_reg_1 = nn.Linear(1, 20)
        # self.z2_reg_2 = nn.Linear(20, 20)
        # self.z2_reg_3 = nn.Linear(20, 20)

        self.regressor_1 = nn.Linear(2, 20)
        self.regressor_f = nn.Linear(20, 1)

        self.sigmoid = nn.Sigmoid()

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

        total = torch.cat((x, p), 1)

        total = self.final_reggression_1(total)

        mu = self.linear_mu(total)
        sigma = self.linear_sigma(total)
        z = self.z_calculator(mu, sigma)

        # print(z.shape , z[:, 0:1].shape, z[:, 1:2].shape)
        reg = self.regressor_1(z)
        reg = F.leaky_relu(reg)
        reg = self.regressor_f(reg)

        return reg.squeeze(1), z, mu, sigma

    def z_calculator(self, mu, sigma):
        batch = mu.shape[0]
        dim = mu.shape[1]
        epsilon = torch.rand(batch, dim).to(DEVICE)
        return mu + torch.exp(0.5 * sigma) * epsilon


# %%
whole_val_data = []
Cycle_val = []
SOH_val = []
Cycle_index = 0

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
                    Discharge_Voltage = interpolate_timeseries(data[1], inter_value)
                    Discharge_Voltage = (Discharge_Voltage - 3) / (3.6 - 3)

                    temporal_data.append(Discharge_Voltage)

                    break
            if np.array(temporal_data).shape == (4, inter_value):
                Cycle_val.append(bat_dict[battery_key]["summary"]["cycle"][int(cycle_num)])
                SOH_val.append(bat_dict[battery_key]["summary"]["QD"][int(cycle_num)])
                whole_val_data.append(np.array(temporal_data))

    temporal_Cycle = np.array(Cycle_val)[Cycle_index:]
    max_Cycle = np.max(temporal_Cycle)
    temporal_Cycle = temporal_Cycle / max_Cycle
    temporal_Cycle.tolist()

    Cycle_val[Cycle_index:] = temporal_Cycle
    Cycle_index = len(Cycle_val)

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
model = SOH_model(4, inter_value, 5, 2, 5, 2).to(DEVICE)
model = model.train()

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss()

Total_loss_best = 1000
SOH_loss_best = 1000


for epoch in range(0, 300):
    idx = 0
    for batch_X1, batch_X2, batch_y1, batch_y2 in data_loader:
        model.train()
        batch_X1 = batch_X1.to(DEVICE)
        batch_X2 = batch_X2.to(DEVICE)
        batch_y1 = batch_y1.to(DEVICE)
        batch_y2 = batch_y2.to(DEVICE)
        # print(batch_X1.shape)
        SOH_pred, z, mu, sigma = model(batch_X1, batch_X2)
        l1 = torch.sqrt(loss_fn(SOH_pred, batch_y1))
        l2 = torch.sqrt(loss_fn(z[:, 0:1], z[:, 1:2]))
        l3 = torch.sqrt(loss_fn(z[:, 0:1].squeeze(1), batch_y2))

        kl_loss = torch.mean(
            torch.sum(-0.5 * (1 + sigma - torch.square(mu) - torch.exp(sigma)), dim=1)
        )

        loss = l1 + 0.00001 * kl_loss + l2 + l3
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()

        SOH_pred_val, _, _, _ = model(whole_val_data, minus_val_data)

        SOH_pred_val = SOH_pred_val.cpu().detach().numpy()

        SOH_val_loss = mean_squared_error(SOH_val, SOH_pred_val) ** 0.5

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
            "/kl_loss :",
            round(float(kl_loss), 4),
            "/val_loss :",
            round(float(SOH_val_loss), 4),
        )

        if SOH_val_loss < SOH_loss_best:
            SOH_loss_best = SOH_val_loss
            torch.save(model.state_dict(), artifact_path("New_normalization_300_SOH_VAE_200"))

        idx += 1

# %%
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
# test_bat_key = ['b1c9','b1c12','b1c21','b1c29']

# %%
rmse_list = []
mape_list = []

mode = "latent"

fig = plt.figure(figsize=(15, 12))

for k in range(len(test_bat_key) - 1):
    battery_list = [test_bat_key[k]]

    model_new = SOH_model(4, inter_value, 5, 2, 5, 2).to(DEVICE)

    model_new.load_state_dict(torch.load(artifact_path("New_normalization_300_SOH_VAE_200")))
    model_new.eval()

    whole_test_data = []
    Cycle_test = []
    SOH_test = []
    Cycle_index = 0

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
                        Discharge_Voltage = interpolate_timeseries(data[1], inter_value)
                        Discharge_Voltage = (Discharge_Voltage - 3) / (3.6 - 3)

                        temporal_data.append(Discharge_Voltage)

                        break
                if np.array(temporal_data).shape == (4, inter_value):
                    Cycle_test.append(bat_dict[battery_key]["summary"]["cycle"][int(cycle_num)])
                    SOH_test.append(bat_dict[battery_key]["summary"]["QD"][int(cycle_num)])
                    whole_test_data.append(np.array(temporal_data))

        temporal_Cycle = np.array(Cycle_test)[Cycle_index:]
        max_Cycle = np.max(temporal_Cycle)
        temporal_Cycle = temporal_Cycle / max_Cycle
        temporal_Cycle.tolist()

        Cycle_test[Cycle_index:] = temporal_Cycle
        Cycle_index = len(Cycle_test)

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

    SOH_test_pred, pred_z, pred_mu, pred_sigma = model_new(whole_test_data, test_data)

    SOH_test_pred = SOH_test_pred.cpu().detach().numpy()
    pred_z = pred_z.cpu().detach().numpy()

    SOH_test = np.array(SOH_test)

    """
    if battery_key == 'b2c42':
        SOH_test_pred = np.delete(SOH_test_pred, [248])
        pred_z = np.delete(pred_z, [248, 1])
        pred_z = np.delete(pred_z, [248, 0])
        SOH_test = np.delete(SOH_test, [248])"""

    plot_num = 331 + k

    if mode == "SOH":
        if battery_key == "b2c42":
            SOH_test_pred = np.delete(SOH_test_pred, [248])
            SOH_test = np.delete(SOH_test, [248])

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

    if mode == "latent":
        cmap = matplotlib.colormaps["viridis"]
        # cmap = plt.cm.get_cmap('viridis')
        normalize = mcolors.Normalize(vmin=0.88, vmax=1.1)

        plt.subplot(plot_num)
        plt.title(battery_list[0])

        plt.ylabel("Z_1", fontsize=15)
        plt.xlabel("Z_2", fontsize=15)

        plot_list = np.linspace(97.5, 80, 8)

        for refer in plot_list:
            for k in range(0, len(pred_z)):
                if ((SOH_test[k] / 1.1) * 100) <= refer:
                    plt.scatter(pred_z[k, 1], pred_z[k, 0], color=cmap(normalize(SOH_test[k])))
                    print((SOH_test[k] / 1.1) * 100)
                    break
        #     plt.scatter(k, pred_z[k, 0], color = cmap(normalize(SOH_test[k])))
        # , pred_z[k, 0]

        plt.xticks(np.linspace(0, 1, 11))
        plt.yticks(np.linspace(0, 1, 11))

        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
        scalarmappaple.set_array(1.1)
        plt.colorbar(scalarmappaple)


fig.tight_layout(pad=3.0)
plt.show()

# %%
whole_val_data = whole_val_data.to(DEVICE)
minus_val_data = minus_val_data.to(DEVICE)

SOH_pred_val, Cycle_pred_val = model_new(whole_val_data, minus_val_data)

SOH_pred_val = SOH_pred_val.cpu().detach().numpy()
Cycle_pred_val = Cycle_pred_val.cpu().detach().numpy()

SOH_val_loss = mean_squared_error(SOH_val, SOH_pred_val) ** 0.5
Cycle_val_loss = mean_squared_error(Cycle_val, Cycle_pred_val) ** 0.5

SOH_mape = MAPE(np.array(SOH_val), np.array(SOH_pred_val))
Cycle_mape = MAPE(np.array(Cycle_val), np.array(Cycle_pred_val))

SOH_rmspe = RMSPE(np.array(SOH_val), np.array(SOH_pred_val)) * 100
Cycle_rmspe = RMSPE(np.array(Cycle_val), np.array(Cycle_pred_val)) * 100

print("SOH_RMSPE : ", str(np.round(SOH_rmspe, 4)), " %")
# print("Cycle_RMSPE : ", str(np.round(Cycle_rmspe, 4)), ' %')
print("SOH_MAPE : ", str(np.round(SOH_mape, 4)), " %")
print("Cycle_MAPE : ", str(np.round(Cycle_val_loss, 4)))
