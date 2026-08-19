# %% [markdown]
# # Consolidated research script
#
# Method group **G08**: DeepHPM SOH model. Architecture: CNN-Transformer + MLP + multihead attention. Method tags: DeepHPM|physics loss|cycle input.
#
# This copy preserves the research implementation while removing saved outputs, execution counters, pinned GPU selection, and scratch-only cells. Closely related variants are summarized in the research script index. During cleanup, 0 syntactically invalid scratch cell(s) and 2 display-only scratch cell(s) were omitted.

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

# %% [markdown]
# ## 환경세팅

# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## 데이터 불러오기

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
delete_key = [
    "b2c24",
    "b2c1",
    "b2c2",
    "b2c3",
    "b2c4",
    "b2c6",
    "b2c7",
    "b2c8",
    "b2c48",
    "b1c47",
    "b1c25",
    "b1c30",
    "b1c48",
]

for ele in delete_key:
    selected_keys.remove(ele)

# %%
trin_val_bat_key = selected_keys[:65]
full_test_bat_key = selected_keys[65:]

# %%
random.shuffle(trin_val_bat_key)

# %%
train_bat_key = trin_val_bat_key[:55]
val_bat_key = trin_val_bat_key[55:]

# %% [markdown]
# ## 데이터 전처리

# %% [markdown]
# ### Train Data 전처리

# %%
time_value = 5
inter_value = 300

# %%
whole_data = []
Cycle_true = []
Cycle_true_global = []
SOH_true = []
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
                Cycle_true.append(bat_dict[battery_key]["summary"]["cycle"][int(cycle_num)])
                Cycle_true_global.append(bat_dict[battery_key]["summary"]["cycle"][int(cycle_num)])
                SOH_true.append(bat_dict[battery_key]["summary"]["QD"][int(cycle_num)])
                whole_data.append(np.array(temporal_data))

    temporal_Cycle = np.array(Cycle_true)[Cycle_index:]
    max_Cycle = np.max(temporal_Cycle)
    temporal_Cycle = temporal_Cycle / max_Cycle
    temporal_Cycle.tolist()

    Cycle_true[Cycle_index:] = temporal_Cycle
    Cycle_index = len(Cycle_true)


whole_data = np.array(whole_data)
Cycle_true = np.array(Cycle_true)
SOH_true = np.array(SOH_true)
Cycle_true_global = np.array(Cycle_true_global) / 1050

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
Cycle_true_global = np.reshape(Cycle_true_global, (len(Cycle_true_global), 1))
print(whole_data.shape)
print(minus_data.shape)
print(Cycle_true_global.shape)


# %%
class MyDataset(Dataset):
    def __init__(self, X1, X2, X3, y1):
        self.X1 = X1
        self.X2 = X2
        self.X3 = X3
        self.y1 = y1

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.X3[idx], self.y1[idx]


dataset = MyDataset(
    torch.tensor(whole_data, dtype=torch.float32),
    torch.tensor(minus_data, dtype=torch.float32),
    torch.tensor(Cycle_true_global, dtype=torch.float32),
    torch.tensor(SOH_true, dtype=torch.float32),
)

data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

# %% [markdown]
# ### Validation Data 전처리

# %%
whole_val_data = []
Cycle_val = []
SOH_val = []
Cycle_val_global = []
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
                Cycle_val_global.append(bat_dict[battery_key]["summary"]["cycle"][int(cycle_num)])
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
Cycle_val_global = np.array(Cycle_val_global) / 1050

whole_val_data = np.reshape(
    whole_val_data, (whole_val_data.shape[0], 1, whole_val_data.shape[1], whole_val_data.shape[2])
)
whole_val_data = torch.tensor(whole_val_data, dtype=torch.float32)

Cycle_val_global = np.reshape(Cycle_val_global, (len(Cycle_val_global), 1))
Cycle_val_global = torch.tensor(Cycle_val_global, dtype=torch.float32)

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
print(Cycle_val_global.shape)
whole_val_data = whole_val_data.to(DEVICE)
minus_val_data = minus_val_data.to(DEVICE)
Cycle_val_global = Cycle_val_global.to(DEVICE)

# %% [markdown]
# ## 모델

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

        self.final_reggression1 = nn.Linear(20, 3)

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

        h = self.final_reggression1(final)

        return h


# %%
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.features = input_dim
        self.dnn = nn.Sequential(
            nn.Linear(self.features, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 5),
        )

    def forward(self, X):
        x = self.dnn(X)
        SOH, par_a, par_b, par_c, par_d = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4], x[:, 4:5]
        return SOH, par_a, par_b, par_c, par_d


# %%
class DeepHPM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(DeepHPM, self).__init__()
        self.hidden_dim = hidden_dim
        self.features = input_dim
        self.multihead_attn = nn.MultiheadAttention(self.features, 1)  # self-Attention layer
        self.Dense1 = nn.Linear(self.features, self.features)
        self.Dense2 = nn.Linear(self.features, self.hidden_dim)
        self.LN = nn.LayerNorm(self.features)
        self.activation = nn.ReLU()

    def forward(self, X):
        x, weight = self.multihead_attn(X, X, X)
        x = self.LN(x + X)
        x1 = self.Dense1(x)
        x1 = self.activation(x1 + x)
        return self.Dense2(x1)


# %%
class Final_Model(nn.Module):
    def __init__(self, module1, module2, module3, order):
        super(Final_Model, self).__init__()
        self.XNN = module1
        self.MLP = module2
        self.DeepHPM = module3
        self.order = order

    def forward(self, x, p, N):

        u, h, f1, f2 = self.net_f(x, p, N)

        return u.squeeze(1), h, f1, f2

    def net_u(self, x, p, N):
        hidden = self.XNN(x, p)
        hidden.requires_grad_(True)
        SOH, par_a, par_b, par_c, par_d = self.MLP(torch.concat([hidden, N], dim=1))

        return SOH, par_a, par_b, par_c, par_d, hidden

    def net_f(self, x, p, N):
        N.requires_grad_(True)
        u, par_a, par_b, par_c, par_d, h = self.net_u(x, p, N)
        SOH = u
        u_t = torch.autograd.grad(
            u, N, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
        )[0]
        u_h = [u]
        for i in range(self.order):
            u_ = torch.autograd.grad(
                u_h[-1],
                h,
                grad_outputs=torch.ones_like(u_h[-1]),
                retain_graph=True,
                create_graph=True,
            )[0]
            u_h.append(u_)
        deri = h
        for data in u_h:
            deri = torch.concat([deri, data], dim=1)

        f1 = u_t - self.DeepHPM(deri)
        f2 = SOH - par_a * torch.exp(N * par_b) - par_c * torch.exp(N * par_d)
        return SOH, h, f1, f2


# %%
model = Final_Model(SOH_model(4, inter_value, 5, 2, 5, 2), MLP(4), DeepHPM(13, 1), 3).to(DEVICE)
# model = Final_Model(SOH_model(4, inter_value, 5,2,5,2), MLP(3)).to(DEVICE)
model = model.train()

lam_1 = torch.tensor([1.0]).to(DEVICE)
lam_2 = torch.tensor([1.0]).to(DEVICE)
lam_3 = torch.tensor([1.0]).to(DEVICE)

lam_1 = torch.nn.Parameter(lam_1)
lam_2 = torch.nn.Parameter(lam_2)
lam_3 = torch.nn.Parameter(lam_2)

model.register_parameter("lam_1", lam_1)
model.register_parameter("lam_2", lam_2)
model.register_parameter("lam_3", lam_3)

optimizer = torch.optim.Adam(model.parameters())
loss_fn = torch.nn.MSELoss()

Total_loss_best = 1000
SOH_loss_best = 1000


for epoch in range(0, 300):
    idx = 0
    for batch_X1, batch_X2, batch_X3, batch_y1 in data_loader:
        model.train()
        batch_X1 = batch_X1.to(DEVICE)
        batch_X2 = batch_X2.to(DEVICE)
        batch_X3 = batch_X3.to(DEVICE)
        batch_y1 = batch_y1.to(DEVICE)
        SOH_pred, hidden_value, f1, f2 = model(batch_X1, batch_X2, batch_X3)
        l1 = torch.sqrt(loss_fn(SOH_pred, batch_y1))
        l2 = torch.sqrt(loss_fn(f1, torch.zeros(f1.shape).to(DEVICE)))
        l3 = torch.sqrt(loss_fn(f2, torch.zeros(f2.shape).to(DEVICE)))

        # loss = l1 + l2
        loss = lam_1 * l1 + lam_2 * l2 + lam_3 * l3 - torch.log(lam_1 * lam_2 * lam_3)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()

        SOH_pred_val, _, f1_val, f2_val = model(whole_val_data, minus_val_data, Cycle_val_global)

        SOH_pred_val = SOH_pred_val.cpu().detach().numpy()
        f2_val = f2_val.cpu().detach().numpy()

        SOH_val_loss = mean_squared_error(SOH_val, SOH_pred_val) ** 0.5
        Physics_val_loss = mean_squared_error(f2_val, np.zeros(len(f2_val))) ** 0.5

        print(
            "epoch : ",
            epoch,
            "/idx : ",
            idx,
            "/l1_loss :",
            round(float(l1), 4),
            "/l2_loss :",
            round(float(l3), 4),
            "/SOH_val_loss : ",
            round(SOH_val_loss, 4),
            "/Physics_val_loss: ",
            round(Physics_val_loss, 6),
        )

        Total_loss = SOH_val_loss

        if Total_loss < SOH_loss_best:
            SOH_loss_best = Total_loss
            torch.save(
                model.state_dict(), artifact_path("New_normalization_300_SOH_HPM_input_cylce_all")
            )

        idx += 1

# %% [markdown]
# ## 모델평가

# %%
# par_a = torch.tensor([-0.5]).to(DEVICE)
# par_b = torch.tensor([-0.5]).to(DEVICE)
# par_c = torch.tensor([-0.5]).to(DEVICE)
# par_d = torch.tensor([1.0]).to(DEVICE)

lam_1 = torch.tensor([1.0]).to(DEVICE)
lam_2 = torch.tensor([1.0]).to(DEVICE)
lam_3 = torch.tensor([1.0]).to(DEVICE)
# lam_4 = torch.tensor([1.0]).to(DEVICE)


# par_a = torch.nn.Parameter(par_a)
# par_b = torch.nn.Parameter(par_b)
# par_c = torch.nn.Parameter(par_c)
# par_d = torch.nn.Parameter(par_d)

lam_1 = torch.nn.Parameter(lam_1)
lam_2 = torch.nn.Parameter(lam_2)
lam_3 = torch.nn.Parameter(lam_3)

# %%
rmse_list = []
mape_list = []

mode = "SOH"

fig = plt.figure(figsize=(20, 12))

for k in range(len(full_test_bat_key) - 1):
    battery_list = [full_test_bat_key[k]]

    model_new = Final_Model(SOH_model(4, inter_value, 5, 2, 5, 2), MLP(4), DeepHPM(13, 1), 3).to(
        DEVICE
    )

    model_new.register_parameter("lam_1", lam_1)
    model_new.register_parameter("lam_2", lam_2)
    model_new.register_parameter("lam_3", lam_3)

    # model_new.register_parameter('lam_4', lam_4)

    model_new.load_state_dict(
        torch.load(artifact_path("New_normalization_300_SOH_HPM_input_cylce_all"))
    )
    model_new.eval()

    whole_test_data = []
    Cycle_test = []
    SOH_test = []
    Cycle_index = 0
    Cycle_test_global = []

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
                    Cycle_test_global.append(
                        bat_dict[battery_key]["summary"]["cycle"][int(cycle_num)]
                    )
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
    Cycle_test_global = np.array(Cycle_test_global) / 1050

    whole_test_data = np.reshape(
        whole_test_data,
        (whole_test_data.shape[0], 1, whole_test_data.shape[1], whole_test_data.shape[2]),
    )
    whole_test_data = torch.tensor(whole_test_data, dtype=torch.float32)

    Cycle_test_global = np.reshape(Cycle_test_global, (len(Cycle_test_global), 1))
    Cycle_test_global = torch.tensor(Cycle_test_global, dtype=torch.float32)

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
    Cycle_test_global = Cycle_test_global.to(DEVICE)

    SOH_test_pred, hidden_test, _, _ = model_new(whole_test_data, test_data, Cycle_test_global)

    SOH_test_pred = SOH_test_pred.cpu().detach().numpy()

    SOH_test = np.array(SOH_test)
    Cycle_test = np.array(Cycle_test)

    if battery_key == "b2c42":
        SOH_test_pred = np.delete(SOH_test_pred, [248])
        Cycle_test = np.delete(Cycle_test, [248])
        SOH_test = np.delete(SOH_test, [248])

    # plot_num = 331 + k

    if mode == "SOH":
        soh_rmse = mean_squared_error(SOH_test, SOH_test_pred) ** 0.5
        soh_mape = MAPE(np.array(SOH_test), np.array(SOH_test_pred))

        rmse_list.append(soh_rmse)
        mape_list.append(soh_mape)

        # plt.subplot(plot_num)
        # plt.plot(SOH_test_pred , color = 'r', label = 'Estimated SOH')
        # plt.plot(SOH_test, color = 'b', label = 'real SOH')
        # plt.text(1, 0.95, 'RMSE : ' + str(round(soh_rmse, 5)), fontsize = 20)

        # plt.title(battery_list[0])
        # plt.ylabel('Capacity (AH)' , fontsize=15)
        # plt.xlabel('cycle', fontsize=15)
        # plt.legend()

    if mode == "physics":
        physics_pred = (
            float(model_new.par_a) * (Cycle_test_pred**3)
            + float(model_new.par_b) * (Cycle_test_pred**2)
            + float(model_new.par_c) * (Cycle_test_pred)
            + float(model_new.par_d)
        )

        soh_rmse = mean_squared_error(SOH_test, physics_pred) ** 0.5
        soh_mape = MAPE(np.array(SOH_test), np.array(physics_pred))

        rmse_list.append(soh_rmse)
        mape_list.append(soh_mape)

        plt.subplot(plot_num)
        plt.plot(physics_pred, color="r", label="Physics SOH")
        plt.plot(SOH_test_pred, color="y", label="Data driven SOH")
        plt.plot(SOH_test, color="b", label="real SOH")

        # plt.text(1, 0.95, 'RMSE : ' + str(round(soh_rmse, 5)), fontsize = 20)

        plt.title(battery_list[0])
        plt.ylabel("Capacity (AH)", fontsize=15)
        plt.xlabel("cycle", fontsize=15)
        plt.legend(fontsize=10)

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


# fig.tight_layout(pad=3.0)
# plt.show()

# %%
fig, ax = plt.subplots()

rmse_arr = np.array(rmse_list) * 100 / 1.1
rmse_avg = np.mean(rmse_arr)

mape_arr = np.array(mape_list)
mape_avg = np.mean(mape_arr)

ax.boxplot([rmse_arr, mape_arr])
plt.xticks([1, 2], ["RMSE(%)", "MAPE"])
plt.text(0.6, 0.5, "[RMSE_AVG]\n   " + str(round(rmse_avg, 3)) + "%")
plt.text(1.6, 0.5, "[MAPE_AVG]\n   " + str(round(mape_avg, 3)) + "%")

plt.show

# %%
# par_a = torch.tensor([-0.5]).to(DEVICE)
# par_b = torch.tensor([-0.5]).to(DEVICE)
# par_c = torch.tensor([-0.5]).to(DEVICE)
# par_d = torch.tensor([1.0]).to(DEVICE)

lam_1 = torch.tensor([1.0]).to(DEVICE)
lam_2 = torch.tensor([1.0]).to(DEVICE)
lam_3 = torch.tensor([1.0]).to(DEVICE)
# lam_4 = torch.tensor([1.0]).to(DEVICE)


# par_a = torch.nn.Parameter(par_a)
# par_b = torch.nn.Parameter(par_b)
# par_c = torch.nn.Parameter(par_c)
# par_d = torch.nn.Parameter(par_d)

lam_1 = torch.nn.Parameter(lam_1)
lam_2 = torch.nn.Parameter(lam_2)
lam_3 = torch.nn.Parameter(lam_3)

# %%
model_new = Final_Model(SOH_model(4, inter_value, 5, 2, 5, 2), MLP(4), DeepHPM(13, 1), 3).to(DEVICE)

model_new.register_parameter("lam_1", lam_1)
model_new.register_parameter("lam_2", lam_2)
model_new.register_parameter("lam_3", lam_3)

# model_new.register_parameter('lam_4', lam_4)

model_new.load_state_dict(
    torch.load(artifact_path("New_normalization_300_SOH_HPM_input_cylce_all"))
)
model_new.eval()

# %%
SOH_pred_val, _, _, _ = model_new(whole_val_data, minus_val_data, Cycle_val_global)

SOH_pred_val = SOH_pred_val.cpu().detach().numpy()

SOH_val_loss = mean_squared_error(SOH_val, SOH_pred_val) ** 0.5

SOH_mape = MAPE(np.array(SOH_val), np.array(SOH_pred_val))

SOH_rmspe = RMSPE(np.array(SOH_val), np.array(SOH_pred_val)) * 100

print("SOH_RMSPE : ", str(np.round(SOH_rmspe, 4)), " %")
# print("Cycle_RMSPE : ", str(np.round(Cycle_rmspe, 4)), ' %')
print("SOH_MAPE : ", str(np.round(SOH_mape, 4)), " %")
