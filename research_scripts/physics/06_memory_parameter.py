# %% [markdown]
# # Consolidated research script
#
# Method group **G31**: Memory-parameter physics model. Architecture: CNN-LSTM + learnable memory. Method tags: parameter memory|learnable parameters.
#
# This copy preserves the research implementation while removing saved outputs, execution counters, pinned GPU selection, and scratch-only cells. Closely related variants are summarized in the research script index. During cleanup, 1 syntactically invalid scratch cell(s) and 2 display-only scratch cell(s) were omitted.

# %%
from battery_soh.data import resolve_data_dir
from battery_soh.evaluation import MAPE, RMSPE
from battery_soh.paths import artifact_path
from battery_soh.preprocessing import interpolate_timeseries
from battery_soh.raw_data import load_battery_dictionary
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import optuna
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

# %% [markdown]
# # Data Loading

# %%
batch_keys = [*bat_dict.keys()]
len(batch_keys)

# %%
random.seed(12)

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
selected_keys.remove("b2c9")
selected_keys.remove("b2c1")
selected_keys.remove("b2c17")
selected_keys.remove("b2c25")
selected_keys.remove("b2c33")
selected_keys.remove("b2c41")
selected_keys.remove("b2c6")
selected_keys.remove("b2c3")
selected_keys.remove("b2c2")
selected_keys.remove("b3c46")
selected_keys.remove("b3c18")

# %%
len(selected_keys)

# %%
random.shuffle(selected_keys)

# %%
train_bat_key = selected_keys[:75]
val_bat_key = selected_keys[75:90]
test_bat_key = selected_keys[90:]

# %% [markdown]
# # Preprocessing

# %%
time_value = 5
inter_value = 100

# %% [markdown]
# ## Train data

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
            cap_list.append(1.034)
            indi_94 = False
        if (temp_SOH < 92.1) & indi_92:
            cycle_list.append(int(cycle_str))
            cap_list.append(1.012)
            indi_92 = False
        if (temp_SOH < 90.1) & indi_90:
            cycle_list.append(int(cycle_str))
            cap_list.append(0.99)
            indi_90 = False
        if (temp_SOH < 88.1) & indi_88:
            cycle_list.append(int(cycle_str))
            cap_list.append(0.968)
            indi_88 = False
        if (temp_SOH < 86.2) & indi_86:
            cycle_list.append(int(cycle_str))
            cap_list.append(0.946)
            indi_86 = False
        if (temp_SOH < 84.2) & indi_84:
            cycle_list.append(int(cycle_str))
            cap_list.append(0.924)
            indi_84 = False
        if (temp_SOH < 82.2) & indi_82:
            cycle_list.append(int(cycle_str))
            cap_list.append(0.902)
            indi_82 = False
        if (temp_SOH < 80.2) & indi_80:
            cycle_list.append(int(cycle_str))
            cap_list.append(0.88)
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
train_parameter = []


def objective_function(params):
    y_predicted = model_function(params, RUL_true_temp)
    difference = Cap_temp - y_predicted
    error = np.abs(difference)
    return np.mean(error)


for index in range(0, len(train_bat_key)):
    RUL_true_temp = np.append((1 / 1400), RUL_true[index])
    Cap_temp = Cap_true[index]

    best_error = 10

    for par in [9, 10, 11, 12]:

        def model_function(params, RUL_true_temp):
            return params[0] * np.exp(RUL_true_temp * params[1]) + params[2] * np.exp(
                RUL_true_temp * params[3]
            )

        initial_params = [-3, par, 0.1, 0.1]

        result = minimize(objective_function, initial_params, method="BFGS")
        temp_error = result.fun

        if temp_error < best_error:
            best_error = temp_error
            best_param = [result.x[0], result.x[1], result.x[2], result.x[3]]

    train_parameter.append(best_param)

train_parameter = np.array(train_parameter)

# %%
X = np.linspace(1, len(train_parameter), len(train_parameter))

plt.scatter(X, np.array(train_parameter)[:, 1])

# %%
whole_data_tensor = torch.tensor(whole_data, dtype=torch.float32).to(DEVICE)
RUL_true_tensor = torch.tensor(RUL_true, dtype=torch.float32).to(DEVICE)
Cap_true_tensor = torch.tensor(Cap_true, dtype=torch.float32).to(DEVICE)


# %%
class MyDataset(Dataset):
    def __init__(self, X1, y1, y2, y3):
        self.X1 = X1
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        return self.X1[idx], self.y1[idx], self.y2[idx], self.y3[idx]


dataset = MyDataset(
    torch.tensor(whole_data, dtype=torch.float32),
    torch.tensor(RUL_true, dtype=torch.float32),
    torch.tensor(Cap_true, dtype=torch.float32),
    torch.tensor(train_parameter, dtype=torch.float32),
)

data_loader = DataLoader(dataset, batch_size=15, shuffle=True)

# %% [markdown]
# ## Val data

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
            cap_list.append(1.034)
            indi_94 = False
        if (temp_SOH < 92.1) & indi_92:
            cycle_list.append(int(cycle_str))
            cap_list.append(1.012)
            indi_92 = False
        if (temp_SOH < 90.1) & indi_90:
            cycle_list.append(int(cycle_str))
            cap_list.append(0.99)
            indi_90 = False
        if (temp_SOH < 88.1) & indi_88:
            cycle_list.append(int(cycle_str))
            cap_list.append(0.968)
            indi_88 = False
        if (temp_SOH < 86.2) & indi_86:
            cycle_list.append(int(cycle_str))
            cap_list.append(0.946)
            indi_86 = False
        if (temp_SOH < 84.2) & indi_84:
            cycle_list.append(int(cycle_str))
            cap_list.append(0.924)
            indi_84 = False
        if (temp_SOH < 82.2) & indi_82:
            cycle_list.append(int(cycle_str))
            cap_list.append(0.902)
            indi_82 = False
        if (temp_SOH < 80.2) & indi_80:
            cycle_list.append(int(cycle_str))
            cap_list.append(0.88)
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

# %%
RUL_val = RUL_val / 1400

# %%
val_parameter = []


def objective_function(params):
    y_predicted = model_function(params, RUL_true_temp)
    difference = Cap_temp - y_predicted
    error = np.abs(difference)
    return np.mean(error)


for index in range(0, len(val_bat_key)):
    RUL_true_temp = np.append((1 / 1400), RUL_val[index])
    Cap_temp = Cap_val[index]

    best_error = 10

    for par in [9, 10, 11, 12]:

        def model_function(params, RUL_true_temp):
            return params[0] * np.exp(RUL_true_temp * params[1]) + params[2] * np.exp(
                RUL_true_temp * params[3]
            )

        initial_params = [-3, par, 0.1, 0.1]

        result = minimize(objective_function, initial_params, method="BFGS")
        temp_error = result.fun

        if temp_error < best_error:
            best_error = temp_error
            best_param = [result.x[0], result.x[1], result.x[2], result.x[3]]

    val_parameter.append(best_param)

val_parameter = np.array(val_parameter)

# %% [markdown]
# ## Test data

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
            cap_list.append(1.034)
            indi_94 = False
        if (temp_SOH < 92.1) & indi_92:
            cycle_list.append(int(cycle_str))
            cap_list.append(1.012)
            indi_92 = False
        if (temp_SOH < 90.1) & indi_90:
            cycle_list.append(int(cycle_str))
            cap_list.append(0.99)
            indi_90 = False
        if (temp_SOH < 88.1) & indi_88:
            cycle_list.append(int(cycle_str))
            cap_list.append(0.968)
            indi_88 = False
        if (temp_SOH < 86.2) & indi_86:
            cycle_list.append(int(cycle_str))
            cap_list.append(0.946)
            indi_86 = False
        if (temp_SOH < 84.2) & indi_84:
            cycle_list.append(int(cycle_str))
            cap_list.append(0.924)
            indi_84 = False
        if (temp_SOH < 82.2) & indi_82:
            cycle_list.append(int(cycle_str))
            cap_list.append(0.902)
            indi_82 = False
        if (temp_SOH < 80.2) & indi_80:
            cycle_list.append(int(cycle_str))
            cap_list.append(0.88)
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

# %%
RUL_test = RUL_test / 1400

# %%
test_parameter = []


def objective_function(params):
    y_predicted = model_function(params, RUL_true_temp)
    difference = Cap_temp - y_predicted
    error = np.abs(difference)
    return np.mean(error)


for index in range(0, len(test_bat_key)):
    RUL_true_temp = np.append((1 / 1400), RUL_test[index])
    Cap_temp = Cap_test[index]

    best_error = 10

    for par in [9, 10, 11, 12]:

        def model_function(params, RUL_true_temp):
            return params[0] * np.exp(RUL_true_temp * params[1]) + params[2] * np.exp(
                RUL_true_temp * params[3]
            )

        initial_params = [-3, par, 0.1, 0.1]

        result = minimize(objective_function, initial_params, method="BFGS")
        temp_error = result.fun

        if temp_error < best_error:
            best_error = temp_error
            best_param = [result.x[0], result.x[1], result.x[2], result.x[3]]

    test_parameter.append(best_param)

test_parameter = np.array(test_parameter)

# %%
len(bat_dict[train_bat_key[0]]["summary"]["QD"])

# %%
X1 = np.linspace(1, len(train_parameter), len(train_parameter))
X2 = np.linspace(
    len(train_parameter), len(train_parameter) + len(val_parameter), len(val_parameter)
)
X3 = np.linspace(
    len(train_parameter) + len(val_parameter),
    len(train_parameter) + len(val_parameter) + len(test_parameter),
    len(test_parameter),
)

id_ = 1

plt.scatter(X1, np.array(train_parameter)[:, id_], label="train")
plt.scatter(X2, np.array(val_parameter)[:, id_], label="val")
plt.scatter(X3, np.array(test_parameter)[:, id_], label="test")

plt.xlabel("Cell number", fontsize=20)
plt.legend()

# %%
X1 = np.linspace(1, len(train_parameter), len(train_parameter))
X2 = np.linspace(
    len(train_parameter), len(train_parameter) + len(val_parameter), len(val_parameter)
)
X3 = np.linspace(
    len(train_parameter) + len(val_parameter),
    len(train_parameter) + len(val_parameter) + len(test_parameter),
    len(test_parameter),
)

id_ = 1

for i in range(0, len(train_bat_key)):
    plt.scatter(
        np.array(train_parameter)[i, id_],
        len(bat_dict[train_bat_key[i]]["summary"]["QD"]),
        label="train",
        color="black",
    )

for i in range(0, len(val_bat_key)):
    plt.scatter(
        np.array(val_parameter)[i, id_],
        len(bat_dict[val_bat_key[i]]["summary"]["QD"]),
        label="train",
        color="blue",
    )

for i in range(0, len(test_bat_key)):
    plt.scatter(
        np.array(test_parameter)[i, id_],
        len(bat_dict[test_bat_key[i]]["summary"]["QD"]),
        label="train",
        color="red",
    )

plt.xlabel("Parameter b", fontsize=20)
plt.ylabel("End of Life", fontsize=20)

# plt.legend()

# %% [markdown]
# ## Data tensor

# %%
whole_val_data_tensor = torch.tensor(whole_val_data, dtype=torch.float32).to(DEVICE)
RUL_val_tensor = torch.tensor(RUL_val, dtype=torch.float32).to(DEVICE)
Cap_val_tensor = torch.tensor(Cap_val, dtype=torch.float32).to(DEVICE)
val_parameter_tensor = torch.tensor(val_parameter, dtype=torch.float32).to(DEVICE)

# %%
whole_test_data_tensor = torch.tensor(whole_test_data, dtype=torch.float32).to(DEVICE)

# %% [markdown]
# # Network

# %%
## Memoery module infused


def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(100, 4608))
        self.shrink_thres = 0.01
        self.network = nn.Sequential(
            nn.Conv2d(4, 8, 2, padding="same"),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(8, 16, 2, padding="same"),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 32, 2, padding="same"),
            nn.AvgPool2d(2, 2),
        )

        self.fc1 = nn.Linear(4608, 8)
        # self.lstm = nn.LSTM(1, 32, 3,batch_first=True)
        # self.fc2 = nn.Linear(100, 8)

    def forward(self, x):
        x = self.network(x)
        x = x.view(-1, 4608)
        att_weight = F.linear(x, self.weight)
        att_weight = F.softmax(att_weight, dim=1)

        if self.shrink_thres > 0:
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        mem_trans = self.weight.permute(1, 0)

        x = F.linear(att_weight, mem_trans)
        x = self.fc1(x)

        return x.squeeze(1), att_weight.squeeze(1)


# %%
## VAE


def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(100, 4608))
        self.shrink_thres = 0.01
        self.network = nn.Sequential(
            nn.Conv2d(4, 8, 2, padding="same"),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(8, 16, 2, padding="same"),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 32, 2, padding="same"),
            nn.AvgPool2d(2, 2),
        )

        self.fc1 = nn.Linear(4608, 50)
        self.linear_mu = nn.Linear(50, 2)
        self.linear_sigma = nn.Linear(50, 2)
        self.regressor_1 = nn.Linear(2, 20)
        self.regressor_f = nn.Linear(20, 8)
        self.regressor_2 = nn.Linear(2, 20)
        self.regressor_par = nn.Linear(20, 3)
        # self.lstm = nn.LSTM(1, 32, 3,batch_first=True)
        # self.fc2 = nn.Linear(100, 8)

    def forward(self, x):
        x = self.network(x)
        x = x.view(-1, 4608)
        x = F.leaky_relu(x)
        x = self.fc1(x)

        mu = self.linear_mu(x)
        sigma = self.linear_sigma(x)
        z = self.z_calculator(mu, sigma)

        reg = self.regressor_1(z)
        reg = F.leaky_relu(reg)
        reg = self.regressor_f(reg)

        par = self.regressor_2(z)
        par = F.leaky_relu(par)
        par = self.regressor_par(par)

        return reg.squeeze(1), par.squeeze(1), z, mu, sigma

    def z_calculator(self, mu, sigma):
        batch = mu.shape[0]
        dim = mu.shape[1]
        epsilon = torch.rand(batch, dim).to(DEVICE)
        return mu + torch.exp(0.5 * sigma) * epsilon


# %%
## VAE


def hard_shrink_relu(input, lambd=0, epsilon=1e-12):
    output = (F.relu(input - lambd) * input) / (torch.abs(input - lambd) + epsilon)
    return output


class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(100, 4608))
        self.shrink_thres = 0.01
        self.network = nn.Sequential(
            nn.Conv2d(4, 8, 2, padding="same"),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(8, 16, 2, padding="same"),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 32, 2, padding="same"),
            nn.AvgPool2d(2, 2),
        )

        self.fc1 = nn.Linear(4608, 50)
        self.regressor_1 = nn.Linear(50, 20)
        self.regressor_f = nn.Linear(20, 8)

        self.regressor_par_1 = nn.Linear(50, 30)
        self.regressor_par_11 = nn.Linear(30, 20)
        self.regressor_par_111 = nn.Linear(20, 1)

        self.regressor_par_2 = nn.Linear(50, 70)
        self.regressor_par_22 = nn.Linear(70, 1)

        self.regressor_par_3 = nn.Linear(50, 1)
        # self.lstm = nn.LSTM(1, 32, 3,batch_first=True)
        # self.fc2 = nn.Linear(100, 8)

    def forward(self, x):
        x = self.network(x)
        x = x.view(-1, 4608)
        x = F.leaky_relu(x)
        x = self.fc1(x)

        reg = self.regressor_1(x)
        reg = F.leaky_relu(reg)
        reg = self.regressor_f(reg)

        par_a = self.regressor_par_1(x)
        par_a = F.leaky_relu(par_a)
        par_a = self.regressor_par_11(par_a)
        par_a = F.leaky_relu(par_a)
        par_a = self.regressor_par_111(par_a)

        par_b = self.regressor_par_2(x)
        par_b = F.leaky_relu(par_b)
        par_b = self.regressor_par_22(par_b)

        par_c = self.regressor_par_3(x)

        par = torch.cat([par_a, par_b, par_c], dim=1)

        return reg.squeeze(1), par.squeeze(1)

    def z_calculator(self, mu, sigma):
        batch = mu.shape[0]
        dim = mu.shape[1]
        epsilon = torch.rand(batch, dim).to(DEVICE)
        return mu + torch.exp(0.5 * sigma) * epsilon


# %%
class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding="same"),
            nn.BatchNorm2d(8),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(8, 16, 3, padding="same"),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding="same"),
            nn.BatchNorm2d(32),
            nn.AvgPool2d(2, 2),
        )

        self.fc1 = nn.Linear(4608, 100)
        self.lstm = nn.LSTM(1, 32, 3, batch_first=True)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.network(x)

        x = x.view(-1, 4608)
        x = self.fc1(x)
        x = x.unsqueeze(2)
        x = self.lstm(x)[0]
        x = self.fc2(x)
        x = x.squeeze(2)
        x = x[:, -8:]

        return x.squeeze(1), x.squeeze(1)


# %%
def mape_loss(preds, target):
    epsilon = 1e-8  # Small value to avoid division by zero
    return torch.mean(torch.abs((preds - target) / (target + epsilon))) * 100


# %% [markdown]
# # Train

# %% [markdown]
# ## BO

# %%
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=200)

# %% [markdown]
# ## Main train

# %%
model = DenseNet().to(DEVICE)
model = model.train()

lam_1 = torch.tensor([1.0]).to(DEVICE)
lam_2 = torch.tensor([1.0]).to(DEVICE)
# lam_3 = torch.tensor([0.01]).to(DEVICE)

lam_1 = torch.nn.Parameter(lam_1)
lam_2 = torch.nn.Parameter(lam_2)
# lam_3 = torch.nn.Parameter(lam_3)

model.register_parameter("lam_1", lam_1)
model.register_parameter("lam_2", lam_2)
# model.register_parameter('lam_3', lam_3)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

loss_best = 1000

lam_1_list = []
lam_2_list = []

for epoch in range(0, 2000):
    lam_1_list.append(lam_1.cpu().detach().numpy())
    lam_2_list.append(lam_2.cpu().detach().numpy())
    idx = 0
    for batch_X1, batch_y1, batch_y2, batch_y3 in data_loader:
        batch_X1 = batch_X1.to(DEVICE)
        batch_y1 = batch_y1.to(DEVICE)
        batch_y2 = batch_y2.to(DEVICE)
        batch_y3 = batch_y3.to(DEVICE)
        model.train()
        Cycle_pred, Param_pred, z, mu, sigma = model(batch_X1)
        # Cycle_pred,Param_pred= model(batch_X1)
        cycle_loss = mape_loss(Cycle_pred, batch_y1)
        kl_loss = torch.mean(
            torch.sum(-0.5 * (1 + sigma - torch.square(mu) - torch.exp(sigma)), dim=1)
        )

        # param_loss = mape_loss(Param_pred, batch_y3)
        Cycle_integrated = torch.cat(
            ((torch.ones(Cycle_pred.shape[0], 1) / 1400).to(DEVICE), Cycle_pred), dim=1
        )
        par_a = Param_pred[:, 0:1]
        par_b = Param_pred[:, 1:2]
        par_c = Param_pred[:, 2:3]
        physics_term = par_a * torch.exp(Cycle_integrated * par_b) + par_c
        # physics_term = par_a*(Cycle_integrated**3) + par_b*(Cycle_integrated**2) +par_c*(Cycle_integrated**1)+par_d
        # physics_loss =torch.sqrt(loss_fn(physics_term, batch_y2))
        physics_loss = mape_loss(physics_term, batch_y2)

        # loss = lam_1*cycle_loss + lam_2 *physics_loss + lam_3*param_loss - torch.log(lam_1*lam_2*lam_3)
        # loss = lam_1*cycle_loss + lam_2 *physics_loss - torch.log(lam_1*lam_2)
        loss = cycle_loss + 0.001 * kl_loss
        # loss = 1.0*cycle_loss + 3.0*physics_loss
        # loss = cycle_loss *physics_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print('epoch : ', epoch ,'/idx : ', idx , '/Cycle_loss :', round(float(cycle_loss),4), '/physics_loss :', round(float(physics_loss),4))

        model.eval()

        Cycle_pred_val, Param_pred_val, _, _, _ = model(whole_val_data_tensor)
        Cycle_integrated_val = torch.cat(
            ((torch.ones(Cycle_pred_val.shape[0], 1) / 1400).to(DEVICE), Cycle_pred_val), dim=1
        )

        par_a_val = -Param_pred_val[:, 0:1]
        par_b_val = Param_pred_val[:, 1:2]
        par_c_val = Param_pred_val[:, 2:3]

        # Param_val_loss = mape_loss(Param_pred_val, val_parameter_tensor).cpu().detach().numpy()
        RUL_val_loss = mape_loss(Cycle_pred_val, RUL_val_tensor).cpu().detach().numpy()
        physics_term_val = par_a_val * torch.exp(Cycle_integrated_val * par_b_val) + par_c_val
        # physics_term_val = par_a_val*(Cycle_integrated_val**3) * par_b_val*(Cycle_integrated_val**2)+ par_c_val*Cycle_integrated_val + par_d_val
        physics_loss_val = mape_loss(physics_term_val, Cap_val_tensor).cpu().detach().numpy()
        # product_loss = RUL_val_loss * physics_loss_val

        # print('epoch : ', epoch ,'/val_loss :', round(float(RUL_val_loss),4), '/Cycle_loss :', round(float(cycle_loss),4), '/physics_loss :', round(float(physics_loss),4), )

        if RUL_val_loss < loss_best:
            loss_best = RUL_val_loss
            torch.save(
                model.state_dict(),
                artifact_path("physics_prediction_parameter_given_memory_module_2"),
            )
            print(
                "epoch",
                epoch,
                "/val_loss :",
                round(float(RUL_val_loss), 4),
                "/Cycle_loss :",
                round(float(cycle_loss), 4),
                "/physics_loss :",
                round(float(physics_loss), 4),
                "/kl_loss :",
                round(float(kl_loss), 4),
            )

        idx += 1

# %%
plt.plot(lam_1_list, label="lam1")
plt.plot(lam_2_list, label="lam2")

plt.legend()

# %% [markdown]
# # Evaluation

# %%
model_new = DenseNet().to(DEVICE)
model_new.register_parameter("lam_1", lam_1)
model_new.register_parameter("lam_2", lam_2)
model_new.load_state_dict(
    torch.load(artifact_path("physics_prediction_parameter_given_memory_module_1"))
)
# model_new = model
model_new.eval()

RUL_train_pred, train_param_pred, z_pred, _, _ = model_new(whole_data_tensor)
RUL_train_pred = RUL_train_pred.cpu().detach().numpy()
RUL_train_pred = RUL_train_pred * 1400
train_param_pred = train_param_pred.cpu().detach().numpy()
z_pred = z_pred.cpu().detach().numpy()

RUL_val_pred, val_param_pred, z_pred_val, _, _ = model_new(whole_val_data_tensor)
RUL_val_pred = RUL_val_pred.cpu().detach().numpy()
RUL_val_pred = RUL_val_pred * 1400
val_param_pred = val_param_pred.cpu().detach().numpy()
z_pred_val = z_pred_val.cpu().detach().numpy()


RUL_test_pred, test_param_pred, z_pred_test, _, _ = model_new(whole_test_data_tensor)
RUL_test_pred = RUL_test_pred.cpu().detach().numpy()
RUL_test_pred = RUL_test_pred * 1400
test_param_pred = test_param_pred.cpu().detach().numpy()
z_pred_test = z_pred_test.cpu().detach().numpy()

# %%
cmap = plt.cm.get_cmap("jet")

normalize = mcolors.Normalize(vmin=300, vmax=1400)

for i in range(0, len(train_param_pred)):
    plt.scatter(
        z_pred[i, 0],
        z_pred[i, 1],
        color=cmap(normalize(len(bat_dict[train_bat_key[i]]["summary"]["QD"]))),
    )


for i in range(0, len(val_param_pred)):
    plt.scatter(
        z_pred_val[i, 0],
        z_pred_val[i, 1],
        color=cmap(normalize(len(bat_dict[val_bat_key[i]]["summary"]["QD"]))),
    )

for i in range(0, len(test_param_pred)):
    plt.scatter(
        z_pred_test[i, 0],
        z_pred_test[i, 1],
        color=cmap(normalize(len(bat_dict[test_bat_key[i]]["summary"]["QD"]))),
    )


scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
scalarmappaple.set_array(1400)
cb = plt.colorbar(scalarmappaple)
plt.xlabel("Z 2", fontsize=15)
plt.ylabel("Z 1", fontsize=15)

# %%
print(MAPE(RUL_true * 1400, RUL_train_pred))
print(MAPE(RUL_val * 1400, RUL_val_pred))
print(MAPE(RUL_test * 1400, RUL_test_pred))

# %%
print(MAPE(RUL_true[:, -1] * 1400, RUL_train_pred[:, -1]))
print(MAPE(RUL_val[:, -1] * 1400, RUL_val_pred[:, -1]))
print(MAPE(RUL_test[:, -1] * 1400, RUL_test_pred[:, -1]))

# %%
print(mean_squared_error(RUL_test[:, -1] * 1400, RUL_test_pred[:, -1]) ** 0.5)

# %%
P = np.linspace(1, 1400, 1400)

# %%
fig = plt.figure(figsize=(5, 5))
plt.plot(P, P, color="black")

plt.scatter(RUL_train_pred[:, -1], RUL_true[:, -1] * 1400, color="blue", label="train")
plt.scatter(RUL_val_pred[:, -1], RUL_val[:, -1] * 1400, color="green", label="val")
plt.scatter(RUL_test_pred[:, -1], RUL_test[:, -1] * 1400, color="red", label="test")
plt.xlabel("Real EOL", fontsize=15)
plt.ylabel("Predicted EOL", fontsize=15)
plt.legend()

# %%
fig = plt.figure(figsize=(5, 5))
plt.plot(P, P, color="black")

plt.scatter(RUL_train_pred, RUL_true * 1400, color="blue", label="train")
plt.scatter(RUL_val_pred, RUL_val * 1400, color="green", label="val")
plt.scatter(RUL_test_pred, RUL_test * 1400, color="red", label="test")
plt.xlabel("Real EOL", fontsize=15)
plt.ylabel("Predicted EOL", fontsize=15)
plt.legend()

# %%
index = 3
print(train_bat_key[index])
print(MAPE(RUL_true[index] * 1400, RUL_train_pred[index]))


plt.plot(np.append(0, RUL_train_pred[index]), Cap_true[index])
plt.scatter(np.append(0, RUL_train_pred[index]), Cap_true[index], label="Predict")
plt.plot(np.append(0, RUL_true[index]) * 1400, Cap_true[index])
plt.scatter(np.append(0, RUL_true[index]) * 1400, Cap_true[index], label="True")
data_length = len(bat_dict[train_bat_key[index]]["summary"]["QD"])
X = np.linspace(1, data_length, data_length, dtype=int)
# plt.plot(X, bat_dict[train_bat_key[index]]['summary']['QD'], label = 'Real Capcity')

fitted_function = (
    (train_param_pred[index][0] * (X**3)) / (1400**3)
    + (train_param_pred[index][1] * (X**2)) / (1400**2)
    + (train_param_pred[index][2] * X) / 1400
    + train_param_pred[index][3]
)
fitted_function = (
    (train_param_pred[index][0] * (X**3)) / (1400**3)
    + (train_param_pred[index][1] * (X**2)) / (1400**2)
    + (train_param_pred[index][2] * X) / 1400
    + train_param_pred[index][3]
)
print(MAPE(fitted_function, bat_dict[train_bat_key[index]]["summary"]["QD"]))

# plt.plot(X, fitted_function, label='fitted equation')

plt.ylabel("Capcity", fontsize=15)
plt.xlabel("Cycle", fontsize=15)

plt.ylim([0.87, 1.12])
plt.legend()

# %%
index = 10

print(val_bat_key[index])
print(MAPE(RUL_val[index] * 1400, RUL_val_pred[index]))
print(RUL_val_pred[index])


plt.plot(np.append(index, RUL_val_pred[index]), Cap_val[index])
plt.scatter(np.append(0, RUL_val_pred[index]), Cap_val[index], label="Prediction")
plt.plot(np.append(0, RUL_val[index]) * 1400, Cap_val[index])
plt.scatter(np.append(0, RUL_val[index]) * 1400, Cap_val[index], label="True")
data_length = len(bat_dict[val_bat_key[index]]["summary"]["QD"])
X = np.linspace(1, data_length, data_length, dtype=int)
plt.plot(X, bat_dict[val_bat_key[index]]["summary"]["QD"], label="Real Capcity")


# fitted_function = (val_param_pred[index][0] * (X**3))/(1400**3) + (val_param_pred[index][1] * (X**2))/(1400**2) + (val_param_pred[index][2] * X)/1400 + val_param_pred[index][3]
# print(MAPE(fitted_function, bat_dict[val_bat_key[index]]['summary']['QD']))

# plt.plot(X, fitted_function)

plt.ylabel("Capcity", fontsize=15)
plt.xlabel("Cycle", fontsize=15)

plt.ylim([0.87, 1.12])
plt.legend()

# %%
for index in range(0, len(RUL_val)):
    plt.plot(np.append(index, RUL_val_pred[index]), Cap_val[index])

plt.ylabel("Capcity", fontsize=15)
plt.xlabel("Cycle", fontsize=15)

# %%
index = 6
print(test_bat_key[index])
print(MAPE(RUL_test[index] * 1400, RUL_test_pred[index]))
print(RUL_test_pred[index])

plt.plot(np.append(index, RUL_test_pred[index]), Cap_test[index])
plt.scatter(np.append(0, RUL_test_pred[index]), Cap_test[index], label="Predict")
plt.plot(np.append(0, RUL_test[index] * 1400), Cap_test[index])
plt.scatter(np.append(0, RUL_test[index] * 1400), Cap_test[index], label="True")
data_length = len(bat_dict[test_bat_key[index]]["summary"]["QD"])
X = np.linspace(1, data_length, data_length, dtype=int)
plt.plot(X, bat_dict[test_bat_key[index]]["summary"]["QD"], label="Real Capcity")

# fitted_function = (test_param_pred[index][0] * (X**3))/(1400**3) + (test_param_pred[index][1] * (X**2))/(1400**2) + (test_param_pred[index][2] * X)/1400 + test_param_pred[index][3]
fitted_function = (
    test_param_pred[index][0] * np.exp(test_param_pred[index][1] * X / 1400)
    + test_param_pred[index][2]
)
plt.plot(X, fitted_function, label="fitted equation")

plt.ylabel("Capcity", fontsize=15)
plt.xlabel("Cycle", fontsize=15)
print(MAPE(fitted_function, bat_dict[test_bat_key[index]]["summary"]["QD"]))
# plt.ylim([0.87, 1.1])
plt.legend()

# %% [markdown]
# # Beta Test

# %%
np.append(1, RUL_true[0] * 1400)


# %%
def model_function(params, RUL_true_temp):
    return (
        params[0] * (RUL_true_temp**3)
        + params[1] * (RUL_true_temp**2)
        + params[2] * (RUL_true_temp)
        + params[3]
    )


# %%
def model_function(params, RUL_true_temp):
    return params[0] * np.exp(RUL_true_temp * params[1]) + params[2] * np.exp(
        RUL_true_temp * params[3]
    )


# %%
def model_function(params, RUL_true_temp):
    return params[0] * np.exp(RUL_true_temp * params[1]) + params[2]


# %%
RUL_true_temp = np.append((1 / 1400), RUL_true[3])
Cap_temp = Cap_true[3]


# %%
def objective_function(params):
    y_predicted = model_function(params, RUL_true_temp)
    difference = Cap_temp - y_predicted
    error = np.abs(difference)
    return np.mean(error)


# %%
RUL_true_temp = np.append((1 / 1400), RUL_true[5])
Cap_temp = Cap_true[5]


def model_function(params, RUL_true_temp):
    return params[0] * np.exp(RUL_true_temp * params[1]) + params[2]


initial_params = [-5, 0, 1]

print(result.x[0], result.x[1], result.x[2])

result = minimize(objective_function, initial_params)

X = np.linspace(0, RUL_true_temp.max(), 1000)

plt.scatter(RUL_true_temp, Cap_temp)
plt.plot(X, result.x[0] * np.exp(X * result.x[1]) + result.x[2])

# %%
index = 7

RUL_true_temp = np.append((1 / 1400), RUL_val[index])
Cap_temp = Cap_val[index]


def model_function(params, RUL_true_temp):
    return params[0] * np.exp(RUL_true_temp * params[1]) + params[2] * np.exp(
        RUL_true_temp * params[3]
    )


initial_params = [-3, 11, 0.1, 0.1]
# initial_params = [0.1,,0.1,0.1]

# Parameter Optimization

result = minimize(objective_function, initial_params, method="BFGS")

print(result.x[0], result.x[1], result.x[2], result.x[3])

X = np.linspace(0, RUL_true_temp.max(), 1000)

plt.scatter(RUL_true_temp, Cap_temp)
plt.plot(X, result.x[0] * np.exp(X * result.x[1]) + result.x[2] * np.exp(X * result.x[3]))
print(result.fun)

# %%
index = 12

RUL_true_temp = np.append((1 / 1400), RUL_true[index])
Cap_temp = Cap_true[index]


def model_function(params, RUL_true_temp):
    return (
        params[0] * (RUL_true_temp**3)
        + params[1] * (RUL_true_temp**2)
        + params[2] * (RUL_true_temp)
        + params[3]
    )


initial_params = [0.01, 0.01, 0.01, 0.01]

# Parameter Optimization

result = minimize(objective_function, initial_params)

X = np.linspace(0, RUL_true_temp.max(), 1000)

plt.scatter(RUL_true_temp, Cap_temp, color="black", label="REF")
plt.plot(
    X, result.x[0] * (X**3) + result.x[1] * (X**2) + result.x[2] * (X) + result.x[3], label="3rd"
)


def model_function(params, RUL_true_temp):
    return (
        params[0] * (RUL_true_temp**4)
        + params[1] * (RUL_true_temp**3)
        + params[2] * (RUL_true_temp**2)
        + params[3] * RUL_true_temp
        + params[4]
    )


initial_params = [0.01, 0.01, 0.01, 0.01, 0.01]

# Parameter Optimization

result = minimize(objective_function, initial_params)

X = np.linspace(0, RUL_true_temp.max(), 1000)

plt.plot(
    X,
    result.x[0] * (X**4)
    + result.x[1] * (X**3)
    + result.x[2] * (X**2)
    + result.x[3] * X
    + result.x[4],
    label="4th",
)


def model_function(params, RUL_true_temp):
    return (
        params[0] * (RUL_true_temp**5)
        + params[1] * (RUL_true_temp**4)
        + params[2] * (RUL_true_temp**3)
        + params[3] * (RUL_true_temp**2)
        + params[4] * RUL_true_temp
        + params[5]
    )


initial_params = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

# Parameter Optimization

result = minimize(objective_function, initial_params)

X = np.linspace(0, RUL_true_temp.max(), 1000)

plt.plot(
    X,
    result.x[0] * (X**5)
    + result.x[1] * (X**4)
    + result.x[2] * (X**3)
    + result.x[3] * (X**2)
    + result.x[4] * X
    + result.x[5],
    label="5th",
)

plt.legend()

# %%
index = 2

RUL_true_temp = RUL_true[index]
Cap_temp = Cap_true[index][1:]


def model_function(params, RUL_true_temp):
    return (
        params[0] * (RUL_true_temp**3)
        + params[1] * (RUL_true_temp**2)
        + params[2] * (RUL_true_temp)
        + params[3]
    )


initial_params = [0.01, 0.01, 0.01, 0.01]

# Parameter Optimization

result = minimize(objective_function, initial_params)

X = np.linspace(RUL_true_temp.min(), RUL_true_temp.max(), 1000)

plt.scatter(RUL_true_temp, Cap_temp, color="black", label="REF")
plt.plot(
    X, result.x[0] * (X**3) + result.x[1] * (X**2) + result.x[2] * (X) + result.x[3], label="3rd"
)

print(result.x[0], result.x[1], result.x[2], result.x[3])


def model_function(params, RUL_true_temp):
    return (
        params[0] * (RUL_true_temp**4)
        + params[1] * (RUL_true_temp**3)
        + params[2] * (RUL_true_temp**2)
        + params[3] * RUL_true_temp
        + params[4]
    )


initial_params = [0.01, 0.01, 0.01, 0.01, 0.01]

# Parameter Optimization

result = minimize(objective_function, initial_params)

X = np.linspace(RUL_true_temp.min(), RUL_true_temp.max(), 1000)

plt.plot(
    X,
    result.x[0] * (X**4)
    + result.x[1] * (X**3)
    + result.x[2] * (X**2)
    + result.x[3] * X
    + result.x[4],
    label="4th",
)


def model_function(params, RUL_true_temp):
    return (
        params[0] * (RUL_true_temp**5)
        + params[1] * (RUL_true_temp**4)
        + params[2] * (RUL_true_temp**3)
        + params[3] * (RUL_true_temp**2)
        + params[4] * RUL_true_temp
        + params[5]
    )


initial_params = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

# Parameter Optimization

result = minimize(objective_function, initial_params)

X = np.linspace(RUL_true_temp.min(), RUL_true_temp.max(), 1000)


plt.plot(
    X,
    result.x[0] * (X**5)
    + result.x[1] * (X**4)
    + result.x[2] * (X**3)
    + result.x[3] * (X**2)
    + result.x[4] * X
    + result.x[5],
    label="5th",
)

plt.legend()

# %%
index = 46

RUL_true_temp = RUL_true[index]
Cap_temp = Cap_true[index][1:]


def model_function(params, RUL_true_temp):
    return params[0] * (RUL_true_temp**2) + params[1] * (RUL_true_temp**1) + params[2]


initial_params = [0.01, 0.2, 0.01]

# Parameter Optimization

result = minimize(objective_function, initial_params)

X = np.linspace(RUL_true_temp.min(), RUL_true_temp.max(), 1000)

# plt.scatter(RUL_true_temp, Cap_temp)
plt.plot(X, result.x[0] * (X**2) + result.x[1] * (X**1) + result.x[2], label="2rd")


def model_function(params, RUL_true_temp):
    return (
        params[0] * (RUL_true_temp**3)
        + params[1] * (RUL_true_temp**2)
        + params[2] * (RUL_true_temp)
        + params[3]
    )


initial_params = [-0.3, -2, 0.3, 1.6]

# Parameter Optimization

result = minimize(objective_function, initial_params)

X = np.linspace(RUL_true_temp.min(), RUL_true_temp.max(), 1000)

plt.scatter(RUL_true_temp, Cap_temp, color="black", label="REF")
plt.plot(
    X, result.x[0] * (X**3) + result.x[1] * (X**2) + result.x[2] * (X) + result.x[3], label="3rd"
)

print(result.x[0], result.x[1], result.x[2], result.x[3])

# %%
index = 45

RUL_true_temp = RUL_true[index]
Cap_temp = Cap_true[index][1:]


def model_function(params, RUL_true_temp):
    return (
        params[0] * (RUL_true_temp**3)
        + params[1] * (RUL_true_temp**2)
        + params[2] * (RUL_true_temp)
        + params[3]
        + params[4] * np.tanh(params[5] * (RUL_true_temp))
    )


initial_params = [-0.1, -2, 0, 0, 0.1, 0.5]

# Parameter Optimization

result = minimize(objective_function, initial_params)

X = np.linspace(RUL_true_temp.min(), RUL_true_temp.max(), 1000)

plt.scatter(RUL_true_temp, Cap_temp, color="black", label="REF")
plt.plot(
    X,
    result.x[0] * (X**3)
    + result.x[1] * (X**2)
    + result.x[2] * (X)
    + result.x[3]
    + result.x[4] * np.tanh(X * result.x[5]),
    label="3rd",
)

print(result.x[0], result.x[1], result.x[2], result.x[3], result.x[4], result.x[5])
