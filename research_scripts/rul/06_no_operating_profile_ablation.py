# %% [markdown]
# # Consolidated research script
#
# Method group **G17**: No-OP ablation. Architecture: CNN-Transformer encoder. Method tags: operating-profile ablation.
#
# This copy preserves the research implementation while removing saved outputs, execution counters, pinned GPU selection, and scratch-only cells. Closely related variants are summarized in the research script index. During cleanup, 0 syntactically invalid scratch cell(s) and 2 display-only scratch cell(s) were omitted.

# %%
from battery_soh.data import resolve_data_dir
from battery_soh.evaluation import MAPE, RMSPE
from battery_soh.paths import artifact_path
from battery_soh.preprocessing import interpolate_timeseries
from battery_soh.raw_data import load_battery_dictionary
from sklearn.metrics import mean_absolute_error
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

# %%
batch_keys = [*bat_dict.keys()]
print(len(batch_keys))

# %%
batch_keys = [*bat_dict.keys()]

batch_keys.remove("b2c9")
batch_keys.remove("b2c8")
batch_keys.remove("b2c7")
batch_keys.remove("b2c1")
batch_keys.remove("b2c4")
batch_keys.remove("b1c11")
# batch_keys.remove('b2c17')
batch_keys.remove("b2c35")
# batch_keys.remove('b2c25')
# batch_keys.remove('b2c33')
# batch_keys.remove('b2c41')
batch_keys.remove("b2c6")
batch_keys.remove("b2c3")
batch_keys.remove("b2c2")
# batch_keys.remove('b2c24')
batch_keys.remove("b3c38")
batch_keys.remove("b3c46")
# batch_keys.remove('b3c18')
batch_keys.remove("b1c48")
batch_keys.remove("b1c47")

# %%
# batch_keys = [*bat_dict.keys()]
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
EOL_array = []
early_key = []
mid_key = []
long_key = []

for battery_key in selected_keys:
    for k in range(0, len(bat_dict[battery_key]["summary"]["QD"])):
        cycle_str = int(k)
        QD = bat_dict[battery_key]["summary"]["QD"][int(cycle_str)]
        temp_SOH = (QD / 1.1) * 100
        cycle_str = cycle_str + 1

        if temp_SOH < 80.3:
            EOL_Cycle = int(cycle_str) + 1
            EOL_array.append(EOL_Cycle)

            if EOL_Cycle < 600:
                early_key.append(battery_key)
            elif (EOL_Cycle > 600) & (EOL_Cycle < 890):
                mid_key.append(battery_key)
            elif (EOL_Cycle > 900) & (EOL_Cycle < 1500):
                long_key.append(battery_key)
            break

# %%
random.seed(29)

random.shuffle(early_key)
random.shuffle(mid_key)
random.shuffle(long_key)

train_bat_key = early_key[:26]
train_bat_key.extend(mid_key[:26])
train_bat_key.extend(long_key[:20])

val_bat_key = early_key[26:32]
val_bat_key.extend(mid_key[26:32])
val_bat_key.extend(long_key[20:25])

test_bat_key = early_key[32:]
test_bat_key.extend(mid_key[32:])
test_bat_key.extend(long_key[25:])

# %%
time_value = 5
dis_time_value = 5
inter_value = 300


# %%
def data_maker(inter_list, window_size):

    train_x = []
    train_x_OP = []
    train_y = []

    Cycle_index = 0
    idx = 0

    origin_df = []
    cleared_df = []

    for battery_key in inter_list:
        battery_key_array = []
        Cycle_true = []
        Char_I_var_array = []
        Char_V_var_array = []
        Disc_V_var_array = []
        Char_I_mean_array = []
        Char_V_mean_array = []
        Disc_V_mean_array = []
        SOH_true = []
        RUL = []
        idx += 1
        for k in range(0, len(bat_dict[battery_key]["summary"]["QD"])):
            cycle_str = int(k)
            QD = bat_dict[battery_key]["summary"]["QD"][int(cycle_str)]
            temp_SOH = (QD / 1.1) * 100
            cycle_str = cycle_str + 1

            if temp_SOH < 80.3:
                EOL_Cycle = int(cycle_str) + 1
                # print(temp_SOH, EOL_Cycle, idx)
                break

        for cycle_num in range(0, len(bat_dict[battery_key]["summary"]["cycle"])):
            temporal_data = []
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
                    Voltage_data = bat_dict[battery_key]["cycles"][cycle_num]["V"][
                        i : i + final_idx
                    ]

                    df = pd.DataFrame(
                        {"time": time_data, "Current": Current_data, "Voltage": Voltage_data}
                    )

                    df = df.drop_duplicates(["time"])

                    data = np.array(df.T)

                    Current_data = interpolate_timeseries(data[1], inter_value)
                    Voltage_data = interpolate_timeseries(data[2], inter_value)
                    Voltage_data = (Voltage_data - 3.37) / (3.62 - 3.37)
                    Char_I_var = np.var(Current_data)
                    Char_V_var = np.var(Voltage_data)
                    Char_I_mean = np.mean(Current_data)
                    Char_V_mean = np.mean(Voltage_data)

                    temporal_data.append(Current_data)
                    temporal_data.append(Voltage_data)

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
                        if total_time > dis_time_value:
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
                    Disc_V_var = np.var(Discharge_Voltage)
                    Disc_V_mean = np.mean(Discharge_Voltage)

                    temporal_data.append(Discharge_Voltage)

                    break

            if np.array(temporal_data).shape == (3, inter_value):
                battery_key_array.append(battery_key)
                Cycle_true.append(bat_dict[battery_key]["summary"]["cycle"][int(cycle_num)])
                # RUL.append(EOL_Cycle - bat_dict[battery_key]['summary']['cycle'][int(cycle_num)])
                RUL.append(EOL_Cycle - int(cycle_num))
                SOH_true.append(bat_dict[battery_key]["summary"]["QD"][int(cycle_num)])
                Char_I_var_array.append(Char_I_var)
                Char_I_mean_array.append(Char_I_mean)
                Char_V_var_array.append(Char_V_var)
                Char_V_mean_array.append(Char_V_mean)
                Disc_V_var_array.append(Disc_V_var)
                Disc_V_mean_array.append(Disc_V_mean)

        train_df = pd.DataFrame(
            {
                "battery_key": battery_key_array,
                "Cycle": Cycle_true,
                "RUL": np.array(RUL) / 1500,
                "SOH": np.array(SOH_true),
                "Char_I_var": (np.array(Char_I_var_array) / 0.152) + 1e-6,
                "Char_I_mean": np.array(Char_I_mean_array) + 1e-6,
                "Char_V_mean": (np.array(Char_V_mean_array) / 0.93) + 1e-6,
                "Char_V_var": (np.array(Char_V_var_array) / 0.085) + 1e-6,
                "Disc_V_mean": (np.array(Disc_V_mean_array) / 0.58) + 1e-6,
                "Disc_V_var": (np.array(Disc_V_var_array) / 0.06) + 1e-6,
            }
        )

        origin_df.append(train_df.copy())

        filter_window = 10

        for column_name in list(train_df)[4:]:
            filtered = []

            for k in range(0, filter_window - 1):
                filtered.append(float(np.array(train_df[column_name][k], dtype=np.float32)))

            for i in range(len(train_df[column_name]) - filter_window + 1):
                filtered.append(
                    np.array(train_df[column_name][i : i + filter_window], dtype=np.float32).mean()
                )

            train_df[column_name] = filtered

        window_size = window_size

        for i in range(len(train_df) - window_size + 1):
            # for i in range(100):
            # print(float(np.array(train_df.iloc[i+window_size-1, 3], dtype=np.float32)))
            train_x.append(np.array(train_df.iloc[i : i + window_size, 4:], dtype=np.float32).T)
            train_y.append(np.array(train_df.iloc[i + window_size - 1, 2], dtype=np.float32))

            if EOL_Cycle < 600:
                train_x_OP.append(np.array([1, 0, 0]))
            elif (EOL_Cycle > 600) & (EOL_Cycle < 900):
                train_x_OP.append(np.array([0, 1, 0]))
            elif (EOL_Cycle > 900) & (EOL_Cycle < 1500):
                train_x_OP.append(np.array([0, 0, 1]))

        cleared_df.append(train_df)

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    train_x_OP = np.array(train_x_OP)

    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1], train_x.shape[2]))
    train_x_OP = np.reshape(train_x_OP, (train_x_OP.shape[0], 3))

    return train_x, train_x_OP, train_y, cleared_df, origin_df


# %%
train_x, train_x_OP, train_y, cleared_df, origin_df = data_maker(train_bat_key, 30)

# %%
print(train_x_OP.shape)

# %%
interest = "Char_I_var"
sequence = 5


plt.plot(origin_df[sequence][interest], color="black", label="origin", linewidth=2)
plt.plot(cleared_df[sequence][interest], color="red", label="filtered", linewidth=2)
plt.xlabel("Cycle", fontsize=15)
plt.legend(fontsize=15)

# %%
val_x, val_x_OP, val_y, _, _ = data_maker(val_bat_key, 30)

# %%
val_x_tensor = torch.tensor(val_x, dtype=torch.float32).to(DEVICE)
val_x_OP_tensor = torch.tensor(val_x_OP, dtype=torch.float32).to(DEVICE)
val_y_tensor = torch.tensor(val_y, dtype=torch.float32).to(DEVICE)


# %%
class MyDataset(Dataset):
    def __init__(self, X1, X2, y1):
        self.X1 = X1
        self.X2 = X2
        self.y1 = y1

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y1[idx]


dataset = MyDataset(
    torch.tensor(train_x, dtype=torch.float32),
    torch.tensor(train_x_OP, dtype=torch.float32),
    torch.tensor(train_y, dtype=torch.float32),
)

data_loader = DataLoader(dataset, batch_size=4000, shuffle=True)


# %%
class SOH_model(nn.Module):
    def __init__(self, in_features, time_length, first_head, first_layer):
        super().__init__()
        self.in_features = in_features
        self.time_length = time_length
        self.first_head = first_head
        self.first_layer = first_layer

        self.linear_first = nn.Linear(self.time_length, 100)

        self.conv_1 = nn.Conv2d(1, 10, kernel_size=(1, 10), padding="same")
        self.conv_2 = nn.Conv2d(10, 10, kernel_size=(1, 10), padding="same")
        self.conv_3 = nn.Conv2d(10, 20, kernel_size=(1, 10), padding="same")
        self.conv_4 = nn.Conv2d(20, 10, kernel_size=(1, 5), padding="same")
        self.conv_5 = nn.Conv2d(10, 1, kernel_size=(1, 5), padding="same")
        self.conv_6 = nn.Conv2d(
            1, 1, kernel_size=(self.in_features, self.in_features), padding="same"
        )

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=100, nhead=self.first_head, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=self.first_layer
        )

        self.linear1 = nn.Linear(100 * self.in_features, 100)
        self.linear2 = nn.Linear(100, 1)

        # Trunk Net ---------------------------
        self.Linear_T1 = nn.Linear(3, 100)
        self.Linear_T2 = nn.Linear(100, 100)
        self.Linear_T3 = nn.Linear(100, 100)
        self.Linear_T4 = nn.Linear(100, 100)

    def forward(self, x):
        x = self.linear_first(x)
        x1 = x
        # MSDCNN
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
        x = x + x1  # Skip connection
        x = self.conv_6(x)
        x = F.leaky_relu(x)

        x = x.squeeze(1)

        # Multi Head Attention
        x = self.encoder(x)
        x = x.reshape(-1, 100 * self.in_features)
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)

        return x.squeeze(1)


# %%
def mape_loss(preds, target):
    epsilon = 1e-8  # Small value to avoid division by zero
    return torch.mean(torch.abs((preds - target) / (target + epsilon))) * 100


# %%
def objective(trial):
    # Define the hyperparameters to optimize
    first_head = trial.suggest_categorical("first_head", [1, 2, 4, 5, 10])
    first_layer = trial.suggest_int("first_layer", 2, 5)

    model = SOH_model(6, 30, first_head, first_layer).to(DEVICE)
    model = model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.L1Loss()

    SOH_loss_best = 1000

    for epoch in range(0, 200):
        idx = 0
        train_loss = []
        for batch_X1, batch_X2, batch_y1 in data_loader:
            model.train()
            batch_X1 = batch_X1.to(DEVICE)
            batch_X2 = batch_X2.to(DEVICE)
            batch_y1 = batch_y1.to(DEVICE)
            SOH_pred = model(batch_X1, batch_X2)
            loss = torch.sqrt(loss_fn(SOH_pred, batch_y1))
            train_loss.append(float(loss))
            # loss = mape_loss(SOH_pred, batch_y1)
            # loss =loss_fn(SOH_pred, batch_y1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                SOH_pred_val = model(val_x_tensor, val_x_SOH_tensor)
                SOH_pred_val = SOH_pred_val.cpu().detach().numpy()
                SOH_val_loss = mean_squared_error(val_y, SOH_pred_val) ** 0.5
                # SOH_val_loss = mape_loss(val_y_tensor, SOH_pred_val).cpu().detach().numpy()
                SOH_val_loss = float(SOH_val_loss)

            if SOH_val_loss < SOH_loss_best:
                SOH_loss_best = SOH_val_loss

    if not np.isnan(SOH_loss_best):
        return SOH_loss_best
    else:
        return float("inf")


# %%
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# %%
model = SOH_model(6, 30, 2, 4).to(DEVICE)
model = model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.L1Loss()

SOH_loss_best = 1000


for epoch in range(0, 300):
    idx = 0
    train_loss = []
    for (
        batch_X1,
        batch_X2,
        batch_y1,
    ) in data_loader:
        model.train()
        batch_X1 = batch_X1.to(DEVICE)
        batch_X2 = batch_X2.to(DEVICE)
        batch_y1 = batch_y1.to(DEVICE)
        SOH_pred = model(batch_X1)
        # loss = torch.sqrt(loss_fn(SOH_pred, batch_y1))
        loss = torch.sqrt(loss_fn(SOH_pred, batch_y1))
        train_loss.append(float(loss))
        # loss = mape_loss(SOH_pred, batch_y1)
        # loss =loss_fn(SOH_pred, batch_y1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            SOH_pred_val = model(val_x_tensor)
            SOH_pred_val = SOH_pred_val.cpu().detach().numpy()
            SOH_val_loss = mean_squared_error(val_y, SOH_pred_val) ** 0.5
            # SOH_val_loss = mape_loss(val_y_tensor, SOH_pred_val).cpu().detach().numpy()
            SOH_val_loss = float(SOH_val_loss)

        if SOH_val_loss < SOH_loss_best:
            SOH_loss_best = SOH_val_loss
            torch.save(model.state_dict(), artifact_path("RUL_prediction_without_OP"))

    train_loss = np.array(train_loss).mean()
    print(
        "epoch : ",
        epoch,
        "/idx : ",
        idx,
        "/train_loss :",
        round(float(train_loss), 4),
        "/SOH_val_loss : ",
        round(SOH_loss_best, 4),
    )

# %%
len(test_bat_key)

# %%
rmse_list = []
mape_list = []

mode = "SOH"

fig = plt.figure(figsize=(15, 15))

for k in range(0, 9):
    # for k in range(10,len(test_bat_key)):
    battery_list = [test_bat_key[k]]

    model_new = SOH_model(6, 30, 2, 4).to(DEVICE)

    model_new.load_state_dict(torch.load(artifact_path("RUL_prediction_without_OP")))

    test_x, test_x_OP, test_y, _, _ = data_maker(battery_list, 30)

    test_x_tensor = torch.tensor(test_x, dtype=torch.float32).to(DEVICE)
    test_x_OP_tensor = torch.tensor(test_x_OP, dtype=torch.float32).to(DEVICE)
    test_y_tensor = torch.tensor(test_y, dtype=torch.float32).to(DEVICE)

    test_y = test_y * 1500

    with torch.no_grad():
        model_new.eval()
        SOH_test_pred = model_new(test_x_tensor)
        SOH_test_pred = SOH_test_pred.cpu().detach().numpy() * 1500

    if battery_key == "b2c14":
        SOH_test_pred = np.delete(SOH_test_pred, [246])
        # Cycle_test = np.delete(Cycle_test, [246])
        SOH_test = np.delete(SOH_test, [246])

    plot_num = 331 + k
    # plot_num = 331 + (k-10)

    if mode == "SOH":
        soh_rmse = mean_squared_error(test_y, SOH_test_pred) ** 0.5
        soh_mape = (
            mean_absolute_error(np.array(test_y), np.array(SOH_test_pred)) / test_y[0]
        ) * 100
        EOL_mape = MAPE(np.array(test_y[0]), np.array(SOH_test_pred[0]))

        rmse_list.append(soh_rmse)
        mape_list.append(soh_mape)
        data_length = len(np.array(test_y))

        plt.subplot(plot_num)
        plt.plot(SOH_test_pred, color="r", label="Predicted RUL")
        plt.plot(test_y, color="b", label="real RUL")
        plt.text(1, data_length * 0.18, "EOL APE(%) : " + str(round(EOL_mape, 2)), fontsize=16)
        plt.text(1, data_length * 0.10, "RMSE : " + str(round(soh_rmse, 2)), fontsize=16)
        plt.text(1, data_length * 0.02, "APE(%) : " + str(round(soh_mape, 2)), fontsize=16)

        plt.title(battery_list[0], fontsize=15)
        plt.ylabel("RUL", fontsize=15)
        plt.xlabel("cycle", fontsize=15)
        plt.legend()


fig.tight_layout(pad=3.0)
plt.show()

# %%
test_x, test_x_SOH, test_y, df, _ = data_maker(["b2c6"], 30)

# %%
plt.plot(df[0]["Char_I_var"])

# %%
fig, ax = plt.subplots()

rmse_arr = np.array(rmse_list)
rmse_avg = np.mean(rmse_arr)

mape_arr = np.array(mape_list)
mape_avg = np.mean(mape_arr)

ax.boxplot([rmse_arr, mape_arr])
plt.xticks([1, 2], ["RMSE(%)", "MAPE"])
plt.text(0.6, 10, "[RMSE_AVG]\n   " + str(round(rmse_avg, 3)) + "%")
plt.text(1.6, 10, "[MAPE_AVG]\n   " + str(round(mape_avg, 3)) + "%")

plt.show()

# %%
model_new = SOH_model(4, inter_value, 2, 2, 2, 7).to(DEVICE)

model_new.register_parameter("par_a", par_a)
model_new.register_parameter("par_b", par_b)
model_new.register_parameter("par_c", par_c)
model_new.register_parameter("par_d", par_d)

model_new.register_parameter("lam_1", lam_1)
model_new.register_parameter("lam_2", lam_2)
model_new.register_parameter("lam_3", lam_3)
# model_new.register_parameter('lam_4', lam_4)

model_new.load_state_dict(torch.load(artifact_path("New_normalization_180_SOH_1")))
model_new.eval()

# %%
whole_val_data = []
Cycle_val = []
SOH_val = []
Cycle_index = 0

for battery_key in test_bat_key:
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

for battery_key in test_bat_key:
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
print("Cycle_MAPE : ", str(np.round(Cycle_mape, 4)), " %")
