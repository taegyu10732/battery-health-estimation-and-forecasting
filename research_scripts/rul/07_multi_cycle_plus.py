# %% [markdown]
# # Consolidated research script
#
# Method group **G18**: Multi-cycle additive fusion. Architecture: CNN-Transformer encoder. Method tags: multi-cycle|additive fusion.
#
# This copy preserves the research implementation while removing saved outputs, execution counters, pinned GPU selection, and scratch-only cells. Closely related variants are summarized in the research script index.

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
random.seed(319)

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
selected_keys.remove("b2c8")
selected_keys.remove("b2c7")
selected_keys.remove("b2c1")
selected_keys.remove("b2c4")
selected_keys.remove("b2c17")
selected_keys.remove("b2c25")
selected_keys.remove("b2c33")
selected_keys.remove("b2c41")
selected_keys.remove("b2c6")
selected_keys.remove("b2c3")
selected_keys.remove("b2c2")
selected_keys.remove("b2c24")
selected_keys.remove("b3c46")
selected_keys.remove("b3c18")
selected_keys.remove("b1c48")
selected_keys.remove("b1c47")

# %%
random.shuffle(selected_keys)
train_bat_key = selected_keys[:65]
val_bat_key = selected_keys[65:85]
test_bat_key = selected_keys[85:]

# %%
time_value = 5
inter_value = 300


# %%
def df_loader(inter_key):
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

    Cycle_index = 0
    idx = 0

    for battery_key in inter_key:
        idx += 1
        for k in range(0, len(bat_dict[battery_key]["summary"]["QD"])):
            cycle_str = int(k)
            QD = bat_dict[battery_key]["summary"]["QD"][int(cycle_str)]
            temp_SOH = (QD / 1.1) * 100
            cycle_str = cycle_str + 1

            if temp_SOH < 80.2:
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
            "SOH": SOH_true,
            "Char_I_var": (np.array(Char_I_var_array) / 0.152) + 1e-6,
            "Char_V_var": (np.array(Char_V_var_array) / 0.085) + 1e-6,
            "Disc_V_var": (np.array(Disc_V_var_array) / 0.06) + 1e-6,
            "Char_I_mean": np.array(Char_I_mean_array) + 1e-6,
            "Char_V_mean": (np.array(Char_V_mean_array) / 0.93) + 1e-6,
            "Disc_V_mean": (np.array(Disc_V_mean_array) / 0.58) + 1e-6,
        }
    )
    return train_df


# %%
train_df = df_loader(train_bat_key)
val_df = df_loader(val_bat_key)

# %%
train_x_long = []
train_x_mid = []
train_x_short = []
train_y = []

for bat_key in train_bat_key:
    cell_df = train_df[train_df["battery_key"] == bat_key]
    for i in range(len(cell_df) - 29):
        train_x_long.append(np.array(cell_df.iloc[i : i + 30, 4:], dtype=np.float32).T)
        train_x_mid.append(np.array(cell_df.iloc[i + 10 : i + 30, 4:], dtype=np.float32).T)
        train_x_short.append(np.array(cell_df.iloc[i + 20 : i + 30, 4:], dtype=np.float32).T)
        train_y.append(np.array(cell_df.iloc[i + 29, 2], dtype=np.float32))


train_x_long = np.array(train_x_long)
train_x_mid = np.array(train_x_mid)
train_x_short = np.array(train_x_short)
train_y = np.array(train_y)

train_x_long = np.reshape(
    train_x_long, (train_x_long.shape[0], 1, train_x_long.shape[1], train_x_long.shape[2])
)
train_x_mid = np.reshape(
    train_x_mid, (train_x_mid.shape[0], 1, train_x_mid.shape[1], train_x_mid.shape[2])
)
train_x_short = np.reshape(
    train_x_short, (train_x_short.shape[0], 1, train_x_short.shape[1], train_x_short.shape[2])
)

# %%
print(train_x_long.shape)
print(train_x_mid.shape)
print(train_x_short.shape)

# %%
val_x_long = []
val_x_mid = []
val_x_short = []
val_y = []

for bat_key in val_bat_key:
    cell_df = val_df[val_df["battery_key"] == bat_key]
    for i in range(len(cell_df) - 29):
        val_x_long.append(np.array(cell_df.iloc[i : i + 30, 4:], dtype=np.float32).T)
        val_x_mid.append(np.array(cell_df.iloc[i + 10 : i + 30, 4:], dtype=np.float32).T)
        val_x_short.append(np.array(cell_df.iloc[i + 20 : i + 30, 4:], dtype=np.float32).T)
        val_y.append(np.array(cell_df.iloc[i + 29, 2], dtype=np.float32))


val_x_long = np.array(val_x_long)
val_x_mid = np.array(val_x_mid)
val_x_short = np.array(val_x_short)
val_y = np.array(val_y)

val_x_long = np.reshape(
    val_x_long, (val_x_long.shape[0], 1, val_x_long.shape[1], val_x_long.shape[2])
)
val_x_mid = np.reshape(val_x_mid, (val_x_mid.shape[0], 1, val_x_mid.shape[1], val_x_mid.shape[2]))
val_x_short = np.reshape(
    val_x_short, (val_x_short.shape[0], 1, val_x_short.shape[1], val_x_short.shape[2])
)

# %%
print(val_x_long.shape)
print(val_x_mid.shape)
print(val_x_short.shape)

# %%
train_x_long_tensor = torch.tensor(train_x_long, dtype=torch.float32)
train_x_mid_tensor = torch.tensor(train_x_mid, dtype=torch.float32)
train_x_short_tensor = torch.tensor(train_x_short, dtype=torch.float32)
train_y_tensor = torch.tensor(train_y, dtype=torch.float32)

# %%
val_x_long_tensor = torch.tensor(val_x_long, dtype=torch.float32).to(DEVICE)
val_x_mid_tensor = torch.tensor(val_x_mid, dtype=torch.float32).to(DEVICE)
val_x_short_tensor = torch.tensor(val_x_short, dtype=torch.float32).to(DEVICE)
val_y_tensor = torch.tensor(val_y, dtype=torch.float32).to(DEVICE)


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


dataset = MyDataset(train_x_long_tensor, train_x_mid_tensor, train_x_short_tensor, train_y_tensor)

data_loader = DataLoader(dataset, batch_size=1024, shuffle=True)


# %%
class RUL_model(nn.Module):
    def __init__(self, in_features, time_length, first_head, first_layer):
        super().__init__()
        self.in_features = in_features
        self.time_length = time_length
        self.first_head = first_head
        self.first_layer = first_layer

        # -------------------------------------------------------------------------------

        self.linear_1_first = nn.Linear(30, 100)

        self.conv_1_1 = nn.Conv2d(1, 10, kernel_size=(1, 10), padding="same")
        self.conv_1_2 = nn.Conv2d(10, 10, kernel_size=(1, 10), padding="same")
        self.conv_1_3 = nn.Conv2d(10, 20, kernel_size=(1, 10), padding="same")
        self.conv_1_4 = nn.Conv2d(20, 10, kernel_size=(1, 5), padding="same")
        self.conv_1_5 = nn.Conv2d(10, 1, kernel_size=(1, 5), padding="same")
        self.conv_1_6 = nn.Conv2d(
            1, 1, kernel_size=(self.in_features, self.in_features), padding="same"
        )

        # -------------------------------------------------------------------------------

        self.linear_2_first = nn.Linear(20, 100)

        self.conv_2_1 = nn.Conv2d(1, 10, kernel_size=(1, 10), padding="same")
        self.conv_2_2 = nn.Conv2d(10, 10, kernel_size=(1, 10), padding="same")
        self.conv_2_3 = nn.Conv2d(10, 20, kernel_size=(1, 10), padding="same")
        self.conv_2_4 = nn.Conv2d(20, 10, kernel_size=(1, 5), padding="same")
        self.conv_2_5 = nn.Conv2d(10, 1, kernel_size=(1, 5), padding="same")
        self.conv_2_6 = nn.Conv2d(
            1, 1, kernel_size=(self.in_features, self.in_features), padding="same"
        )

        # -------------------------------------------------------------------------------

        self.linear_3_first = nn.Linear(10, 100)

        self.conv_3_1 = nn.Conv2d(1, 10, kernel_size=(1, 10), padding="same")
        self.conv_3_2 = nn.Conv2d(10, 10, kernel_size=(1, 10), padding="same")
        self.conv_3_3 = nn.Conv2d(10, 20, kernel_size=(1, 10), padding="same")
        self.conv_3_4 = nn.Conv2d(20, 10, kernel_size=(1, 5), padding="same")
        self.conv_3_5 = nn.Conv2d(10, 1, kernel_size=(1, 5), padding="same")
        self.conv_3_6 = nn.Conv2d(
            1, 1, kernel_size=(self.in_features, self.in_features), padding="same"
        )

        # -------------------------------------------------------------------------------

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=100, nhead=self.first_head, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer, num_layers=self.first_layer
        )

        self.linear1 = nn.Linear(600, 50)
        self.dropout = nn.Dropout(0.2)
        self.linear2 = nn.Linear(50, 1)

    def forward(self, x_1, x_2, x_3):
        # ------------------------------------------------------------------------------
        x_1 = self.linear_1_first(x_1)
        x_1_temp = x_1
        x_1 = self.conv_1_1(x_1)
        x_1 = F.leaky_relu(x_1)
        x_1 = self.conv_1_2(x_1)
        x_1 = F.leaky_relu(x_1)
        x_1 = self.conv_1_3(x_1)
        x_1 = F.leaky_relu(x_1)
        x_1 = self.conv_1_4(x_1)
        x_1 = F.leaky_relu(x_1)
        x_1 = self.conv_1_5(x_1)
        x_1 = F.leaky_relu(x_1)
        x_1 = x_1 + x_1_temp  # Skip connection
        x_1 = self.conv_1_6(x_1)
        x_1 = F.leaky_relu(x_1)
        x_1 = x_1.squeeze(1)
        # --------------------------------------------------------------------------------
        x_2 = self.linear_2_first(x_2)
        x_2_temp = x_2
        x_2 = self.conv_2_1(x_2)
        x_2 = F.leaky_relu(x_2)
        x_2 = self.conv_2_2(x_2)
        x_2 = F.leaky_relu(x_2)
        x_2 = self.conv_2_3(x_2)
        x_2 = F.leaky_relu(x_2)
        x_2 = self.conv_2_4(x_2)
        x_2 = F.leaky_relu(x_2)
        x_2 = self.conv_2_5(x_2)
        x_2 = F.leaky_relu(x_2)
        x_2 = x_2 + x_2_temp  # Skip connection
        x_2 = self.conv_2_6(x_2)
        x_2 = F.leaky_relu(x_2)
        x_2 = x_2.squeeze(1)
        # ---------------------------------------------------------------------------------

        x_3 = self.linear_3_first(x_3)
        x_3_temp = x_3
        x_3 = self.conv_3_1(x_3)
        x_3 = F.leaky_relu(x_3)
        x_3 = self.conv_3_2(x_3)
        x_3 = F.leaky_relu(x_3)
        x_3 = self.conv_3_3(x_3)
        x_3 = F.leaky_relu(x_3)
        x_3 = self.conv_3_4(x_3)
        x_3 = F.leaky_relu(x_3)
        x_3 = self.conv_3_5(x_3)
        x_3 = F.leaky_relu(x_3)
        x_3 = x_3 + x_3_temp  # Skip connection
        x_3 = self.conv_3_6(x_3)
        x_3 = F.leaky_relu(x_3)
        x_3 = x_3.squeeze(1)

        # --------------------------------------------------------------------------------
        x = x_1 + x_2 + x_3
        # x = torch.cat((x_1,x_2,x_3), dim=2)

        # Multi Head Attention
        x = self.encoder(x)
        x = x.reshape(-1, 600)
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
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

    model = RUL_model(6, 30, first_head, first_layer).to(DEVICE)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    SOH_loss_best = 1000

    for epoch in range(0, 50):
        idx = 0
        for batch_X1, batch_X2, batch_X3, batch_y1 in data_loader:
            model.train()
            batch_X1 = batch_X1.to(DEVICE)
            batch_X2 = batch_X2.to(DEVICE)
            batch_X3 = batch_X3.to(DEVICE)
            batch_y1 = batch_y1.to(DEVICE)
            SOH_pred = model(batch_X1, batch_X2, batch_X3)
            loss = torch.sqrt(loss_fn(SOH_pred, batch_y1))
            # loss = mape_loss(SOH_pred, batch_y1)
            # loss =loss_fn(SOH_pred, batch_y1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                model.eval()
                SOH_pred_val = model(val_x_long_tensor, val_x_mid_tensor, val_x_short_tensor)
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
study.optimize(objective, n_trials=25)

# %%
fig = optuna.visualization.plot_param_importances(study)
fig.show()

# %%
model = RUL_model(6, 30, 4, 4).to(DEVICE)
model = model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()
# loss_fn = torch.nn.L1Loss()

SOH_loss_best = 1000


for epoch in range(0, 300):
    idx = 0
    for batch_X1, batch_X2, batch_X3, batch_y1 in data_loader:
        model.train()
        batch_X1 = batch_X1.to(DEVICE)
        batch_X2 = batch_X2.to(DEVICE)
        batch_X3 = batch_X3.to(DEVICE)
        batch_y1 = batch_y1.to(DEVICE)
        SOH_pred = model(batch_X1, batch_X2, batch_X3)
        loss = torch.sqrt(loss_fn(SOH_pred, batch_y1))
        # loss = mape_loss(SOH_pred, batch_y1)
        # loss =loss_fn(SOH_pred, batch_y1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            SOH_pred_val = model(val_x_long_tensor, val_x_mid_tensor, val_x_short_tensor)
            SOH_pred_val = SOH_pred_val.cpu().detach().numpy()
            SOH_val_loss = mean_squared_error(val_y, SOH_pred_val) ** 0.5
            # SOH_val_loss = mape_loss(val_y_tensor, SOH_pred_val).cpu().detach().numpy()
            SOH_val_loss = float(SOH_val_loss)

        if ((epoch % 20) == 0) & (idx == 0):
            print("Epoch : ", epoch, "Train ongoing")
            print(
                "epoch : ",
                epoch,
                "/idx : ",
                idx,
                "/train_loss :",
                round(float(loss), 4),
                "/SOH_val_loss : ",
                round(SOH_val_loss, 4),
            )

        if SOH_val_loss < SOH_loss_best:
            SOH_loss_best = SOH_val_loss
            torch.save(
                model.state_dict(), artifact_path("RUL_prediction_multiple_cycle_input_plus")
            )
            print(
                "epoch : ",
                epoch,
                "/idx : ",
                idx,
                "/train_loss :",
                round(float(loss), 4),
                "/SOH_val_loss : ",
                round(SOH_val_loss, 4),
            )

        idx += 1

# %%
rmse_list = []
mape_list = []

mode = "SOH"

fig = plt.figure(figsize=(15, 15))

for k in range(len(test_bat_key) - 6):
    # for k in range(10,len(test_bat_key)):
    battery_list = [test_bat_key[k]]

    model_new = RUL_model(6, 30, 4, 4).to(DEVICE)

    model_new.load_state_dict(torch.load(artifact_path("RUL_prediction_multiple_cycle_input_plus")))

    test_df = df_loader(battery_list)

    test_x_long = []
    test_x_mid = []
    test_x_short = []
    test_y = []

    for i in range(len(test_df) - 29):
        test_x_long.append(np.array(test_df.iloc[i : i + 30, 4:], dtype=np.float32).T)
        test_x_mid.append(np.array(test_df.iloc[i + 10 : i + 30, 4:], dtype=np.float32).T)
        test_x_short.append(np.array(test_df.iloc[i + 20 : i + 30, 4:], dtype=np.float32).T)
        test_y.append(np.array(test_df.iloc[i + 29, 2], dtype=np.float32))

    test_x_long = np.array(test_x_long)
    test_x_mid = np.array(test_x_mid)
    test_x_short = np.array(test_x_short)
    test_y = np.array(test_y) * 1500

    test_x_long = np.reshape(
        test_x_long, (test_x_long.shape[0], 1, test_x_long.shape[1], test_x_long.shape[2])
    )
    test_x_mid = np.reshape(
        test_x_mid, (test_x_mid.shape[0], 1, test_x_mid.shape[1], test_x_mid.shape[2])
    )
    test_x_short = np.reshape(
        test_x_short, (test_x_short.shape[0], 1, test_x_short.shape[1], test_x_short.shape[2])
    )

    test_x_long_tensor = torch.tensor(test_x_long, dtype=torch.float32).to(DEVICE)
    test_x_mid_tensor = torch.tensor(test_x_mid, dtype=torch.float32).to(DEVICE)
    test_x_short_tensor = torch.tensor(test_x_short, dtype=torch.float32).to(DEVICE)
    test_y_tensor = torch.tensor(test_y, dtype=torch.float32).to(DEVICE)

    # print(test_x.shape)
    with torch.no_grad():
        model_new.eval()
        SOH_test_pred = model_new(test_x_long_tensor, test_x_mid_tensor, test_x_short_tensor)
        # SOH_test_pred = model_new(test_x_long_tensor,test_x_mid_tensor,test_x_short_tensor)
        SOH_test_pred = SOH_test_pred.cpu().detach().numpy() * 1500

    # if battery_key == 'b2c14':
    # SOH_test_pred = np.delete(SOH_test_pred, [246])
    # Cycle_test = np.delete(Cycle_test, [246])
    # SOH_test = np.delete(SOH_test, [246])

    plot_num = 331 + k
    # plot_num = 331 + (k-10)

    if mode == "SOH":
        soh_rmse = mean_squared_error(test_y, SOH_test_pred) ** 0.5
        soh_mape = MAPE(np.array(test_y), np.array(SOH_test_pred))
        EOL_mape = MAPE(np.array(test_y[0]), np.array(SOH_test_pred[0]))

        rmse_list.append(soh_rmse)
        mape_list.append(soh_mape)
        data_length = len(np.array(test_y))

        plt.subplot(plot_num)
        plt.plot(SOH_test_pred, color="r", label="Predicted RUL")
        plt.plot(test_y, color="b", label="real RUL")
        plt.text(1, data_length * 0.18, "EOL MAPE(%) : " + str(round(EOL_mape, 2)), fontsize=16)
        plt.text(1, data_length * 0.10, "RMSE : " + str(round(soh_rmse, 2)), fontsize=16)
        plt.text(1, data_length * 0.02, "MAPE(%) : " + str(round(soh_mape, 2)), fontsize=16)

        plt.title(battery_list[0], fontsize=15)
        plt.ylabel("RUL", fontsize=15)
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
