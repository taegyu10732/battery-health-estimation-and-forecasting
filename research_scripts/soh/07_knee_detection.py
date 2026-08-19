# %% [markdown]
# # Consolidated research script
#
# Method group **G09**: Knee-point estimation. Architecture: CNN-Transformer encoder. Method tags: knee detection|scarce data.
#
# This copy preserves the research implementation while removing saved outputs, execution counters, pinned GPU selection, and scratch-only cells. Closely related variants are summarized in the research script index.

# %%
from battery_soh.data import resolve_data_dir
from battery_soh.evaluation import MAPE
from battery_soh.paths import artifact_path
from battery_soh.preprocessing import interpolate_timeseries
from battery_soh.raw_data import load_battery_dictionary
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
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


batch_keys.remove("b1c11")
batch_keys.remove("b1c48")
batch_keys.remove("b1c47")

batch_keys.remove("b2c1")
batch_keys.remove("b2c2")
batch_keys.remove("b2c3")
batch_keys.remove("b2c4")
batch_keys.remove("b2c6")
batch_keys.remove("b2c9")
batch_keys.remove("b2c8")
batch_keys.remove("b2c12")
batch_keys.remove("b2c7")
batch_keys.remove("b2c35")

batch_keys.remove("b3c38")
batch_keys.remove("b3c46")

# %%
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
len(selected_keys)

# %%
random.seed(12)

random.shuffle(selected_keys)
train_bat_key = selected_keys[:65]
val_bat_key = selected_keys[65:85]
test_bat_key = selected_keys[85:]


# %%
def objective_function(params):
    a0, a1, a2, x1 = params
    predicted_values = a0 + a1 * (N - x1) + a2 * (N - x1) * np.tanh((N - x1) / 0.00001)
    return np.sum((Q - predicted_values) ** 2)


# %%
time_value = 5
inter_value = 300


# %%
def data_maker(inter_key):
    whole_data = []
    Cycle_true = []
    Knee_label = []
    SOH_true = []
    Cycle_index = 0

    for battery_key in inter_key:
        Q = bat_dict[battery_key]["summary"]["QD"][1:]
        N = np.linspace(1, len(Q), len(Q))

        def objective_function(params):
            a0, a1, a2, x1 = params
            predicted_values = a0 + a1 * (N - x1) + a2 * (N - x1) * np.tanh((N - x1) / 0.00001)
            return np.sum((Q - predicted_values) ** 2)

        initial_guess = [1, -0.0001, -0.0001, 400]

        # minimize 함수를 사용하여 파라미터 추정
        result = minimize(objective_function, initial_guess)
        knee_point = round(result.x[3])

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

                        df = pd.DataFrame(
                            {"time": time_data, "Current": Current_data, "Voltage": Voltage_data}
                        )

                        df = df.drop_duplicates(["time"])

                        data = np.array(df.T)

                        Current_data = interpolate_timeseries(data[1], inter_value)
                        Voltage_data = interpolate_timeseries(data[2], inter_value)
                        Voltage_data = (Voltage_data - 3.37) / (3.62 - 3.37)

                        temporal_data.append(Current_data)
                        temporal_data.append(Voltage_data)

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

                if np.array(temporal_data).shape == (3, inter_value):
                    if bat_dict[battery_key]["summary"]["cycle"][int(cycle_num)] < knee_point:
                        Knee_label.append(0)
                    else:
                        Knee_label.append(1)

                    Cycle_true.append(bat_dict[battery_key]["summary"]["cycle"][int(cycle_num)])
                    SOH_true.append(bat_dict[battery_key]["summary"]["QD"][int(cycle_num)])
                    whole_data.append(np.array(temporal_data))

    Knee_label = np.array(Knee_label)
    whole_data = np.array(whole_data)
    Cycle_true = np.array(Cycle_true)
    SOH_true = np.array(SOH_true)
    whole_data = np.reshape(
        whole_data, (whole_data.shape[0], 1, whole_data.shape[1], whole_data.shape[2])
    )

    return whole_data, SOH_true, Knee_label, Cycle_true


# %%
train_x, train_y1, train_y2, train_cylce = data_maker(train_bat_key)
val_x, val_y1, val_y2, val_cylce = data_maker(val_bat_key)


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
    torch.tensor(train_x, dtype=torch.float32),
    torch.tensor(train_y1, dtype=torch.float32),
    torch.tensor(train_y2, dtype=torch.float32),
)

data_loader = DataLoader(dataset, batch_size=2048, shuffle=True)

# %%
val_x = torch.tensor(val_x, dtype=torch.float32).to(DEVICE)


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

        self.linear1 = nn.Linear(100 * self.in_features, 50)
        self.linear2 = nn.Linear(50, 2)

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
        SOH = x[:, 0:1].squeeze(1)
        Knee = torch.sigmoid(x[:, 1:2]).squeeze(1)

        return SOH, Knee


# %%
model = SOH_model(3, inter_value, 5, 3).to(DEVICE)
model = model.train()

optimizer = torch.optim.Adam(model.parameters())
loss_fn_SOH = torch.nn.MSELoss()
loss_fn_Knee = torch.nn.BCELoss()

SOH_loss_best = 1000

for epoch in range(0, 500):
    idx = 0
    for batch_X1, batch_y1, batch_y2 in data_loader:
        model.train()
        batch_X1 = batch_X1.to(DEVICE)
        batch_y1 = batch_y1.to(DEVICE)
        batch_y2 = batch_y2.to(DEVICE)
        SOH_pred, Knee_pred = model(batch_X1)
        loss_SOH = torch.sqrt(loss_fn_SOH(SOH_pred, batch_y1))
        loss_Knee = loss_fn_Knee(Knee_pred, batch_y2)

        loss = loss_SOH + loss_Knee

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            SOH_pred_val, Knee_pred_val = model(val_x)
            SOH_pred_val = SOH_pred_val.cpu().detach().numpy()
            SOH_val_loss = mean_squared_error(val_y1, SOH_pred_val) ** 0.5

        if SOH_val_loss < SOH_loss_best:
            SOH_loss_best = SOH_val_loss
            torch.save(model.state_dict(), artifact_path("SOH_estimation_Knee_detection_1"))
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
fig = plt.figure(figsize=(20, 12))

rmse_list = []
mape_list = []

for k in range(0, 1):
    battery_list = [test_bat_key[k + 4]]

    test_x, test_y1, test_y2, test_cycle = data_maker(battery_list)
    test_x = torch.tensor(test_x, dtype=torch.float32).to(DEVICE)

    model_new = SOH_model(3, inter_value, 5, 3).to(DEVICE)

    model_new.load_state_dict(torch.load(artifact_path("SOH_estimation_Knee_detection_1")))

    with torch.no_grad():
        model_new.eval()
        SOH_test_pred, Knee_test_pred = model_new(test_x)

    SOH_test_pred = SOH_test_pred.cpu().detach().numpy()

    soh_rmse = mean_squared_error(test_y1, SOH_test_pred) ** 0.5
    soh_mape = MAPE(np.array(test_y1), np.array(SOH_test_pred))

    rmse_list.append(soh_rmse)
    mape_list.append(soh_mape)

    plot_num = 331 + k

    plt.subplot(plot_num)
    plt.plot(SOH_test_pred, color="r", label="Estimated SOH")
    plt.plot(test_y1, color="b", label="real SOH")
    plt.text(1, 0.95, "RMSE : " + str(round(soh_rmse, 4)), fontsize=20)
    plt.text(1, 0.93, "MAPE(%) : " + str(round(soh_mape, 3)), fontsize=20)

    plt.title(battery_list[0], fontsize=15)
    plt.ylabel("Capacity (AH)", fontsize=15)
    plt.xlabel("cycle", fontsize=15)
    plt.legend()

fig.tight_layout(pad=3.0)
plt.show()

# %%
test_x, test_y1, test_y2, test_cycle = data_maker(test_bat_key)
test_x = torch.tensor(test_x, dtype=torch.float32).to(DEVICE)

model_new = SOH_model(3, inter_value, 4, 4).to(DEVICE)

model_new.load_state_dict(torch.load(artifact_path("SOH_estimation_Knee_detection")))

with torch.no_grad():
    model_new.eval()
    SOH_test_pred, Knee_test_pred = model_new(test_x)

# %%
Knee_test_pred = Knee_test_pred.cpu().detach().numpy()
SOH_test_pred = SOH_test_pred.cpu().detach().numpy()

# %%
Knee_test_pred_round = [round(x) for x in Knee_test_pred]

# %%
fig = plt.figure(figsize=(5, 3))
plt.plot(test_y2, label="true", color="black")
plt.plot(Knee_test_pred_round, label="predicted", color="red")
plt.legend(fontsize=15)
plt.xlabel("Cycle", fontsize=15, weight="bold")

# %%
accuracy = accuracy_score(test_y2, Knee_test_pred_round)
precision = precision_score(test_y2, Knee_test_pred_round)
recall = recall_score(test_y2, Knee_test_pred_round)
f1 = f1_score(test_y2, Knee_test_pred_round)
roc_auc = roc_auc_score(test_y2, Knee_test_pred_round)

print(f"정확도: {accuracy}")
print(f"정밀도: {precision}")
print(f"재현율: {recall}")
print(f"F1 점수: {f1}")
print(f"ROC AUC 점수: {roc_auc}")

# %%
print(mean_squared_error(test_y1, SOH_test_pred) ** 0.5)
print(MAPE(np.array(test_y1), np.array(SOH_test_pred)))

# %%
fig, ax1 = plt.subplots()
ax1.plot(SOH_true, color="black", label="SOH")
ax1.set_xlabel("Cycle", fontsize=15)
ax1.set_ylabel("SOH", fontsize=15)

# plt.legend()

ax2 = ax1.twinx()
ax2.plot(Knee_label, color="blue", label="knee index")
ax2.set_ylabel("Knee index", fontsize=15)
# plt.legend()

# %%
key = 28
print(selected_keys[key])
Q = bat_dict[selected_keys[key]]["summary"]["QD"][1:]
N = np.linspace(1, len(Q), len(Q))


# %%
def objective_function(params):
    a0, a1, a2, x1 = params
    predicted_values = a0 + a1 * (N - x1) + a2 * (N - x1) * np.tanh((N - x1) / 0.00001)
    return np.sum((Q - predicted_values) ** 2)


# %%
initial_guess = [1, -0.0001, -0.0001, 400]

# minimize 함수를 사용하여 파라미터 추정
result = minimize(objective_function, initial_guess)

# 결과 출력
print(f"Optimized Parameters: {result.x}")
print(round(result.x[3]))

plt.plot(N, Q, color="black", label="true", linewidth=3)
plt.plot(
    N,
    result.x[0]
    + result.x[1] * (N - result.x[3])
    + result.x[2] * (N - result.x[3]) * np.tanh((N - result.x[3]) / 0.00001),
    color="red",
    label="fitted",
    linewidth=3,
)
# plt.scatter(round(result.x[3]), Q[round(result.x[3])], color='blue', s=50, label='knee point')
plt.vlines(round(result.x[3]), 0.85, 1.13, color="blue", label="knee point", linewidth=3)
plt.ylim([0.88, 1.13])
plt.legend(fontsize=15)
plt.xlabel("Cycle", fontsize=15)
plt.ylabel("Cap", fontsize=15)

# %%
fig = plt.figure(figsize=(15, 15))

for k in range(0, 16):
    key = k + 32
    print(selected_keys[key])
    Q = bat_dict[selected_keys[key]]["summary"]["QD"][1:]
    N = np.linspace(1, len(Q), len(Q))

    initial_guess = [1, -0.0001, -0.0001, 400]

    # minimize 함수를 사용하여 파라미터 추정
    result = minimize(objective_function, initial_guess)

    # 결과 출력
    print(f"Optimized Parameters: {result.x}")
    print(round(result.x[3]))

    plot_num = 441 + k

    plt.subplot(4, 4, k + 1)
    plt.plot(N, Q, color="black", label="true", linewidth=3)
    plt.plot(
        N,
        result.x[0]
        + result.x[1] * (N - result.x[3])
        + result.x[2] * (N - result.x[3]) * np.tanh((N - result.x[3]) / 0.00001),
        color="red",
        label="fitted",
        linewidth=3,
    )
    # plt.scatter(round(result.x[3]), Q[round(result.x[3])], color='blue', s=50, label='knee point')
    plt.vlines(round(result.x[3]), 0.85, 1.13, color="blue", label="knee point", linewidth=3)
    plt.ylim([0.88, 1.13])
    plt.title(selected_keys[key])
    # plt.legend()
    plt.xlabel("Cycle")
    plt.ylabel("Cap")

fig.tight_layout()


# %%
def objective_function(params):
    a0, a1, a2, a3, x0, x2 = params
    predicted_values = (
        a0
        + a1 * (N - x0)
        + a2 * (N - x0) * np.tanh((N - x0) / 0.00001)
        + a3 * (N - x2) * np.tanh((N - x2) / 0.00001)
    )
    return np.sum((Q - predicted_values) ** 2)


# %%
initial_guess = [-1, -1, -1, 1, 400, 500]

# minimize 함수를 사용하여 파라미터 추정
result = minimize(objective_function, initial_guess)

# 결과 출력
print(f"Optimized Parameters: {result.x}")

# %%
plt.plot(N, Q, color="black", label="true", linewidth=3)
plt.plot(
    N,
    result.x[0]
    + result.x[1] * (N - result.x[4])
    + result.x[2] * (N - result.x[4]) * np.tanh((N - result.x[4]) / 0.00001)
    + result.x[3] * (N - result.x[5]) * np.tanh((N - result.x[5]) / 0.00001),
    color="red",
    label="fitted",
    linewidth=3,
)
plt.vlines(round(result.x[4]), 0.85, 1.13, color="blue", label="knee onset", linewidth=3)
print(round(result.x[4]))
plt.ylim([0.88, 1.13])
plt.legend(fontsize=15)
plt.xlabel("Cycle", fontsize=15)
plt.ylabel("Cap", fontsize=15)
plt.ylim([0.88, 1.13])
