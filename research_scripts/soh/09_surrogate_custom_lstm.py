# %% [markdown]
# # Consolidated research script
#
# Method group **G11**: Surrogate degradation modeling. Architecture: LSTM + custom recurrent cell. Method tags: surrogate|multi-cycle.
#
# This copy preserves the research implementation while removing saved outputs, execution counters, pinned GPU selection, and scratch-only cells. Closely related variants are summarized in the research script index. During cleanup, 0 syntactically invalid scratch cell(s) and 1 display-only scratch cell(s) were omitted.

# %%
from battery_soh.data import resolve_data_dir
from battery_soh.evaluation import MAPE
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
import pandas as pd
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# %%
# Shared, portable raw-data loading. This may require substantial memory.
RESEARCH_BATCHES = ("b1", "b2", "b3")
bat_dict = load_battery_dictionary(batches=RESEARCH_BATCHES)


# %%
def mape_loss(preds, target):
    epsilon = 1e-8  # Small value to avoid division by zero
    return torch.mean(torch.abs((preds - target) / (target + epsilon))) * 100


# %%
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
batch_keys = [*bat_dict.keys()]
len(batch_keys)

# %%
batch_keys = [*bat_dict.keys()]
selected_keys = []

for battery_key in batch_keys:
    if (len(bat_dict[battery_key]["summary"]["cycle"]) > 300) & (
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
selected_keys.remove("b2c3")
selected_keys.remove("b2c16")
selected_keys.remove("b2c43")
selected_keys.remove("b2c27")
selected_keys.remove("b2c12")
selected_keys.remove("b2c22")
selected_keys.remove("b2c13")
selected_keys.remove("b2c15")
selected_keys.remove("b2c9")
selected_keys.remove("b2c17")
selected_keys.remove("b2c1")
selected_keys.remove("b3c46")
selected_keys.remove("b1c17")

# %%
len(selected_keys)

# %%
battery_key = "b1c29"

fig, ax = plt.subplots()
cmap = plt.cm.get_cmap("viridis")
normalize = mcolors.Normalize(vmin=1, vmax=100)

temp = []

for num in range(10, 101):
    ax.plot(
        bat_dict[battery_key]["cycles"][str(num)]["dQdV"][:-20],
        color=cmap(normalize(num)),
        linewidth=1,
    )
    min_value = ((-1) * np.min(bat_dict[battery_key]["cycles"][str(num)]["dQdV"][:-10])) - 6
    temp.append(min_value)

# plt.ylim((-6.5, -5))
# plt.xlim((170, 350))

# %%
plt.plot(temp)

# %%
random.seed(12)

random.shuffle(selected_keys)

# %%
train_bat_key = selected_keys[:60]
val_bat_key = selected_keys[60:80]
test_bat_key = selected_keys[80:]


# %%
def train_data_maker(window_size, inter_list):

    cycle = []
    dQdV = []
    cap = []

    for battery_key in inter_list:
        temp = []
        for num in range(10, 101, 10):
            min_value = (
                (-1) * np.min(bat_dict[battery_key]["cycles"][str(num)]["dQdV"][:-20])
            ) / 10
            temp.append(min_value)

        for i in range(len(bat_dict[battery_key]["summary"]["QD"]) - window_size + 1):
            cycle.append(bat_dict[battery_key]["summary"]["cycle"][i : i + window_size] / 1500)
            cap.append(bat_dict[battery_key]["summary"]["QD"][i : i + window_size])
            dQdV.append(temp)

    cycle = np.array(cycle)
    dQdV = np.array(dQdV)
    cap = np.array(cap)

    cycle = np.reshape(cycle, (cycle.shape[0], window_size, 1))

    return cycle, dQdV, cap


# %%
def test_data_maker(inter_list):

    cycle = []
    cap = []

    for battery_key in inter_list:
        for i in range(len(bat_dict[battery_key]["summary"]["QD"])):
            cycle.append(bat_dict[battery_key]["summary"]["cycle"][i] / 1500)
            cap.append(bat_dict[battery_key]["summary"]["QD"][i])

    cycle = np.array(cycle)
    cap = np.array(cap)

    return cycle, cap


# %%
train_x, train_x_T, train_y = train_data_maker(30, train_bat_key)

# %%
val_x, val_x_T, val_y = train_data_maker(30, val_bat_key)

# %%
val_x_tensor = torch.tensor(val_x, dtype=torch.float32).to(DEVICE)
val_x_T_tensor = torch.tensor(val_x_T, dtype=torch.float32).to(DEVICE)
val_y_tensor = torch.tensor(val_y, dtype=torch.float32).to(DEVICE)


# %%
class MyDataset(Dataset):
    def __init__(self, X1, X2, y):
        self.X1 = X1
        self.X2 = X2
        self.y = y

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        return self.X1[idx], self.X2[idx], self.y[idx]


dataset = MyDataset(
    torch.tensor(train_x, dtype=torch.float32),
    torch.tensor(train_x_T, dtype=torch.float32),
    torch.tensor(train_y, dtype=torch.float32),
)

data_loader = DataLoader(dataset, batch_size=4000, shuffle=True)


# %%
class LSTM_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 400, num_layers=1, batch_first=True)
        self.fc = nn.Linear(400, 100)

        self.trunk1 = nn.Linear(10, 100)
        self.trunk2 = nn.Linear(100, 100)
        self.trunk3 = nn.Linear(100, 100)

    def forward(self, x, T):
        x = self.lstm(x)[0]
        x = self.fc(x)

        T = self.trunk1(T)
        T = F.leaky_relu(T)
        T = self.trunk2(T)
        T = F.leaky_relu(T)
        T = self.trunk3(T).unsqueeze(1)

        Final = x * T
        Final = torch.sum(Final, -1)

        return Final


# %%
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(CustomLSTM, self).__init__()
        self.hidden_size = torch.tensor(hidden_size)
        self.batch_first = batch_first

        # 입력 게이트
        self.Wii = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Whi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bi = nn.Parameter(torch.Tensor(hidden_size))

        # 망각 게이트
        self.Wif = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Whf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bf = nn.Parameter(torch.Tensor(hidden_size))

        # 셀 게이트
        self.Wig = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Whg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bg = nn.Parameter(torch.Tensor(hidden_size))

        # 출력 게이트
        self.Wio = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.Who = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bo = nn.Parameter(torch.Tensor(hidden_size))

        self.fc = nn.Linear(hidden_size, 100)

        self.trunk1 = nn.Linear(10, 100)
        self.trunk2 = nn.Linear(100, 100)
        self.trunk3 = nn.Linear(100, 100)

        self.reset_parameters()

        self.alpha = nn.Parameter(torch.Tensor([0.001]))
        self.beta = nn.Parameter(torch.Tensor([9]))
        self.theta = nn.Parameter(torch.Tensor([1]))

    def reset_parameters(self):
        stdv = 1.0 / torch.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, T):

        physics = -torch.abs(self.alpha) * torch.exp(x * self.beta) + self.theta

        if self.batch_first:
            x = x.permute(1, 0, 2)  # Convert to seq_len, batch, input_size

        seq_len, batch_size, _ = x.size()

        # Initialize hidden and cell states
        hx = torch.zeros(batch_size, self.hidden_size, device=x.device)
        cx = torch.zeros(batch_size, self.hidden_size, device=x.device)

        outputs = []

        for t in range(seq_len):
            x_t = x[t]

            # physics = (-torch.exp(x_t*1.145)+6)/4.905

            # i = torch.sigmoid(x_t @ self.Wii + physics @ self.Wii + hx @ self.Whi + self.bi)
            # f = torch.sigmoid(x_t @ self.Wif + physics @ self.Wif + hx @ self.Whf + self.bf)
            # g = torch.tanh(x_t @ self.Wig + physics @ self.Wig + hx @ self.Whg + self.bg)
            # o = torch.sigmoid(x_t @ self.Wio + physics @ self.Wio + hx @ self.Who + self.bo)

            i = torch.sigmoid(x_t @ self.Wii + hx @ self.Whi + self.bi)
            f = torch.sigmoid(x_t @ self.Wif + hx @ self.Whf + self.bf)
            g = torch.tanh(x_t @ self.Wig + hx @ self.Whg + self.bg)
            o = torch.sigmoid(x_t @ self.Wio + hx @ self.Who + self.bo)

            cx = (f * cx) + (i * g)
            hx = o * torch.tanh(cx)

            outputs.append(hx.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)

        if self.batch_first:
            outputs = outputs.permute(1, 0, 2)  # Convert back to batch, seq_len, hidden_size

        # outputs = self.fc(outputs)
        # print(outputs.shape)
        outputs = self.fc(outputs * 0.5 + 0.5 * physics)

        T = self.trunk1(T)
        T = F.leaky_relu(T)
        T = self.trunk2(T)
        T = F.leaky_relu(T)
        T = self.trunk3(T).unsqueeze(1)

        Final = outputs * T
        Final = torch.sum(Final, -1)

        return Final


# %%
model = CustomLSTM(1, 300, batch_first=True).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters())
# loss_fn_train = torch.nn.MSELoss()
loss_fn_train = torch.nn.L1Loss()

SOH_loss_best = 10000

for epoch in range(3000):
    train_loss = 0
    for batch_x1, batch_x2, batch_y in data_loader:
        model.train()
        batch_x1 = batch_x1.to(DEVICE)
        batch_x2 = batch_x2.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        SOH_pred = model(batch_x1, batch_x2)

        loss_minus = []

        for k in range(len(SOH_pred[0]) - 1):
            minus = F.relu(SOH_pred[:, k + 1] - SOH_pred[:, k])
            minus = torch.sum(minus) / len(batch_y)
            loss_minus.append(minus)

        loss_minus = torch.tensor(loss_minus).to(DEVICE)
        loss_minus = torch.mean(loss_minus)

        # loss = torch.sqrt(loss_fn_train(SOH_pred, batch_y))
        loss_data = loss_fn_train(SOH_pred, batch_y)

        loss = loss_data + 100 * loss_minus

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
    train_loss = train_loss / len(data_loader)

    with torch.no_grad():
        model.eval()
        SOH_pred_val = model(val_x_tensor, val_x_T_tensor)
        SOH_pred_val = SOH_pred_val.cpu().detach().numpy()
        SOH_val_loss = mean_squared_error(val_y, SOH_pred_val) ** 0.5
        SOH_val_loss = float(SOH_val_loss)

    if SOH_val_loss < SOH_loss_best:
        SOH_loss_best = SOH_val_loss
        print(epoch, float(train_loss), SOH_val_loss)
        torch.save(model.state_dict(), artifact_path("SOH_prediction_surrogate_val_5"))

    if (epoch % 50) == 0:
        print(epoch, float(train_loss), float(loss_data), float(loss_minus))
        torch.save(model.state_dict(), artifact_path("SOH_prediction_surrogate_5"))

# %%
model_new = CustomLSTM(1, 300, batch_first=True).to(DEVICE)
model_new.load_state_dict(torch.load(artifact_path("SOH_prediction_surrogate_5")))

# %%
empty = []

# %%
interest = [val_bat_key[13]]

test_x, test_x_T, _ = train_data_maker(30, interest)
test_x_index, test_y = test_data_maker(interest)
test_x_tensor = torch.tensor(test_x, dtype=torch.float32).to(DEVICE)
test_x_T_tensor = torch.tensor(test_x_T, dtype=torch.float32).to(DEVICE)
test_y_tensor = torch.tensor(test_y, dtype=torch.float32).to(DEVICE)

# %%
model_new.eval()

SOH_test_pred = model_new(test_x_tensor, test_x_T_tensor).cpu().detach().numpy()

# %%
max_length = len(test_x) + 29
total = []
for i in range(len(SOH_test_pred)):
    temporal = np.zeros(max_length)
    temporal[i : i + 30] = SOH_test_pred[i]
    total.append(temporal)

total = np.array(total)

# %%
mean_array = []

for k in range(total.shape[1]):
    non_zeros = total[:, k][total[:, k] != 0.0]
    mean_array.append(np.mean(non_zeros))

# %%
plt.plot(
    test_x_index.flatten() * 1500, test_y.flatten(), color="green", label="Test true", linewidth=2
)
plt.plot(test_x_index.flatten() * 1500, mean_array, color="red", label="Test pred", linewidth=2)
# plt.plot(test_x_index.flatten()*1500, test_y.flatten()[:fine_length],color='black', label ='Train', linewidth=2)
plt.legend(fontsize=15)
plt.ylabel("Capacity(Ah)", fontsize=15, fontweight="bold")
plt.xlabel("Cycle", fontsize=15, fontweight="bold")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Customize grid properties
print(mean_squared_error(test_y.flatten(), mean_array) ** 0.5)
print(MAPE(test_y.flatten(), mean_array))

# %%
train_x_fine, train_x_T_fine, train_y_fine = train_data_maker(30, interest)
train_x_fine, train_x_T_fine, train_y_fine = (
    train_x_fine[:fine_length],
    train_x_T_fine[:fine_length],
    train_y_fine[:fine_length],
)

dataset = MyDataset(
    torch.tensor(train_x_fine, dtype=torch.float32),
    torch.tensor(train_x_T_fine, dtype=torch.float32),
    torch.tensor(train_y_fine, dtype=torch.float32),
)

data_loader = DataLoader(dataset, batch_size=20, shuffle=True)

# %%
fine_length = 100

for name, param in model_new.named_parameters():
    if (name == "fc.weight") | (name == "fc.bias"):
        param.requires_grad = True
    else:
        param.requires_grad = False

# %%
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_new.parameters()), lr=0.0001)
loss_fn_train = torch.nn.MSELoss()

for epoch in range(0, 100):
    train_loss = 0
    for batch_x1, batch_x2, batch_y in data_loader:
        model_new.train()
        batch_x1 = batch_x1.to(DEVICE)
        batch_x2 = batch_x2.to(DEVICE)
        batch_y = batch_y.to(DEVICE)
        SOH_pred = model_new(batch_x1, batch_x2)

        loss = torch.sqrt(loss_fn_train(SOH_pred, batch_y))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss
    train_loss = train_loss / len(data_loader)
    print(train_loss)

# %%
for name, param in model.named_parameters():
    print("레이어 이름: ", name)
    print("가중치: ", param.data)

# %%
X = np.linspace(0, 1, 1500)
Y = -0.0008 * np.exp(X * 7.814) + 0.9883

# %%
plt.plot(X * 1500, Y, color="green")
plt.ylim((0.88, 1))

plt.ylabel("Capacity(Ah)", fontsize=15, fontweight="bold")
plt.xlabel("Cycle", fontsize=15, fontweight="bold")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Customize grid properties
