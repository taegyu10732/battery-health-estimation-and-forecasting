# %% [markdown]
# # Consolidated research script
#
# Method group **G25**: Dense-to-sequence model. Architecture: DenseNet CNN + GRU decoder. Method tags: raw encoder|GRU decoding.
#
# This copy preserves the research implementation while removing saved outputs, execution counters, pinned GPU selection, and scratch-only cells. Closely related variants are summarized in the research script index.

# %%
from battery_soh.data import resolve_data_dir
from battery_soh.preprocessing import interpolate_timeseries
from battery_soh.raw_data import load_battery_dictionary
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
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
input_data = []
target_data = []

max_length = 1100

for battery_key in train_bat_key:
    Cycle_length = len(bat_dict[battery_key]["summary"]["QD"]) - 1
    input_data.append(bat_dict[battery_key]["summary"]["QD"][1:101])
    temp_target = bat_dict[battery_key]["summary"]["QD"][101:]
    target_bool = temp_target >= 0.88
    temp_target = temp_target[target_bool]
    zeros1 = np.ones(max_length) * 0.88
    zeros1[: len(temp_target)] = temp_target
    zeros1[len(temp_target)] = 2

    target_data.append(zeros1)

# %%
input_data = np.array(input_data)
target_data = np.array(target_data)
print(input_data.shape)
print(target_data.shape)

# %%
input_data = np.reshape(input_data, (input_data.shape[0], input_data.shape[1], 1))
target_data = np.reshape(target_data, (target_data.shape[0], target_data.shape[1], 1))
print(input_data.shape)
print(target_data.shape)

# %%
whole_data_tensor = torch.tensor(whole_data, dtype=torch.float32).to(DEVICE)
RUL_true_tensor = torch.tensor(RUL_true, dtype=torch.float32).to(DEVICE)


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
    torch.tensor(target_data, dtype=torch.float32),
)

data_loader = DataLoader(dataset, batch_size=10, shuffle=True)


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

        self.features.add_module(
            "dense_block_{}".format(len(nblocks) - 1),
            self._make_dense_block(nblocks[len(nblocks) - 1], inner_channels),
        )
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module("bn", nn.BatchNorm2d(inner_channels))
        self.features.add_module("relu", nn.ReLU())

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear_1 = nn.Linear(inner_channels, 16)
        # self.linear_2 = nn.Linear(inner_channels, 16)
        self.linear_3 = nn.Linear(inner_channels, 50)
        self.linear_f = nn.Linear(50, 1)

        # weight initialization
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.features(x)
        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)

        hidden_1 = self.linear_1(x)
        hidden_1 = hidden_1.view(1, hidden_1.shape[0], hidden_1.shape[1])
        # hidden_2 = self.linear_2(x)
        # hidden_2 = hidden_2.view(1, hidden_2.shape[0], hidden_2.shape[1])
        x = self.linear_3(x)

        x = F.leaky_relu(x)
        x = self.linear_f(x)
        return x.squeeze(1), hidden_1

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
class lstm_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(-1), encoder_hidden_states)
        output = self.linear(lstm_out)

        return output, self.hidden


# %%
class lstm_encoder_decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(lstm_encoder_decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = DenseNet([6, 12, 24, 6])
        self.decoder = lstm_decoder(input_size=input_size, hidden_size=hidden_size)

    def forward(self, inputs, targets, target_len, teacher_forcing_ratio):
        batch_size = inputs.shape[0]

        outputs = torch.zeros(batch_size, 1100, 1)

        EOL, hidden = self.encoder(inputs)
        decoder_input = targets[:, 0, :]

        # 원하는 길이가 될 때까지 decoder를 실행한다.
        for t in range(target_len):
            out, hidden = self.decoder(decoder_input, hidden)
            out = out.squeeze(1)

            # teacher forcing을 구현한다.
            # teacher forcing에 해당하면 다음 인풋값으로는 예측한 값이 아니라 실제 값을 사용한다.
            if random.random() < teacher_forcing_ratio:
                decoder_input = targets[:, t, :]
            else:
                decoder_input = out
            outputs[:, t, :] = out

        return outputs, EOL

    # 편의성을 위해 예측해주는 함수도 생성한다.
    def predict(self, inputs, target_len):
        self.eval()
        inputs = inputs.unsqueeze(0)
        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]
        outputs = torch.zeros(batch_size, target_len, input_size)
        _, hidden = self.encoder(inputs)
        decoder_input = inputs[:, -1, :]
        for t in range(target_len):
            out, hidden = self.decoder(decoder_input, hidden)
            out = out.squeeze(1)
            decoder_input = out
            outputs[:, t, :] = out
        return outputs.detach().numpy()[0, :, 0]


# %%
learning_rate = 0.1
epoch = 500
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=3)
criterion = nn.MSELoss()

# %%
model = lstm_encoder_decoder(input_size=1, hidden_size=16).to(DEVICE)

# %%
model.train()
with tqdm(range(epoch)) as tr:
    for i in tr:
        total_loss = 0.0
        for x, y1, y2 in data_loader:
            optimizer.zero_grad()
            x = x.to(DEVICE).float()
            y1 = y1.to(DEVICE).float()
            y2 = y2.to(DEVICE).float()
            output, EOL = model(x, y2, 1100, 0.9)
            output = output.to(DEVICE)
            EOL = EOL.to(DEVICE)
            loss = torch.sqrt(criterion(output, y2))
            # loss = torch.sqrt(criterion(EOL, y1))
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().item()
            print(loss)
        tr.set_postfix(loss="{0:.5f}".format(total_loss / len(data_loader)))


# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(count_parameters(model))  # 사용
