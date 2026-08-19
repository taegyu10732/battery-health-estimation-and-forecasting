# %% [markdown]
# # Consolidated research script
#
# Method group **G24**: Masked teacher-forcing sequence model. Architecture: LSTM encoder-decoder. Method tags: masking|teacher forcing.
#
# This copy preserves the research implementation while removing saved outputs, execution counters, pinned GPU selection, and scratch-only cells. Closely related variants are summarized in the research script index. During cleanup, 0 syntactically invalid scratch cell(s) and 2 display-only scratch cell(s) were omitted.

# %%
from battery_soh.data import resolve_data_dir
from battery_soh.paths import artifact_path
from battery_soh.raw_data import load_battery_dictionary
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
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
random.shuffle(selected_keys)

# %%
train_bat_key = selected_keys[:90]
val_bat_key = selected_keys[90:100]
test_bat_key = selected_keys[100:]

# %%
len(train_bat_key)

# %%
input_data = []
target_data = []

max_length = 900

for battery_key in train_bat_key:
    input_data.append((bat_dict[battery_key]["summary"]["QD"][1:301] / 1.1) * 100)
    temp_target = bat_dict[battery_key]["summary"]["QD"][301:]
    target_bool = temp_target >= 0.88
    temp_target = temp_target[target_bool]
    temp_target = (temp_target / 1.1) * 100
    zeros1 = np.zeros(max_length)
    zeros1[: len(temp_target)] = temp_target
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
class MyDataset(Dataset):
    def __init__(self, X1, y1):
        self.X1 = X1
        self.y1 = y1

    def __len__(self):
        return len(self.X1)

    def __getitem__(self, idx):
        return self.X1[idx], self.y1[idx]


dataset = MyDataset(
    torch.tensor(input_data, dtype=torch.float32), torch.tensor(target_data, dtype=torch.float32)
)
data_loader = DataLoader(dataset, batch_size=10, shuffle=True)


# %%
class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )

    def forward(self, x_input):
        lstm_out, self.hidden = self.lstm(x_input)
        return lstm_out, self.hidden


# %%
class lstm_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=4):
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
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

        self.encoder = lstm_encoder(input_size=input_size, hidden_size=hidden_size)
        self.decoder = lstm_decoder(input_size=input_size, hidden_size=hidden_size)

    def forward(self, inputs, targets, target_len, teacher_forcing_ratio):
        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]

        outputs = torch.zeros(batch_size, target_len, input_size)

        _, hidden = self.encoder(inputs)
        decoder_input = inputs[:, -1, :]

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

        return outputs

    def val_predict(self, inputs, target_len):
        self.eval()
        # inputs = inputs.unsqueeze(0)
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
            # print(outputs.detach().numpy().shape)
        return outputs

    # 편의성을 위해 예측해주는 함수도 생성한다.
    def predict(self, inputs, target_len):
        self.eval()

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
model = lstm_encoder_decoder(input_size=1, hidden_size=16).to(DEVICE)

# %%
learning_rate = 0.001
epoch = 500
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# %%
loss_best = 100

model.train()
with tqdm(range(epoch)) as tr:
    for i in tr:
        total_loss = 0.0
        for x, y in data_loader:
            model.train()
            optimizer.zero_grad()
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).float()
            mask = torch.where(y > 0, 1.0, 0.0).to(DEVICE)
            output = model(x, y, 900, 0.0).to(DEVICE)
            output = torch.mul(mask, output)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().item()

            model.eval()
            output_val = model.val_predict(input_val, target_len=900).to(DEVICE)
            val_mask = torch.where(target_val > 0, 1.0, 0.0).to(DEVICE)
            output_val = torch.mul(output_val, val_mask)
            loss_val = mean_squared_error(
                output_val.squeeze(2).cpu().detach().numpy(), target_val.squeeze(2).cpu()
            )
            # print(mean_absolute_error(output_val.squeeze(2).cpu().detach().numpy(), target_val.squeeze(2).cpu()))

            if loss_val < loss_best:
                loss_best = loss_val
                print(loss_val)
                torch.save(model.state_dict(), artifact_path("seq2seq_masked"))

        tr.set_postfix(loss="{0:.5f}".format(total_loss / len(data_loader)))

# %%
input_val = []
target_val = []

max_length = 900

for battery_key in train_bat_key:
    input_val.append((bat_dict[battery_key]["summary"]["QD"][1:301] / 1.1) * 100)
    temp_target = bat_dict[battery_key]["summary"]["QD"][301:]
    target_bool = temp_target >= 0.88
    temp_target = temp_target[target_bool]
    temp_target = (temp_target / 1.1) * 100
    zeros1 = np.zeros(max_length)
    zeros1[: len(temp_target)] = temp_target
    target_val.append(zeros1)

# %%
input_val = np.array(input_val)
target_val = np.array(target_val)
input_val = np.reshape(input_val, (input_val.shape[0], input_val.shape[1], 1))
target_val = np.reshape(target_val, (target_val.shape[0], target_val.shape[1], 1))

# %%
input_val = torch.tensor(input_val, dtype=torch.float32).to(DEVICE)
target_val = torch.tensor(target_val, dtype=torch.float32).to(DEVICE)

# %%
print(input_val.shape)
print(target_val.shape)

# %%
model_new = lstm_encoder_decoder(input_size=1, hidden_size=16).to(DEVICE)
model_new.load_state_dict(torch.load(artifact_path("seq2seq_masked")))

# %%
print(1)

# %%
predict = model_new.predict(input_val[2:3, :, :], target_len=900)

# %%
plt.plot(predict, label="predict")
plt.plot(target_val[2].cpu(), label="true")
plt.legend()

# %%
mean_absolute_error(predict, target_val[10].cpu())

# %%
plt.plot(input_val[0].cpu())
plt.plot(input_val[1].cpu())


# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(count_parameters(model))  # 사용
