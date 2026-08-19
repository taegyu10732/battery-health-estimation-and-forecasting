# %% [markdown]
# # Consolidated research script
#
# Method group **G23**: LSTM sequence-to-sequence. Architecture: LSTM encoder-decoder. Method tags: beta weighting|extra information.
#
# This copy preserves the research implementation while removing saved outputs, execution counters, pinned GPU selection, and scratch-only cells. Closely related variants are summarized in the research script index. During cleanup, 0 syntactically invalid scratch cell(s) and 3 display-only scratch cell(s) were omitted.

# %%
from battery_soh.data import resolve_data_dir
from battery_soh.evaluation import MAPE
from battery_soh.paths import artifact_path
from battery_soh.raw_data import load_battery_dictionary
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
selected_keys.remove("b2c28")
selected_keys.remove("b2c20")
selected_keys.remove("b1c7")
selected_keys.remove("b2c3")
selected_keys.remove("b2c16")
selected_keys.remove("b2c43")
selected_keys.remove("b2c27")
selected_keys.remove("b2c12")
selected_keys.remove("b2c22")
selected_keys.remove("b2c26")
selected_keys.remove("b2c45")
selected_keys.remove("b2c13")
selected_keys.remove("b2c38")
selected_keys.remove("b2c29")
selected_keys.remove("b2c15")
selected_keys.remove("b2c9")
selected_keys.remove("b2c17")
selected_keys.remove("b2c1")
selected_keys.remove("b3c46")
selected_keys.remove("b2c33")
selected_keys.remove("b1c17")
selected_keys.remove("b2c41")
selected_keys.remove("b2c25")

# %%
len(selected_keys)

# %%
random.shuffle(selected_keys)

# %%
train_bat_key = selected_keys[:70]
val_bat_key = selected_keys[70:80]
test_bat_key = selected_keys[80:]

# %%
len(train_bat_key)

# %%
input_data = []
input_extra_data = []
target_data = []


for battery_key in train_bat_key:
    c10 = bat_dict[battery_key]["cycles"]["10"]
    c100 = bat_dict[battery_key]["cycles"]["100"]
    dQ_100_10 = c100["Qdlin"] - c10["Qdlin"]

    # minimum_dQ_100_10 = np.log(np.abs(np.min(dQ_100_10)))
    variance_dQ_100_10 = np.log(np.var(dQ_100_10))

    SOH_array = bat_dict[battery_key]["summary"]["QD"]
    SOH_bool = SOH_array >= 0.85
    SOH_array = SOH_array[SOH_bool]
    Cycle_length = len(SOH_array)
    for i in range(0, Cycle_length - 199):
        input_data.append(SOH_array[i : i + 100])
        target_data.append(SOH_array[i + 100 : i + 200])
        input_extra_data.append([variance_dQ_100_10])

# %%
input_data = np.array(input_data)
target_data = np.array(target_data)
input_extra_data = np.array(input_extra_data)
print(input_data.shape)
print(input_extra_data.shape)
print(target_data.shape)

# %%
input_data = np.reshape(input_data, (input_data.shape[0], input_data.shape[1], 1))
target_data = np.reshape(target_data, (target_data.shape[0], target_data.shape[1], 1))
input_extra_data = np.reshape(input_extra_data, (len(input_extra_data), 1, 1))
print(input_data.shape)
print(input_extra_data.shape)
print(target_data.shape)

# %%
input_val = []
input_extra_val = []
target_val = []


for battery_key in val_bat_key:
    c10 = bat_dict[battery_key]["cycles"]["10"]
    c100 = bat_dict[battery_key]["cycles"]["100"]
    dQ_100_10 = c100["Qdlin"] - c10["Qdlin"]

    minimum_dQ_100_10 = np.log(np.abs(np.min(dQ_100_10)))
    variance_dQ_100_10 = np.log(np.var(dQ_100_10))

    SOH_array = bat_dict[battery_key]["summary"]["QD"]
    SOH_bool = SOH_array >= 0.85
    SOH_array = SOH_array[SOH_bool]
    Cycle_length = len(SOH_array)
    for i in range(0, Cycle_length - 199):
        input_val.append(SOH_array[i : i + 100])
        target_val.append(SOH_array[i + 100 : i + 200])
        input_extra_val.append([variance_dQ_100_10])

# %%
input_val = np.array(input_val)
target_val = np.array(target_val)
input_extra_val = np.array(input_extra_val)
input_val = np.reshape(input_val, (input_val.shape[0], input_val.shape[1], 1))
target_val = np.reshape(target_val, (target_val.shape[0], target_val.shape[1], 1))
input_extra_val = np.reshape(input_extra_val, (len(input_extra_val), 1, 1))

# %%
input_val = torch.tensor(input_val, dtype=torch.float32).to(DEVICE)
input_extra_val = torch.tensor(input_extra_val, dtype=torch.float32).to(DEVICE)
target_val = torch.tensor(target_val, dtype=torch.float32).to(DEVICE)


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
    torch.tensor(input_data, dtype=torch.float32),
    torch.tensor(input_extra_data, dtype=torch.float32),
    torch.tensor(target_data, dtype=torch.float32),
)
data_loader = DataLoader(dataset, batch_size=2048, shuffle=True)


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

        self.linear_t1 = nn.Linear(1, hidden_size)
        self.linear_t2 = nn.Linear(hidden_size, hidden_size)

        self.linear_f = nn.Linear(hidden_size, 1)

    def forward(self, x_input, encoder_hidden_states, temperature_input):
        lstm_out, self.hidden = self.lstm(x_input.unsqueeze(-1), encoder_hidden_states)

        temperature_input = self.linear_t1(temperature_input)
        temperature_input = F.leaky_relu(temperature_input)
        temperature_input = self.linear_t2(temperature_input)

        output = torch.mul(lstm_out, temperature_input)

        output = self.linear_f(output)

        return output, self.hidden


# %%
class lstm_encoder_decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(lstm_encoder_decoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(input_size=input_size, hidden_size=hidden_size)
        self.decoder = lstm_decoder(input_size=input_size, hidden_size=hidden_size)

    def forward(self, inputs, temperature_input, targets, target_len, teacher_forcing_ratio):
        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]

        outputs = torch.zeros(batch_size, target_len, input_size)

        _, hidden = self.encoder(inputs)
        decoder_input = inputs[:, -1, :]

        # 원하는 길이가 될 때까지 decoder를 실행한다.
        for t in range(target_len):
            out, hidden = self.decoder(decoder_input, hidden, temperature_input)
            out = out.squeeze(1)

            # teacher forcing을 구현한다.
            # teacher forcing에 해당하면 다음 인풋값으로는 예측한 값이 아니라 실제 값을 사용한다.
            if random.random() < teacher_forcing_ratio:
                decoder_input = targets[:, t, :]
            else:
                decoder_input = out
            outputs[:, t, :] = out

        return outputs

    # 편의성을 위해 예측해주는 함수도 생성한다.
    def val_predict(self, inputs, temperature_input, target_len):
        self.eval()
        # inputs = inputs.unsqueeze(0)
        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]
        outputs = torch.zeros(batch_size, target_len, input_size)
        # print(outputs.shape)
        _, hidden = self.encoder(inputs)
        decoder_input = inputs[:, -1, :]
        for t in range(target_len):
            out, hidden = self.decoder(decoder_input, hidden, temperature_input)

            out = out.squeeze(1)

            decoder_input = out
            outputs[:, t, :] = out
            # print(outputs.detach().numpy().shape)
        return outputs.detach().numpy()

    def predict(self, inputs, target_len, temperature_input):
        self.eval()
        inputs = inputs.unsqueeze(0)
        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]
        outputs = torch.zeros(batch_size, target_len, input_size)
        _, hidden = self.encoder(inputs)
        decoder_input = inputs[:, -1, :]
        for t in range(target_len):
            out, hidden = self.decoder(decoder_input, hidden, temperature_input)
            out = out.squeeze(1)
            decoder_input = out
            outputs[:, t, :] = out
        return outputs.detach().numpy()[0, :, 0]


# %%
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = lstm_encoder_decoder(input_size=1, hidden_size=32).to(DEVICE)

# %%
learning_rate = 0.001
epoch = 500
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# %%
input_test = []
input_test_extra = []

for battery_key in val_bat_key:
    SOH_array = bat_dict[battery_key]["summary"]["QD"]
    SOH_bool = SOH_array >= 0.85
    SOH_array = SOH_array[SOH_bool]
    input_test.append(SOH_array)

    c10 = bat_dict[battery_key]["cycles"]["10"]
    c100 = bat_dict[battery_key]["cycles"]["100"]
    dQ_100_10 = c100["Qdlin"] - c10["Qdlin"]

    minimum_dQ_100_10 = np.log(np.abs(np.min(dQ_100_10)))
    variance_dQ_100_10 = np.log(np.var(dQ_100_10))
    # variance_dQ_100_10 = [[variance_dQ_100_10]]
    input_test_extra.append([variance_dQ_100_10])

input_test_extra = np.array(input_test_extra)
input_test_extra = np.reshape(input_test_extra, (len(input_test_extra), 1, 1))
input_test_extra = torch.tensor(input_test_extra, dtype=torch.float32).to(DEVICE)

# %%
loss_best = 100

model.train()
with tqdm(range(epoch)) as tr:
    for i in tr:
        total_loss = 0.0
        for x1, x2, y in data_loader:
            model.train()
            optimizer.zero_grad()
            x1 = x1.to(DEVICE).float()
            x2 = x2.to(DEVICE).float()
            y = y.to(DEVICE).float()
            output = model(x1, x2, y, 100, 0.3).to(DEVICE)
            loss = torch.sqrt(criterion(output, y))
            loss.backward()
            optimizer.step()
            total_loss += loss.cpu().item()

        model.eval()
        output_val = model.val_predict(input_val, target_len=100, temperature_input=input_extra_val)
        loss_val = mean_squared_error(output_val.squeeze(2), target_val.squeeze(2).cpu()) ** 0.5

        # output_val = model.val_predict(input_val, input_extra_val, target_len=100)
        # loss_val = mean_squared_error(output_val.squeeze(2), target_val.squeeze(2).cpu())**0.5

        if loss_val < loss_best:
            loss_best = loss_val
            print(loss_val)
            torch.save(
                model.state_dict(), artifact_path("seq2seq_100_100_extra_information_nature")
            )

        tr.set_postfix(loss="{0:.5f}".format(total_loss / len(data_loader)))


# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(count_parameters(model))  # 사용

# %%
input_val = []
input_extra_val = []
target_val = []


for battery_key in val_bat_key:
    c10 = bat_dict[battery_key]["cycles"]["10"]
    c100 = bat_dict[battery_key]["cycles"]["100"]
    dQ_100_10 = c100["Qdlin"] - c10["Qdlin"]

    minimum_dQ_100_10 = np.log(np.abs(np.min(dQ_100_10)))
    variance_dQ_100_10 = np.log(np.var(dQ_100_10))

    SOH_array = bat_dict[battery_key]["summary"]["QD"]
    SOH_bool = SOH_array >= 0.85
    SOH_array = SOH_array[SOH_bool]
    Cycle_length = len(SOH_array)
    for i in range(0, Cycle_length - 199):
        input_val.append(SOH_array[i : i + 100])
        target_val.append(SOH_array[i + 100 : i + 200])
        input_extra_val.append([variance_dQ_100_10])

# %%
input_val = np.array(input_val)
target_val = np.array(target_val)
input_extra_val = np.array(input_extra_val)
input_val = np.reshape(input_val, (input_val.shape[0], input_val.shape[1], 1))
target_val = np.reshape(target_val, (target_val.shape[0], target_val.shape[1], 1))
input_extra_val = np.reshape(input_extra_val, (len(input_extra_val), 1, 1))

# %%
input_val = torch.tensor(input_val, dtype=torch.float32).to(DEVICE)
input_extra_val = torch.tensor(input_extra_val, dtype=torch.float32).to(DEVICE)
target_val = torch.tensor(target_val, dtype=torch.float32).to(DEVICE)

# %%
print(target_val.shape)

# %%
model_new = lstm_encoder_decoder(input_size=1, hidden_size=32).to(DEVICE)
model_new.load_state_dict(torch.load(artifact_path("seq2seq_100_100_extra_information_nature")))

# %%
input_range = np.linspace(1, 100, 100, dtype=int)
output_range = np.linspace(101, 200, 100, dtype=int)

# %%
test_number = 200

# %%
test_number = 2500

predict = model_new.predict(
    input_val[test_number], target_len=100, temperature_input=input_extra_val[test_num]
)

plt.plot(input_range, input_val[test_number].cpu(), label="input", color="black")
plt.plot(output_range, predict, label="predict", color="r")
plt.plot(output_range, target_val[test_number].cpu(), label="correct", color="blue")
# plt.yticks(np.linspace(0.88, 1.1, 11))
print(MAPE(predict, np.array(target_val[test_number].cpu())))
plt.legend()

# %%
input_test = []
input_test_extra = []

for battery_key in test_bat_key:
    SOH_array = bat_dict[battery_key]["summary"]["QD"]
    SOH_bool = SOH_array >= 0.88
    SOH_array = SOH_array[SOH_bool]
    input_test.append(SOH_array)

    c10 = bat_dict[battery_key]["cycles"]["10"]
    c100 = bat_dict[battery_key]["cycles"]["100"]
    dQ_100_10 = c100["Qdlin"] - c10["Qdlin"]

    minimum_dQ_100_10 = np.log(np.abs(np.min(dQ_100_10)))
    variance_dQ_100_10 = np.log(np.var(dQ_100_10))
    # variance_dQ_100_10 = [[variance_dQ_100_10]]
    input_test_extra.append([variance_dQ_100_10])

# %%
input_test_extra = np.array(input_test_extra)
input_test_extra = np.reshape(input_test_extra, (len(input_test_extra), 1, 1))
input_test_extra = torch.tensor(input_test_extra, dtype=torch.float32).to(DEVICE)

# %%
test_num = 1
print(test_bat_key[test_num])
print(len(input_test[test_num]))

predicted_list = [input_test[test_num][:100]]

temp_input = input_test[test_num][:100]
temp_input = np.reshape(temp_input, (100, 1))
temp_input = torch.tensor(temp_input, dtype=torch.float32).to(DEVICE)

for i in range(0, 15):
    predict_temp = model_new.predict(
        temp_input, target_len=100, temperature_input=input_test_extra[test_num]
    )
    predicted_list.append(predict_temp)
    temp_input = predict_temp
    temp_input = np.reshape(temp_input, (100, 1))
    temp_input = torch.tensor(temp_input, dtype=torch.float32).to(DEVICE)

predicted_list = np.array(predicted_list)
predicted_result = predicted_list.ravel()

# %%
input_sequence = np.linspace(1, 100, 100, dtype=int)
predict_sequence = np.linspace(
    101, len(input_test[test_num]), len(input_test[test_num]) - 100, dtype=int
)

# print(predict)

plt.plot(input_sequence, input_test[test_num][:100], color="black", label="input")
plt.plot(predict_sequence, input_test[test_num][100:], color="blue", label="true")
plt.plot(
    predict_sequence,
    predicted_result[100 : len(input_test[test_num])],
    color="red",
    label="predicted",
)

plt.ylabel("Capacity(Ah)", fontsize=15)
plt.xlabel("Cycle", fontsize=15)
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)

plt.legend(fontsize=15)
plt.show()

print(MAPE(predicted_result[100 : len(input_test[test_num])], input_test[test_num][100:]))
print(
    mean_squared_error(
        predicted_result[100 : len(input_test[test_num])], input_test[test_num][100:]
    )
    ** 0.5
)

# %%
test_num = 4
print(test_bat_key[test_num])
print(len(input_test[test_num]))

predicted_list = [input_test[test_num][50:150]]

temp_input = input_test[test_num][50:150]
temp_input = np.reshape(temp_input, (100, 1))
temp_input = torch.tensor(temp_input, dtype=torch.float32).to(DEVICE)

for i in range(0, 15):
    predict_temp = model_new.predict(
        temp_input, target_len=100, temperature_input=input_test_extra[test_num]
    )
    predicted_list.append(predict_temp)
    temp_input = predict_temp
    temp_input = np.reshape(temp_input, (100, 1))
    temp_input = torch.tensor(temp_input, dtype=torch.float32).to(DEVICE)

predicted_list = np.array(predicted_list)
predicted_result = predicted_list.ravel()

# %%
input_sequence = np.linspace(51, 150, 100, dtype=int)
predict_sequence = np.linspace(
    151, len(input_test[test_num]), len(input_test[test_num]) - 150, dtype=int
)

# print(predict)

plt.plot(input_sequence, input_test[test_num][50:150], color="black", label="input")
plt.plot(predict_sequence, input_test[test_num][150:], color="blue", label="true")
plt.plot(
    predict_sequence,
    predicted_result[100 : len(input_test[test_num]) - 50],
    color="red",
    label="predicted",
)

plt.ylabel("Capacity(Ah)", fontsize=15)
plt.xlabel("Cycle", fontsize=15)
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)

plt.legend(fontsize=15)
plt.show()

print(MAPE(predicted_result[100 : len(input_test[test_num]) - 50], input_test[test_num][150:]))
print(
    mean_squared_error(
        predicted_result[100 : len(input_test[test_num]) - 50], input_test[test_num][150:]
    )
    ** 0.5
)
