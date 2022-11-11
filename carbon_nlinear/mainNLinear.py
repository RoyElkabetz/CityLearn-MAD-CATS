import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, time, math
import torch
import torch.nn as nn
# import torch.nn.functional as f

# Torch initialization
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_batch(source, idxs, samples_num, prediction_len):
    x = source[idxs[0]:idxs[0] + samples_num]
    y = source[idxs[0] + samples_num:idxs[0] + prediction_len + samples_num]
    for s in range(len(idxs) - 1):
        x = torch.column_stack((x, source[idxs[s + 1]:idxs[s + 1] + samples_num]))
        y = torch.column_stack((y, source[idxs[s + 1] + samples_num:idxs[s + 1] + samples_num + prediction_len]))

    return x.T, y.T


class Model(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, seq_len=128, pred_len=24):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:].detach()
        x = x - seq_last
        # x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.Linear(x)
        x = x + seq_last
        return x  # [Batch, Output length, Channel]


def train(train_xy, model, criterion, optimizer, scheduler, epoch, batch_size, sample_len, prediction_len):
    model.train()
    train_loss = 0.
    start_time = time.time()

    idx_order = np.random.permutation(np.shape(train_xy)[0] - prediction_len - sample_len)
    batch_num = len(idx_order) // batch_size
    idx_order = np.reshape(idx_order[:batch_num * batch_size], (batch_size, -1))

    for batch in range(batch_num):
        x, y = get_batch(train_xy, idx_order[:, batch], sample_len, prediction_len)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        train_loss += loss.cpu().item() * x.size(0)
        log_interval = int(batch_num / 3)

        # if batch % log_interval == 0 and batch > 0:
        #     cur_loss = train_loss / log_interval
        #     elapsed = time.time() - start_time
        #     print(
        #         '| epoch {:3d} | {:5d}/{:5d} batches | lr {:.2e} | {:5.2f} ms | loss {:5.5f} | loss log {:8.2f}'.format(
        #             epoch, batch, np.shape(train_xy)[0] // batch_size, scheduler.get_last_lr()[0],
        #                           elapsed * 1000 / log_interval, cur_loss, math.log(cur_loss)))
        #     total_loss = 0
        #     start_time = time.time()

    train_loss /= np.size(idx_order)
    return train_loss


def evaluate(data_source, eval_model, criterion, batch_size, sample_len):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.

    idx_order = np.arange(np.shape(data_source)[0] - prediction_len - sample_len)
    batch_num = len(idx_order) // batch_size
    idx_order = np.reshape(idx_order[:batch_num * batch_size], (batch_size, -1), order='F')

    prediction_mat = np.zeros((np.size(idx_order), prediction_len))

    with torch.no_grad():
        for b in range(batch_num):
            x, y = get_batch(data_source, idx_order[:, b], sample_len, prediction_len)
            output = eval_model(x)
            prediction_mat[idx_order[:, b], :] = output.cpu().numpy()
            total_loss += x.size(0) * criterion(output, y).cpu().item()
    return total_loss / np.size(idx_order), prediction_mat
# %%
def save_checkpoint(self):
    print('... saving checkpoint ...')
    torch.save(self.state_dict(), self.ckpt_path)


def load_checkpoint(ckpt_path, mdl, gpu_to_cpu=False):
    print('... loading checkpoint ...')
    if gpu_to_cpu:
        mdl.load_state_dict(torch.load(ckpt_path, map_location=lambda storage, loc: storage))
    else:
        mdl.load_state_dict(torch.load(ckpt_path))

# %%
# Hyper-parameters
sample_len = 128
prediction_len = 24
batch_size = 24*365 - prediction_len#  1024*8

lr = .15 #  5e-3
lr_step_size = 150
lr_gamma = 0.9
epochs = 3000
# val_split = 0 WE USE NO VALIDATION

# Load and preprocess data
carbon_data = pd.read_csv(filepath_or_buffer="../data/citylearn_challenge_2022_phase_1/carbon_intensity.csv")
carbon_data = np.squeeze(carbon_data.values)
carbon_data = (carbon_data - .9 / 5) * 9
carbon_data = np.concatenate((np.zeros(sample_len), carbon_data))

# Define the model
model = Model(seq_len=sample_len, pred_len=prediction_len).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)

# Make folders
if not os.path.exists('models'):
    os.makedirs('models')
if not os.path.exists('plots'):
    os.makedirs('plots')

# Training loop
train_losses = []
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_data = torch.FloatTensor(carbon_data).to(device)
    cur_train_loss = train(train_data, model, criterion, optimizer, scheduler, epoch, batch_size, sample_len,
                           prediction_len)
    train_losses.append(cur_train_loss)
    # torch.save(model.state_dict(), 'models' + '/epoch_' + str(epoch).zfill(3) + '.pt')

    # print('-' * 89)
    if epoch % 10 == 0:
        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.5f} = exp({:3.2f})'.format(epoch, (
                time.time() - epoch_start_time), cur_train_loss, math.log(cur_train_loss)))
    # print('-' * 89)

    scheduler.step()

# Aftermaths - choose best epoch
# max_idx = 0
# for i in range(epochs):
#     if train_losses[i] < train_losses[max_idx]:
#         max_idx = i
# checkpoint = torch.load('models' + '/epoch_' + str(max_idx).zfill(3) + '.pt')
# load_checkpoint(ckpt_path='models/epoch_2000.pt', mdl=model, gpu_to_cpu=True)
# model.load_state_dict(checkpoint)
# print('Train loss is {:2.4f}'.format(train_losses))

# Store results
with open('models' + '/losses.npy', 'wb') as f:
    np.save(f, train_losses)
    # np.save(f, max_idx)


# %% present results
xscale = np.arange(1, len(train_losses)+1)

fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

plt.loglog(xscale, train_losses, 'b-')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Loss plot')
plt.grid(True, which='both')
fig.savefig('plots/learning_curve.png')
plt.close()


# %% evaluate prediction

train_data = torch.FloatTensor(carbon_data).to(device)
val_loss, prediction_mat = evaluate(train_data, model, criterion, batch_size, sample_len)

for hr in [1, 3, 6, 12, 24]:
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.plot(carbon_data[sample_len:sample_len+np.shape(prediction_mat)[0]-hr], 'b-', label='raw')
    plt.plot(prediction_mat[:-hr, hr-1], 'r-', label=str(hr) + 'h pred')
    plt.plot(carbon_data[sample_len:sample_len+np.shape(prediction_mat)[0]-hr] - prediction_mat[:-hr, hr-1], 'g-', label='diff')
    plt.xlabel('time')
    plt.ylabel('carbon intensity')
    plt.title(str(hr) + 'h prediction')
    plt.grid(True, which='both')
    plt.legend()
    fig.savefig('plots/' + str(hr) + 'h.png')
    plt.close()
# %%

fig = plt.figure(figsize=(10, 6))
fig.tight_layout()

for hr in [24, 12, 6, 3, 1]:
    plt.plot(carbon_data[sample_len:sample_len + np.shape(prediction_mat)[0] - hr] - prediction_mat[:-hr, hr - 1],
             label=str(hr))
plt.xlabel('time')
plt.ylabel('carbon intensity')
plt.title('differences')
plt.legend()
plt.grid(True, which='both')
fig.savefig('plots/differences.png')
plt.close()
