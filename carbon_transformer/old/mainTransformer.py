# General imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os, time, math
# Project imports
from carbon_transformer.transformerModel import *
from carbon_transformer.evaluationsNplots import *
# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as f




# Torch initialization
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Hyper-parameterr
batch_size = 64
sample_len = 128
prediction_len = 24

lr = 5e-7
lr_step_size = 30
lr_gamma = 0.8
epochs = 1000
evaluate_after_epoch = 10
# val_split = 0 WE USE NO VALIDATION

# Load and preprocess data
carbon_data = pd.read_csv(filepath_or_buffer="../data/citylearn_challenge_2022_phase_1/carbon_intensity.csv")
carbon_data = np.squeeze(carbon_data.values)
carbon_data = (carbon_data - .9/5)*9
carbon_data = np.concatenate((np.zeros(sample_len), carbon_data))

# Define the model
model = TransformerModel(samp_len=sample_len).to(device)
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
    cur_train_loss = train(train_data, model, criterion, optimizer, scheduler, epoch, batch_size, sample_len, prediction_len)
    train_losses.append(cur_train_loss)
    torch.save(model.state_dict(), 'models' + '/epoch_' + str(epoch).zfill(3) + '.pt')

    if epoch % evaluate_after_epoch == 0:
        plot_examples(train_data, model, criterion, epoch, sample_len)

    cur_train_loss = evaluate(train_data, model, criterion, batch_size, sample_len)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.5f} | train ppl {:8.2f}'.format(epoch, (
            time.time() - epoch_start_time), cur_train_loss, math.exp(cur_train_loss)))
    print('-' * 89)

    scheduler.step()

# Aftermaths - choose best epoch
max_idx = 0
for i in range(epochs):
    if train_losses[i] < train_losses[max_idx]:
        max_idx = i
checkpoint = torch.load('models' + '/epoch_' + str(max_idx).zfill(3) + '.pt')
model.load_state_dict(checkpoint)
print('Train loss is {:2.4f}'.format(train_losses))

# Store results
plot_learning_curves(train_losses, val_losses, max_idx)
with open('models' + '/losses.npy', 'wb') as f:
    np.save(f, train_losses)
    np.save(f, max_idx)
