import numpy as np
import pandas as pd
import time
from data_preprocessing import DataPreprocessing
from mlp_model import MLP
import torch

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


# parameters
input_len = 24
# input_len_vec = [4, 8, 12, 16, 20, 24, 32, 48, 72, 128]
prediction_depth = 24
sequal_prediction = False
validation_split = .1
batch_size = 1024
# batch_size_vec = (2 ** i for i in [8, 9, 10, 11, 12, 13, 14, 15, 16])
lr = 0.05 if sequal_prediction else 0.003
epochs = 200
# lr_vec = [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003]

# load and preprocess data
data = DataPreprocessing(training_mode=True, input_len=input_len, prediction_depth=prediction_depth)
total_dataset_len = data.get_total_dataset_len()
train_len = int(total_dataset_len * (1 - validation_split))

random_order = np.random.permutation(total_dataset_len)
train_idx = np.sort(random_order[:train_len])
val_idx = np.sort(random_order[train_len:]) if validation_split > 0 else None

experiments_df = pd.DataFrame()

# s_vec = [2048, 4096, 8192]

for gamma in [.3, 0.5]:
    # for h1 in s_vec:
    #     for h2 in s_vec:
    #         hidden_size = (128, h1, h2, 64)
    exp_start_time = time.time()
    model = MLP(device=data.device, train_idx=train_idx, val_idx=val_idx, batch_size=batch_size,
                sequal_prediction=sequal_prediction).to(data.device)
    model.set_optimizer(lr=lr)
    model.set_scheduler(step_size=int(np.sqrt(epochs)), gamma=gamma)

    # training loop
    train_loss_vec = []
    val_loss_vec = []
    best_val_loss = 1000.
    best_loss_idx = 0
    epoch = 0

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train_loss = model.train_single_epoch(data, epoch, log_interval=0)
        val_loss = model.predict_and_return_loss(data, model.val_idx) if validation_split > 0 else 0

        train_loss_vec.append(train_loss)
        val_loss_vec.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_loss_idx = epoch
        elif epoch - best_loss_idx > 20:
            print(f"break at epoch {epoch}/{epochs}. loss {val_loss} is higher than {best_val_loss} from 20 steps ago.")
            break
        # if epoch % 20 == 0:
        #     model.plot_example(data, idx=val_idx[22], epoch_num=epoch)
        #     model.plot_learning_curve(train_loss_vec, val_loss_vec, epoch_num=epoch)

        if epoch % 5 == 0:
            print('| end of epoch {:3d} | time: {:5.2f}s | val loss {:5.5f} | val log10-loss {:4.3f}'.format(epoch, (
                time.time() - epoch_start_time), val_loss, np.log10(val_loss)))

        model.scheduler.step()

    experiments_df = pd.concat([experiments_df, pd.DataFrame([[lr, best_val_loss, best_loss_idx, val_loss_vec[-1],
                            epoch, time.time() - exp_start_time]],
                            columns=["lr", "best_val_loss", "best_idx", "last_val_loss", "last_idx", "train_time"])],
                            ignore_index=True)
    experiments_df.to_csv("mlp_results.csv")
