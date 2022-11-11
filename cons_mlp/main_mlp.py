import numpy as np
import time
from data_preprocessing import DataPreprocessing
from mlp_model import MLP
import torch

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


# parameters
input_len = 24
prediction_depth = 24
sequal_prediction = False
validation_split = .1
batch_size = 1024
lr = 0.05 if sequal_prediction else 0.003
epochs = 200
cease_after = 200

save_plots = True
show_plots = True
plots_interval = 200

# load and preprocess data
data = DataPreprocessing(training_mode=True, input_len=input_len, prediction_depth=prediction_depth)
total_dataset_len = data.get_total_dataset_len()
train_len = int(total_dataset_len * (1 - validation_split))

random_order = np.random.permutation(total_dataset_len)
train_idx = np.sort(random_order[:train_len])
val_idx = np.sort(random_order[train_len:]) if validation_split > 0 else None


model = MLP(device=data.device, train_idx=train_idx, val_idx=val_idx, batch_size=batch_size,
            sequal_prediction=sequal_prediction).to(data.device)
model.set_optimizer(lr=lr)
model.set_scheduler(step_size=int(np.sqrt(epochs)), gamma=0.5)

# training loop
train_loss_vec = []
val_loss_vec = []
best_val_loss = 1000.
best_loss_idx = 0
type_str = 'sequal' if sequal_prediction else 'deep'

for epoch in range(1, min(epochs, cease_after) + 1):
    epoch_start_time = time.time()
    train_loss = model.train_single_epoch(data, epoch)
    if validation_split > 0:
        val_loss = model.predict_and_return_loss(data, model.val_idx) if validation_split > 0 else 0

        train_loss_vec.append(train_loss)
        val_loss_vec.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_loss_idx = epoch
            torch.save(model.state_dict(), f'best_epoch_{type_str}.pt')
        elif epoch - best_loss_idx > 30:
            print(f"break at epoch {epoch}/{epochs}. loss {val_loss} is higher than {best_val_loss} from 10 steps ago.")
            break

        if (epoch % plots_interval == 0) and (save_plots or show_plots):
            model.plot_example(data, idx=val_idx[22], epoch_num=epoch, save_plots=save_plots, show_plots=show_plots)
            model.plot_learning_curve(train_loss_vec, val_loss_vec, epoch_num=epoch,
                                      save_plots=save_plots, show_plots=show_plots)

        print('| end of epoch {:3d} | time: {:5.2f}s | val loss {:5.5f} | val log10-loss {:4.3f}'.format(epoch, (
                time.time() - epoch_start_time), val_loss, np.log10(val_loss)))
    else:
        print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.5f} | train log10-loss {:4.3f}'.format(epoch, (
                time.time() - epoch_start_time), train_loss, np.log10(train_loss)))
        if epoch == cease_after:
            torch.save(model.state_dict(), f'best_epoch_{type_str}.pt')
            break

    model.scheduler.step()

model.plot_example(data, idx=val_idx[22], epoch_num=epoch, save_plots=save_plots, show_plots=show_plots)
model.plot_learning_curve(train_loss_vec, val_loss_vec, epoch_num=epoch,
                          save_plots=save_plots, show_plots=show_plots)
