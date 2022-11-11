import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 17, 'figure.figsize': [15, 7]})


class MLP(nn.Module):
    """
    This class is used to train the Multi-Layer Perceptron (MLP) model.
    Args:
        input_len: history length of the input data.
        prediction_depth: future depth of the model.
        hidden_size: tuple of hidden layer sizes.
        device: device to use for training.
        train_idx: indices of the training data.
        val_idx: indices of the validation data.
        batch_size: batch size for training.
        sequal_prediction: get the prediction depth by consecutive predictions of dept 1
        
    """

    def __init__(self, input_len: int = 24, prediction_depth: int = 24, hidden_size: tuple = (128, 16384, 64),
                 device='cpu', train_idx: np.ndarray = None, val_idx: np.ndarray = None, batch_size: int = None,
                 sequal_prediction=False):
        super(MLP, self).__init__()

        self.input_len = input_len
        self.prediction_depth = prediction_depth
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.batch_size = batch_size
        self.sequal_prediction = sequal_prediction

        self.device = device
        self.linear_in = nn.Linear(input_len * 15, hidden_size[0])
        self.BN0 = nn.BatchNorm1d(hidden_size[0])
        self.linear_h0 = nn.Linear(hidden_size[0], hidden_size[1])
        self.BN1 = nn.BatchNorm1d(hidden_size[1])
        self.linear_h1 = nn.Linear(hidden_size[1], hidden_size[2])
        self.BN2 = nn.BatchNorm1d(hidden_size[2])
        self.linear_out = nn.Linear(hidden_size[2], 1) if sequal_prediction else \
            nn.Linear(hidden_size[2], prediction_depth)
        self.init_weights()

        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.scheduler = None

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                return m.bias.data.zero_()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                return nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, inp_data):
        # inp_data: [Batch, Input length, Channel]
        seq_last = inp_data[:, -1:].detach()
        output = inp_data - seq_last
        output = nn.Flatten()(inp_data)
        output = nn.ReLU()(self.linear_in(output))
        output = self.BN0(output)
        output = nn.ReLU()(self.linear_h0(output))
        output = self.BN1(output)
        output = nn.ReLU()(self.linear_h1(output))
        output = self.BN2(output)
        output = self.linear_out(output)
        output = output + seq_last[:, :, 0]
        return output

    def set_optimizer(self, lr: float = 0.001):
        # self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    def set_scheduler(self, step_size: int = 100, gamma: float = 0.8):
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor,
        #                                                            patience=patience, verbose=True)

    def train_single_epoch(self, data, epoch_num: int = 0, log_interval: int = None) -> float:
        self.train()
        total_loss = 0.
        log_interval_loss = 0.
        start_time = time.time()

        rand_idx = self.train_idx[np.random.permutation(len(self.train_idx))]
        if log_interval is None:
            log_interval = int(np.shape(self.train_idx)[0] / self.batch_size / 5)
        print('-' * 89) if log_interval > 0 else None

        for batch, i in enumerate(range(0, np.shape(self.train_idx)[0] - self.batch_size + 1, self.batch_size)):
            x, y = data.get_training_batch(rand_idx[i:i + self.batch_size])
            self.optimizer.zero_grad()
            output = self(x)
            if self.sequal_prediction:
                loss = self.criterion(output, y[:, 0:1])
            else:
                loss = self.criterion(data.sqrt_weight(output), data.sqrt_weight(y))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.7)
            self.optimizer.step()

            cur_loss = loss.item()
            total_loss += x.shape[0] * cur_loss / len(self.train_idx)
            if log_interval > 0:
                log_interval_loss += cur_loss / log_interval
                if batch % log_interval == 0 and batch > 0:
                    elapsed = time.time() - start_time
                    print(
                        '| epoch {:3d} | {:4d}/{:4d} batches | lr {:02.6f} | {:5.2f} ms | loss {:5.5f} | log10-loss {:4.3f}'
                            .format(epoch_num, batch, np.shape(self.train_idx)[0] // self.batch_size,
                                    self.scheduler.get_last_lr()[0], elapsed * 1000 / log_interval, log_interval_loss,
                                    np.log10(cur_loss)))
                    log_interval_loss = 0.
                    start_time = time.time()

        print('-' * 89) if log_interval > 0 else None
        return total_loss

    def sequal_predictor(self, data, idx: np.ndarray) -> torch.tensor:
        assert np.shape(idx)[0] <= self.batch_size, "sequal_predictor is limited to a single batch"
        self.eval()
        with torch.no_grad():
            self.optimizer.zero_grad()
            pred = torch.zeros((len(idx), self.prediction_depth), device=self.device)
            x, _ = data.get_training_batch(idx)
            xf, _ = data.get_training_batch(idx + self.prediction_depth)
            for d in range(self.prediction_depth):
                if d > 0:
                    x = torch.cat((x[:, 1:, :], xf[:, d - 1:d, :]), dim=1)
                    x[:, d, 0] = pred[:, d - 1]
                output = self(x)
                pred[:, d] = output[:, 0]

        return pred

    def predict_and_return_loss(self, data, idx: np.ndarray) -> float:
        self.eval()
        total_loss = 0.
        with torch.no_grad():
            for i in range(0, np.shape(idx)[0], self.batch_size):
                x, y = data.get_training_batch(idx[i:i + self.batch_size])
                self.optimizer.zero_grad()
                output = self(x)
                if self.sequal_prediction:
                    total_loss += x.shape[0] * self.criterion(output[:, 0:1], y[:, 0:1]).cpu().item()
                else:
                    total_loss += x.shape[0] * self.criterion(data.sqrt_weight(output),
                                                              data.sqrt_weight(y)).cpu().item()
        return total_loss / np.shape(idx)[0]

    def plot_example(self, data, idx: int, epoch_num: int, save_plots=True, show_plots=False):
        self.eval()
        x, y = data.get_training_batch(np.array([idx]))
        self.optimizer.zero_grad()
        output = self.sequal_predictor(data, np.array([idx])) if self.sequal_prediction else self(x)

        first_loss = self.criterion(output[:, 0:1], y[:, 0:1]).cpu().item()
        weighted_loss = self.criterion(data.sqrt_weight(output), data.sqrt_weight(y)).cpu().item()
        pred = output[0, :].view(-1).detach().cpu().numpy()
        truth = y[0, :].view(-1).detach().cpu().numpy()

        pred_soc = data.min_max_denorm(pred)
        diff_soc = pred_soc - data.min_max_denorm(truth)
        truth_soc = data.min_max_denorm(np.concatenate((x[0, :, 0].cpu().numpy(), truth)))

        plt.figure()
        plt.plot(np.arange(-self.input_len, self.prediction_depth), truth_soc, color="k", label="truth")
        plt.plot(pred_soc, color="b", label="pred")
        plt.plot(diff_soc, color="r", label="diff")
        plt.grid(True, which='both')
        plt.title(f"epoch={epoch_num}: \n loss_1st={first_loss}, loss_w={weighted_loss} \n"
                  + f"loss_1st / a^2={first_loss / data.NET_CONSUMPTION_NORM_A**2} = "
                  + f" {np.sqrt(first_loss) / data.NET_CONSUMPTION_NORM_A}^2, diff[0] = {diff_soc[0]}")
        plt.xlabel(f"time, w.r.t. idx {idx}")
        plt.ylabel("net consumption w/o SoC")
        plt.legend()
        if show_plots:
            plt.show()
        if save_plots:
            plt.savefig('eval_example.png')
            plt.close()

    def plot_learning_curve(self, train_loss_vec, val_loss_vec, epoch_num, save_plots=True, show_plots=False):

        plt.figure()
        plt.plot(train_loss_vec, label="train")
        plt.plot(val_loss_vec, label="val")
        plt.yscale('log')
        plt.xlabel('epoch, best={}'.format(np.argmin(val_loss_vec)))
        plt.ylabel('loss, best=10^({:.4f})'.format(np.log10(np.min(val_loss_vec))))
        plt.title(f"Learning curve, epoch={epoch_num}")
        plt.grid(True, which='both')
        plt.legend()
        if show_plots:
            plt.show()
        if save_plots:
            plt.savefig('learning_curve.png')
            plt.close()
