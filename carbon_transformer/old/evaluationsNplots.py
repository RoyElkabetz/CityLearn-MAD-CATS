# General imports
import numpy as np
import matplotlib.pyplot as plt
import os, time, math
# Project imports
from dataPreprocessing import *
from transformerModel import *
# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as f

def get_batch(source, idxs, samples_num, prediction_len):
    x = source[idxs[0]:idxs[0] + samples_num]
    y = source[idxs[0] + prediction_len:idxs[0] + prediction_len + samples_num]
    for s in range(len(idxs)-1):
        x = torch.column_stack((x, source[idxs[s+1]:idxs[s+1] + samples_num]))
        y = torch.column_stack((y, source[idxs[s+1] + prediction_len:idxs[s+1] + samples_num + prediction_len]))

    return x.T, y.T

def train(train_xy, model, criterion, optimizer, scheduler, epoch, batch_size, sample_len, prediction_len):
    model.train()
    total_loss = 0.
    train_loss = 0.
    start_time = time.time()

    idx_order = np.random.permutation(np.shape(train_xy)[0])
    batch_num = np.shape(train_xy)[0]//batch_size
    idx_order = np.reshape(idx_order[:batch_num*batch_size], (batch_size, -1))

    for batch in range(batch_num):
        x, y = get_batch(train_xy, idx_order[:, batch], sample_len, prediction_len)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        total_loss += loss.cpu().item()
        train_loss += loss.cpu().item() * x.size(0)
        log_interval = int(np.shape(train_xy)[0] / batch_size / 5)

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:.2e} | {:5.2f} ms | loss {:5.5f} | ppl {:8.2f}'.format(
                epoch, batch, np.shape(train_xy)[0] // batch_size, scheduler.get_last_lr()[0],
                              elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

    train_loss /= np.shape(train_xy)[0]
    return train_loss


def plot_examples(data_source, eval_model, criterion, epoch_num, sample_len):
    eval_model.eval()
    total_loss = 0.

    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    with torch.no_grad():
        for k, j in enumerate(np.random.randint(0, np.shape(data_source)[0] - 1, size=4)):
            data, target = get_batch(data_source, j, 1, [0, 1, 2])
            output = eval_model(data)
            output = f.normalize(output, dim=1)*np.sqrt(sample_len)
            curr_loss = criterion(output, target).item()
            test_result = output[0].view(-1).cpu()
            truth = target[0].view(-1).cpu()

            fig.add_subplot(2, 2, k + 1)
            plt.plot(test_result, color="red", label="pred")
            plt.plot(truth, color="blue", label="gt")
            # plt.plot(test_result - truth, color="green", label="diff")
            plt.grid(True, which='both')
            plt.axhline(y=0, color='k')
            plt.title('sample ' + str(j) + ', MSE ' + '{:2.4f}'.format(curr_loss))
            if k < 2:
                plt.tick_params(labelbottom=False)
        plt.legend()
        fig.savefig('plots' + '/epoch%d_example.png' % epoch_num)
        plt.close()

    return total_loss / np.shape(data_source)[0]


    eval_model.eval()
    loss = np.empty(np.shape(data_source)[0])

    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    with torch.no_grad():
        for s in range(np.shape(data_source)[0]):
            data, targets = get_batch(data_source, s, 1, [0, 1, 2])
            output = eval_model(data)
            output = f.normalize(output, dim=1)*np.sqrt(sample_len)
            loss[s] = criterion(output, targets).cpu().item()
            best_samples = np.argsort(loss)

        for k in range(4):
            data, target = get_batch(data_source, best_samples[k], 1, [0, 1, 2])
            output = eval_model(data)
            output = f.normalize(output, dim=1)*np.sqrt(sample_len)
            curr_loss = criterion(output, target).item()
            test_result = output[0].view(-1).cpu()
            truth = target[0].view(-1).cpu()

            fig.add_subplot(2, 2, k + 1)
            plt.plot(test_result, color="red", label="pred")
            plt.plot(truth, color="blue", label="gt")
            # plt.plot(test_result - truth, color="green", label="diff")
            plt.grid(True, which='both')
            plt.axhline(y=0, color='k')
            plt.title('sample ' + str(best_samples[k]) + ', MSE ' + '{:2.4f}'.format(curr_loss))
            if k < 2:
                plt.tick_params(labelbottom=False)
        plt.legend()
        fig.savefig('plots_' + pe_type + '/epoch%d_best_examples.png' % epoch_num)
        plt.close()


def plot_learning_curves(train_loss_vec, val_loss_vec, vline, pe_type):
    xscale = np.arange(1, np.shape(train_loss_vec)[0]+1)

    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()

    plt.semilogx(xscale, train_loss_vec, 'b-x', label='Training')
    plt.semilogx(xscale, val_loss_vec, 'r-x', label='Validation')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss plot')
    plt.legend()
    plt.grid(True, which='both')
    plt.axvline(x=vline, color='k')
    fig.savefig('plots_' + pe_type + '/learning_curve.png')
    plt.close()


def evaluate(data_source, eval_model, criterion, batch_size, sample_len):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for e in range(0, np.shape(data_source)[0] - 1, batch_size):
            data, targets = get_batch(data_source, e, batch_size, [0, 1, 2])
            output = eval_model(data)
            output = f.normalize(output, dim=1)*np.sqrt(sample_len)
            total_loss += data.size(0) * criterion(output, targets).cpu().item()
    return total_loss / np.shape(data_source)[0]
