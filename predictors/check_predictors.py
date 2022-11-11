import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from predictors.consumption_predictors import ConsPredictors

idx_predictor = ConsPredictors(method='IDX')
dot_predictor = ConsPredictors(method='DOT')
mlp_predictor = ConsPredictors(method='MLP')

csv = pd.read_csv('../data/buildings_net_electricity_consumption_without_storage.csv')

# %% single example
building_num = 0
time_idx = 100

model_input = csv.iloc[time_idx:time_idx + 24, building_num].to_numpy()
csv_output = csv.iloc[time_idx + 24:time_idx + 24 + 24, building_num].to_numpy()
idx_output = idx_predictor.get_prediction(model_input=model_input[0], time_idx=time_idx, building_num=building_num,
                                          known_building=True)
dot_output = dot_predictor.get_prediction(model_input=model_input, time_idx=time_idx, building_num=building_num,
                                          known_building=True)
dot_wb_output = dot_predictor.get_prediction(model_input=model_input, time_idx=time_idx, building_num=building_num,
                                             known_building=False)
mlp_output = mlp_predictor.get_prediction(model_input=model_input, time_idx=time_idx, building_num=building_num,
                                          known_building=True)
mlp_wb_output = mlp_predictor.get_prediction(model_input=model_input, time_idx=time_idx, building_num=building_num,
                                             known_building=False)

plt.figure()
plt.plot(np.arange(-24, 24), np.concatenate((model_input, csv_output)), label='CSV')
plt.plot(idx_output, '-+', label='IDX')
plt.plot(dot_output, '-x', label='DOT')
plt.plot(dot_wb_output, '-o', label='DOT unknown building')
plt.plot(mlp_output, '-1', label='MLP')
plt.plot(mlp_wb_output, '-2', label='MLP unknown building')
plt.xlabel(f"time, w.r.t. idx {time_idx}")
plt.ylabel("net consumption w/o SoC")
plt.legend()
plt.savefig('cons_mlp/predictors_example.png')
plt.close()
# plt.show()

# %% statistics

n = 50  # number of samples

idx_diff = np.zeros(n)
dot_diff = np.zeros(n)
dot_wb_diff = np.zeros(n)
mlp_diff = np.zeros(n)
mlp_wb_diff = np.zeros(n)
buildings_vec = np.random.randint(0, 5, n)
time_vec = np.random.randint(0, 8760 - 24, n)

weights_for_loss = np.exp(- np.arange(0, 24) / 24 * 2) / sum(np.exp(- np.arange(0, 24) / 24 * 2))

for sample in range(n):
    building_num = int(buildings_vec[sample])
    time_idx = int(time_vec[sample])
    model_input = csv.iloc[time_idx:time_idx + 24, building_num].to_numpy()
    csv_output = csv.iloc[time_idx + 24:time_idx + 24 + 24, building_num].to_numpy()
    idx_output = idx_predictor.get_prediction(model_input=model_input[0], time_idx=time_idx, building_num=building_num,
                                          known_building=True)
    idx_diff[sample] = np.sum((csv_output - idx_output) ** 2 * weights_for_loss)

    dot_output = dot_predictor.get_prediction(model_input=model_input, time_idx=time_idx, building_num=building_num,
                                          known_building=True)
    dot_diff[sample] = np.sum((csv_output - dot_output) ** 2 * weights_for_loss)

    dot_wb_output = dot_predictor.get_prediction(model_input=model_input, time_idx=time_idx, building_num=building_num,
                                          known_building=False)
    dot_wb_diff[sample] = np.sum((csv_output - dot_wb_output) ** 2 * weights_for_loss)

    mlp_output = mlp_predictor.get_prediction(model_input=model_input, time_idx=time_idx, building_num=building_num,
                                          known_building=True)
    mlp_wb_diff[sample] = np.sum((csv_output - mlp_output) ** 2 * weights_for_loss)

    mlp_wb_output = mlp_predictor.get_prediction(model_input=model_input, time_idx=time_idx, building_num=building_num,
                                          known_building=False)
    mlp_wb_diff[sample] = np.sum((csv_output - mlp_wb_output) ** 2 * weights_for_loss)

    print('{}/{} samples done.'.format(sample + 1, n))

plt.figure()
plt.plot(idx_diff, '-+', label='IDX')
plt.plot(dot_diff, '-x', label='DOT')
plt.plot(dot_wb_diff, '-o', label='DOT unknown building')
plt.plot(mlp_diff, '-1', label='MLP')
plt.plot(mlp_wb_diff, '-2', label='MLP unknown building')
plt.ylabel("weighted MSE")
plt.legend()
plt.savefig('cons_mlp/predictors_stats.png')
plt.close()
