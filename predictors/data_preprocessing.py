import numpy as np
import pandas as pd
import torch



class DataPreprocessing:
    """
    This class handles the data preprocessing for the LSTM model.
    It operates in two modes:
    1. Training mode: the data is loaded from the csv files and stored in a numpy array.
        This mode initializes stacked_vectorized_data.
    2. Inference mode: the data is loaded from the numpy array.
    """

    def __init__(self, input_len: int = 24, prediction_depth: int = 24, training_mode=False):
        self.input_len = input_len
        self.prediction_depth = prediction_depth
        self.env_global_data = self.get_env_global_data()
        self.NET_CONSUMPTION_NORM_A = 0.1671148973523351
        self.NET_CONSUMPTION_NORM_B = -0.36754414813187397
        # these two parameters scale the training net consumption to the range [-0.966..., 0.966...] (erf of 1.5)

        if training_mode:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.stacked_vectorized_data = self.get_stacked_vectorized_data()  # note that this is a torch tensor
            self.total_dataset_len = self.stacked_vectorized_data.shape[0] - self.prediction_depth - self.input_len
        else:
            self.stacked_vectorized_data = None
            self.total_dataset_len = None
            self.device = torch.device('cpu')

        self.sqrt_weights_for_loss = torch.tensor(np.sqrt(np.exp(- np.arange(0, prediction_depth)
                                                                 / prediction_depth * 2) /
                                                          sum(np.exp(- np.arange(0, prediction_depth) /
                                                                     prediction_depth * 2))), device=self.device)

    def get_total_dataset_len(self) -> int:
        return self.total_dataset_len

    def min_max_norm(self, input_vec: np.ndarray) -> tuple:
        """
        This function normalizes the input vector to the range [-1, 1] and returns the normalization parameters.
        Args:
            input_vec: as the name suggests

        Returns:
            normalized_vec: the normalized vector
            a, b: the normalization parameters, such that normalized_vec = input_vec * a + b

        """
        a = 2 / np.ptp(input_vec)
        b = -1 - a * np.min(input_vec)
        return input_vec * a + b, a, b

    def min_max_denorm(self, normalized_vec: np.ndarray, a: float = None, b: float = None) -> np.ndarray:
        """
        This function denormalizes the input vector from the range [-1, 1] to the original range.
        Args:
            normalized_vec: as the name suggests
            a: the normalization parameters, such that normalized_vec = input_vec * a + b
            b: ^^

        Returns:
            input_vec: the denormalized vector

        """
        if a is None:
            a = self.NET_CONSUMPTION_NORM_A
        if b is None:
            b = self.NET_CONSUMPTION_NORM_B

        return (normalized_vec - b) / a

    def read_and_pad_data(self, path: str, title: str, normalize=False) -> np.ndarray:
        """
        This function reads the data from the csv and pads it with zeros according to the prediction and input lengths.
        Args:
            path: the path to the csv file
            title: the title of the column to be read
            normalize: should the data be normalized to the range [-1, 1]

        Returns:
            data: the padded data

        """
        input_vec = pd.read_csv(path, usecols=[title]).to_numpy().squeeze()
        input_vec = np.pad(input_vec, (self.input_len, self.prediction_depth))

        if normalize:
            input_vec = self.min_max_norm(input_vec)[0]

        return input_vec

    def store_env_global_data(self) -> np.ndarray:
        """
        This function stores and returns the global environmental data, which is common for all buildings.
        Returns:
            self.env_global_data: the global environmental data

        """

        drybulb = self.read_and_pad_data('../data/citylearn_challenge_2022_phase_1/weather.csv',
                                         'Outdoor Drybulb Temperature [C]', normalize=True)
        humidity = self.read_and_pad_data('../data/citylearn_challenge_2022_phase_1/weather.csv',
                                          'Relative Humidity [%]', normalize=True)
        diffuse = self.read_and_pad_data('../data/citylearn_challenge_2022_phase_1/weather.csv',
                                         'Diffuse Solar Radiation [W/m2]', normalize=True)
        direct = self.read_and_pad_data('../data/citylearn_challenge_2022_phase_1/weather.csv',
                                        'Direct Solar Radiation [W/m2]', normalize=True)
        pricing = self.read_and_pad_data('../data/citylearn_challenge_2022_phase_1/pricing.csv',
                                         'Electricity Pricing [$]', normalize=True)
        carbon = self.read_and_pad_data('../data/citylearn_challenge_2022_phase_1/carbon_intensity.csv',
                                        'kg_CO2/kWh', normalize=True)
        total_net_consumption = 0 * carbon
        for i in range(5):
            total_net_consumption += self.read_and_pad_data(
                '../data/buildings_net_electricity_consumption_without_storage.csv',
                f'Building_{i}', normalize=False)
        total_net_consumption = self.min_max_norm(total_net_consumption)[0]

        self.env_global_data = np.stack((drybulb, humidity, diffuse, direct, pricing, carbon, total_net_consumption),
                                        axis=1)
        np.save(f'data/global_env_data_{self.input_len}_{self.prediction_depth}.npy', self.env_global_data)
        return self.env_global_data

    def load_env_global_data(self) -> np.ndarray:
        """
        This function loads the global environmental data, which is common for all buildings, in case it's already
        stored.
        Returns:
            self.env_global_data: the global environmental data

        """
        self.env_global_data = np.load(f'data/global_env_data_{self.input_len}_{self.prediction_depth}.npy')
        return self.env_global_data

    def get_env_global_data(self) -> np.ndarray:
        """
        This function checks if the global environmental data is already stored, and if not, it generates it.
        Returns:
            self.env_global_data: the global environmental data

        """
        try:
            return self.load_env_global_data()
        except FileNotFoundError:
            return self.store_env_global_data()

    def net_consumption_norm_and_stack_env_global_data(self, input_vec: np.ndarray, time_idx: int) -> np.ndarray:
        """
        This function normalizes the net consumption data and stacks it with the corresponding environmental data.
        Args:
            input_vec: net consumption data
            time_idx: time index of input_vec's first element

        Returns:
            stacked_data: the stacked data, with shape (input_len, 1+7+7) [7 is the number of environmental channels]

        """
        time_idx += self.input_len
        assert time_idx < 8760 + self.input_len + self.prediction_depth - 1, 'time_idx out of range'

        input_vec = input_vec * self.NET_CONSUMPTION_NORM_A + self.NET_CONSUMPTION_NORM_B

        return np.concatenate((input_vec[:, None], self.env_global_data[time_idx:time_idx + len(input_vec), :],
                               self.env_global_data[time_idx + self.prediction_depth
                                                    :time_idx + self.prediction_depth + len(input_vec), :]), axis=1)

    def store_stacked_vectorized_data(self) -> np.ndarray:
        """
        This function stores and returns the stacked vectorized data, which is the total training+validation dataset.
        Returns:
            self.stacked_vectorized_data: the stacked vectorized data

        """
        stacked_data = np.zeros((8760 + self.input_len + self.prediction_depth, 8, 5))
        for building_num in range(5):
            net_cons_curr_building = self.read_and_pad_data(
                '../data/buildings_net_electricity_consumption_without_storage.csv', f'Building_{building_num}',
                normalize=False)
            stacked_data[:, :, building_num] = self.net_consumption_norm_and_stack_env_global_data(
                net_cons_curr_building, -self.input_len)[:, :9]

        if self.input_len == self.prediction_depth:
            stacked_vectorized_data = np.reshape(np.swapaxes(stacked_data, 1, 2), (-1, 8))
        elif self.input_len > self.prediction_depth:
            stacked_data = stacked_data[self.input_len - self.prediction_depth:, :, :]
            stacked_vectorized_data = np.reshape(np.swapaxes(stacked_data, 1, 2), (-1, 8))
            stacked_vectorized_data = np.append(
                np.zeros((self.input_len - self.prediction_depth, 8)), stacked_vectorized_data, axis=0)
        else:
            stacked_data = stacked_data[:-self.prediction_depth + self.input_len, :, :]
            stacked_vectorized_data = np.reshape(np.swapaxes(stacked_data, 1, 2), (-1, 8))
            stacked_vectorized_data = np.append(stacked_vectorized_data,
                                                np.zeros((self.prediction_depth - self.input_len, 8)), axis=0)

        np.save(f'data/stacked_vectorized_data_{self.input_len}_{self.prediction_depth}.npy',
                stacked_vectorized_data)
        return stacked_vectorized_data

    def load_stacked_vectorized_data(self) -> np.ndarray:
        """
        This function loads the stacked vectorized data, which is the total training+validation dataset, in case it's
        already stored.
        Returns:
            self.stacked_vectorized_data: the stacked vectorized data

        """
        stacked_vectorized_data = np.load(
            f'data/stacked_vectorized_data_{self.input_len}_{self.prediction_depth}.npy')
        return stacked_vectorized_data

    def get_stacked_vectorized_data(self) -> np.ndarray:
        """
        This function checks if the stacked vectorized data is already stored, and if not, it generates it.
        Returns:
            self.stacked_vectorized_data: a torch tensor with the stacked vectorized data, of shape:
                                    [self.total_dataset_len + self.prediction_depth, 8]

        """
        try:
            stacked_vectorized_data = self.load_stacked_vectorized_data()
        except FileNotFoundError:
            stacked_vectorized_data = self.store_stacked_vectorized_data()

        return torch.from_numpy(stacked_vectorized_data).to(self.device)

    def get_training_batch(self, idx_list: list) -> tuple:
        """
        runs on self.stacked_vectorized_data and gives a batch of data according to idx_list
        Args:
            idx_list: list of first indices per sample
        Returns: (x, y) tuple of torch tensors
            x: [Batch, Input length, Channels=15]. Channel: (net consumption history[1], global env data history[7],
                                                                                         global env data future[7])
            y: [Batch, Prediction length]

        """

        x = torch.zeros((len(idx_list), self.input_len, 15), device=self.device)
        y = torch.zeros((len(idx_list), self.prediction_depth), device=self.device)

        for i, idx in enumerate(idx_list):
            future = self.stacked_vectorized_data[idx + self.input_len:idx + self.input_len + self.prediction_depth, :]
            x[i, :, :] = torch.cat((self.stacked_vectorized_data[idx:idx + self.input_len, :], future[:, 1:]), dim=1)
            y[i, :] = future[:, 0]

        return x, y

    def get_inference_batch(self, input_vec: np.ndarray, time_idx: int) -> np.ndarray:
        """
        generates a batch of data for inference (batch size = 1)
        Args:
            input_vec: net consumption data, of shape [input_len]
            time_idx: time index of input_vec's first element

        Returns:
            x: torch tensor for inference

        """

        x = torch.from_numpy(self.net_consumption_norm_and_stack_env_global_data(input_vec, time_idx)).to(self.device)

        return x

    def translate_prediction_to_net_consumption(self, prediction: torch.tensor) -> np.ndarray:
        """
        translates a prediction to net consumption
        Args:
            prediction: a prediction, of shape [prediction_depth]
        Returns:
            net_consumption: the net consumption, of shape [prediction_depth]

        """
        return self.min_max_denorm(prediction.detach().numpy(), self.NET_CONSUMPTION_NORM_A, self.NET_CONSUMPTION_NORM_B)

    def sqrt_weight(self, input_vec: np.ndarray) -> np.ndarray:
        return input_vec * self.sqrt_weights_for_loss
