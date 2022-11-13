import numpy as np
from data_preprocessing import DataPreprocessing
from mlp_model import MLP
import torch


class PredictorMLP(object):
    def __init__(self, input_len: int = 24, prediction_depth: int = 24, batch_size: int = 1024,
                 sequal_prediction=True):
        self.input_len = input_len
        self.prediction_depth = prediction_depth
        self.sequal_prediction = sequal_prediction

        type_str = 'sequal' if sequal_prediction else 'deep'

        self.data = DataPreprocessing(training_mode=False, input_len=input_len, prediction_depth=prediction_depth)
        self.model = MLP(device=self.data.device, batch_size=batch_size, sequal_prediction=sequal_prediction)\
            .to(self.data.device)

        checkpoint = torch.load(f'best_epoch_{type_str}.pt')
        self.model.load_state_dict(checkpoint)
        self.model.eval()

        self.net_consumption = np.load('data/net_consumption_24.npy')

    def from_known_idx(self, idx: int, building_num: int = None) -> np.ndarray:
        """
        Predict the future based on a known index.
        Args:
            idx: The time index of the net consumption vector.
            building_num: The building number to predict (0 to 4).
        """
        if building_num is not None:
            assert isinstance(building_num, int), "The building number must be an integer"
            if building_num > 4:
                building_num = None  # as we only have CSVs for 5 buildings

        if building_num is not None:
            idx += 23 + (23 + 8760) * building_num

        prediction = self.net_consumption[idx:idx + self.prediction_depth]

        return prediction
        
    def look_in_train(self, model_input: np.ndarray, idx: int, building_num: int = None):
        idx += 23 + (23 + 8760) * building_num
        if model_input[-1] == self.net_consumption[idx]:
            return self.from_known_idx(idx, None)
        else:
            return None

    def predict(self, model_input: np.ndarray, time_idx: int = None, building_num: int = None) -> np.ndarray:
        prediction = self.look_in_train(model_input, time_idx, building_num)
        if prediction is not None:
            return prediction
        else:
            x = self.data.net_consumption_norm_and_stack_env_global_data(model_input, time_idx)
            pred = self.model(x)
            if self.sequal_prediction:
                for d in range(1, self.prediction_depth):
                    model_input = torch.cat((model_input[1:], pred[0]), dim=0)
                    x = self.data.net_consumption_norm_and_stack_env_global_data(model_input, time_idx)
                    pred = self.model(x)

            return self.data.min_max_denorm(pred[0, :].view(-1).detach().cpu().numpy())
