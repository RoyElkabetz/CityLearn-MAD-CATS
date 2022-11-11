import numpy as np
import torch


class ConsPredictors:

    def __init__(self, prediction_depth: int = 24, method: str = 'MLP', sequal_prediction=False):
        assert 0 < prediction_depth < 8760, "The prediction depth must be an integer between 0 and 8760"

        self.input_len = 24  # implemented only for 24 now
        self.method = method
        self.prediction_depth = prediction_depth

        self.net_consumption = np.load('data/net_consumption_24.npy')

        if self.method == "DOT":
            self.norm_vec = np.load('data/net_consumption_norm24.npy')
        elif self.method == "MLP":
            from data_preprocessing import DataPreprocessing
            from cons_mlp.mlp_model import MLP
            import torch

            self.input_len = 24
            self.sequal_prediction = sequal_prediction
            batch_size = 1024
            type_str = 'sequal' if sequal_prediction else 'deep'

            self.data = DataPreprocessing(training_mode=False, input_len=self.input_len,
                                          prediction_depth=self.prediction_depth)
            self.model = MLP(device=self.data.device, batch_size=batch_size,
                             sequal_prediction=self.sequal_prediction).to(self.data.device)

            checkpoint = torch.load(f'cons_mlp/best_epoch_{type_str}.pt', map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint)
            self.model.eval()

        elif self.method != "IDX":
            raise NotImplementedError("Method not implemented")

    def from_known_idx(self, idx: int, building_num: None) -> np.ndarray:
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

    def dot_product(self, input_vec: np.ndarray) -> np.ndarray:
        """
        Predict the future based on maximal correlation with the past.
        Args:
            input_vec: The input vector of length 24.
        """

        assert len(input_vec) == 24, "The input vector must be of length 24"

        normalized_input = input_vec / np.sqrt(sum(input_vec ** 2))

        correlation = np.array([np.sum(normalized_input * self.net_consumption[i:i + 24])
                                for i in range(len(self.net_consumption) - 47)])
        correlation /= self.norm_vec

        idx_from_correlation = np.argmax(correlation)

        return self.from_known_idx(idx_from_correlation + len(input_vec), None)

    def check_train(self, model_input: np.ndarray, idx: int, building_num: int = 10):
        if building_num > 4:
            return False
        else:
            idx += 23 + 23 + (23 + 8760) * building_num
            if model_input[-1] == self.net_consumption[idx]:
                # print(f"{self.method} prediction found")
                return True
            else:
                # print(f"{self.method} from scratch")
                return False


    def look_in_train(self, model_input: np.ndarray, idx: int, building_num: int = 10):
        if self.check_train(model_input, idx, building_num):
            idx += 23 + 23 + (23 + 8760) * building_num
            return self.from_known_idx(idx + 1, None)
        else:
            return None

    def mlp_predict(self, model_input: np.ndarray, time_idx: int = None, building_num: int = None) -> np.ndarray:
        assert len(model_input) == 24, "The input vector must be of length 24"

        data_stack = self.data.net_consumption_norm_and_stack_env_global_data(model_input, time_idx)
        x = torch.unsqueeze(torch.from_numpy(data_stack).float().to('cpu'), 0)
        pred = self.model(x).view(-1).detach().cpu().numpy()
        if self.sequal_prediction:
            prediction = np.ones(self.prediction_depth) * pred[0]
            for d in range(1, self.prediction_depth):
                model_input = np.concatenate((model_input[1:], pred))
                data_stack = self.data.net_consumption_norm_and_stack_env_global_data(model_input, time_idx + d)
                x = torch.unsqueeze(torch.from_numpy(data_stack).float().to('cpu'), 0)
                pred = self.model(x).view(-1).detach().cpu().numpy()
                prediction[d] = pred[0]
        else:  # deep prediction (not sequential)
            prediction = pred

        return self.data.min_max_denorm(prediction)

    def get_prediction(self, model_input: np.ndarray, time_idx: int = None, building_num: int = None,
                       known_building=False) -> np.ndarray:
        """
        Predict the future based on a method.
        Args:
            model_input: The input vector of length 24.
            time_idx: The time index of the net consumption vector (0 to 8760-24).
            building_num: The building number to predict (0 to 4).
            known_building: override prediction from known building
        """

        if self.method == "IDX" or known_building:
            prediction = self.from_known_idx(time_idx + self.input_len, building_num)
        elif self.method == "DOT":
            # prediction = self.look_in_train(model_input, time_idx, building_num)
            # if prediction is None:
            prediction = self.dot_product(model_input)
        elif self.method == "MLP":
            # prediction = self.look_in_train(model_input, time_idx, building_num)
            # if prediction is None:
            prediction = self.mlp_predict(model_input, time_idx, building_num)
        else:
            raise NotImplementedError("Method not implemented")

        return prediction
