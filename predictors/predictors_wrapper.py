import numpy as np
import pandas as pd
from predictors.consumption_predictors import ConsPredictors


class PredictorsWrapper:
    def __init__(self, agent_id: int, method="MLP"):
        """
        A world predictor class
        Args:
            agent_id: The ID of the agent.
        """
        self.method = method
        self.world_predictor = None
        self.agent_predictor = None
        self.load_models(agent_id)
        self.agent_id = agent_id
        self.known_building = None

    def load_models(self, agent_id):
        """
        Load the predictive models of the agent's expected behaviour (solar generation - non shiftable load)
        and environment parameters (pricing and carbon intensity).
        Args:
            agent_id: The ID of the agent.

        Returns: None

        """
        world_model = self.load_world_model()
        if self.method == "DOT" or self.method == "IDX" or self.method == "MLP":
            self.agent_predictor = ConsPredictors(prediction_depth=24, method=self.method, sequal_prediction=False)
        elif self.method == "CSV":
            self.agent_predictor = self.load_agent_model(agent_id)
        else:
            raise NotImplementedError("Method not implemented")

        self.world_predictor = {"pricing": world_model["pricing"],
                                "carbon": world_model["carbon"],
                                "d_pricing": world_model["d_pricing"],
                                "d_carbon": world_model["d_carbon"],
                                "carbon_mean": world_model["carbon_mean"],
                                }

    def load_agent_model(self, agent_id) -> dict:
        """
        Load the agent's predictive models.
        Args:
            agent_id: The ID of the agent.

        Returns: Predictive model of generation-load

        """
        agent_data = pd.read_csv("data/buildings_net_electricity_consumption_without_storage.csv")
        net_consumption_no_storage = np.squeeze(agent_data[f"Building_{str(agent_id)}"].values)

        agent_model = {
            "net_consumption": net_consumption_no_storage,
            "d_net_consumption": np.concatenate((np.array([0]), np.diff(net_consumption_no_storage))),
        }
        return agent_model

    def load_world_model(self) -> dict:
        """
        Load environment's predictive models.
        Returns: Predictive models of pricing and carbon intensity.

        """
        all_pricing = pd.read_csv("data/citylearn_challenge_2022_phase_1/pricing.csv")
        pricing = np.squeeze(all_pricing["Electricity Pricing [$]"].values)
        carbon = pd.read_csv("data/citylearn_challenge_2022_phase_1/carbon_intensity.csv", squeeze=True).values
        carbon_mean = pd.read_csv("data/carbon_above_future_mean.csv", squeeze=True).values

        world_model = {
            "pricing": pricing,
            "carbon": carbon,
            "carbon_mean": carbon_mean,
            "d_pricing": np.concatenate((np.array([0]), np.diff(pricing))),
            "d_carbon": np.concatenate((np.array([0]), np.diff(carbon))),
        }
        return world_model

    def predict(self, history, prediction_depth=24, consumption_only=False):
        """
        Predict the next 24 hours agent's + environment behaviour.
        Args:
            history: A history array of observations
            prediction_depth: depth of prediction in time

        Returns: tuple of predictions and their derivatives
        (carbon, d_carbon, pricing, d_pricing, net_consumption, d_net_consumption)

        """
        idx = int(history[-1, 0]) + 1  # the +1 is set in order to skip the current state in predictions
        building_net_cons = None
        building_d_net_cons = None

        if self.method == "CSV":
            building_net_cons = self.agent_predictor["net_consumption"][idx:idx + prediction_depth]
            building_d_net_cons = self.agent_predictor["d_net_consumption"][idx:idx + prediction_depth]
        elif self.method == "IDX" or self.method == "DOT" or self.method == "MLP":
            idx_obs_net_consumption = 23 + 1
            idx_obs_soc = 22 + 1
            net_consumption_no_battery_history = history[:, idx_obs_net_consumption] - np.diff(
                np.concatenate((np.array([0]), history[:, idx_obs_soc]))) * 6.4
            if self.known_building is None:
                self.known_building = self.agent_predictor.check_train(model_input=net_consumption_no_battery_history,
                                                                       idx=idx - 24, building_num=self.agent_id)
                print(f"Building {self.agent_id} is known: {self.known_building}")
            building_net_cons = self.agent_predictor.get_prediction(model_input=net_consumption_no_battery_history,
                                                                    time_idx=idx - 24, building_num=self.agent_id,
                                                                    known_building=self.known_building)
            building_d_net_cons = np.diff(np.concatenate((np.array([history[-1, idx_obs_net_consumption]]),
                                                          building_net_cons)))
        if consumption_only:
            predictions = {
                "building_net_consumption": building_net_cons[:prediction_depth],
                "building_d_net_consumption": building_d_net_cons[:prediction_depth]
            }
        else:
            predictions = {
                "carbon": self.world_predictor["carbon"][idx:idx + prediction_depth],
                "carbon_mean": self.world_predictor["carbon_mean"][idx:idx + prediction_depth],
                "pricing": self.world_predictor["pricing"][idx:idx + prediction_depth],
                "d_carbon": self.world_predictor["d_carbon"][idx:idx + prediction_depth],
                "d_pricing": self.world_predictor["d_pricing"][idx:idx + prediction_depth],
                "building_net_consumption": building_net_cons[:prediction_depth],
                "building_d_net_consumption": building_d_net_cons[:prediction_depth]
            }

        return predictions


class DistrictPredictorWrapper:
    def __init__(self, num_buildings: int, input_length: int = 24, prediction_depth: int = 24):
        """
        A world predictor class
        Args:
            num_buildings: The number of buildings in the district
        """
        self.prediction_depth = prediction_depth
        self.num_buildings = num_buildings
        self.input_length = input_length
        self.world_predictors = None
        self.plan_predictor = {building_id: np.zeros(self.prediction_depth + 1) for building_id in range(self.num_buildings)}
        self.load_models()

    def load_models(self):
        """
        Load the predictive models of the agent's expected behaviour (solar generation - non shiftable load)
        and environment parameters (pricing and carbon intensity).
        Args:
            agent_id: The ID of the agent.

        Returns: None

        """
        district_model = self.load_district_model()
        world_model = self.load_world_model()
        self.world_predictors = {"buildings_net_consumption": district_model["buildings_net_consumption"],
                                 "pricing": world_model["pricing"],
                                 "carbon": world_model["carbon"],
                                 "buildings_d_net_consumption": district_model["buildings_d_net_consumption"],
                                 "d_pricing": world_model["d_pricing"],
                                 "d_carbon": world_model["d_carbon"]
                                 }

    def load_district_model(self) -> dict:
        """
        Load the agent's predictive models.
        Args:
            agent_id: The ID of the agent.

        Returns: Predictive model of generation-load

        """
        agent_data = pd.read_csv("data/buildings_net_electricity_consumption_without_storage.csv")
        buildings_net_consumption_no_storage = []
        buildings_d_net_consumption_no_storage = []

        # get buildings' net consumption
        for building in range(self.num_buildings):
            buildings_net_consumption_no_storage.append(np.squeeze(agent_data[f"Building_{str(building)}"].values))
            buildings_d_net_consumption_no_storage.append(np.concatenate(
                (np.array([0]), np.diff(buildings_net_consumption_no_storage[building]))
            ))

        district_model = {
            "buildings_net_consumption": np.array(buildings_net_consumption_no_storage),
            "buildings_d_net_consumption": np.array(buildings_d_net_consumption_no_storage),
        }
        return district_model

    def load_world_model(self) -> dict:
        """
        Load environment's predictive models.
        Returns: Predictive models of pricing and carbon intensity.

        """
        all_pricing = pd.read_csv("data/citylearn_challenge_2022_phase_1/pricing.csv")
        pricing = np.squeeze(all_pricing["Electricity Pricing [$]"].values)
        carbon = pd.read_csv("data/citylearn_challenge_2022_phase_1/carbon_intensity.csv", squeeze=True).values

        world_model = {
            "pricing": pricing,
            "carbon": carbon,
            "d_pricing": np.concatenate((np.array([0]), np.diff(pricing))),
            "d_carbon": np.concatenate((np.array([0]), np.diff(carbon))),
        }
        return world_model

    def get_building_id(self, history):
        return 0

    def predict_building(self, history, building_id=None):
        """
        Predict the next 24 hours agent's + environment behaviour.
        Args:
            building_id: the building id for which we would like the get the prediction
            history: A history array of observations
            prediction_depth: depth of prediction in time

        Returns: tuple of predictions and their derivatives
        (carbon, d_carbon, pricing, d_pricing, net_consumption, d_net_consumption)

        """
        if building_id is None:
            building_id = self.get_building_id(history)
        idx = int(history[-1, 0]) + 1   # the +1 is set in order to skip the current state in predictions

        district_plans = np.array([self.plan_predictor[b] for b in np.arange(self.num_buildings)])
        district_d_plans = np.diff(district_plans, axis=1)

        predictions = {
            "carbon": self.world_predictors["carbon"][idx:idx + self.prediction_depth],
            "pricing": self.world_predictors["pricing"][idx:idx + self.prediction_depth],
            "d_carbon": self.world_predictors["d_carbon"][idx:idx + self.prediction_depth],
            "d_pricing": self.world_predictors["d_pricing"][idx:idx + self.prediction_depth],
            "building_net_consumption": self.world_predictors["buildings_net_consumption"][building_id][idx:idx + self.prediction_depth],
            "building_d_net_consumption": self.world_predictors["buildings_d_net_consumption"][building_id][idx:idx + self.prediction_depth],
            "district_net_consumption": self.world_predictors["buildings_net_consumption"][:, idx:idx + self.prediction_depth] + district_plans[:, 1:],
            "district_d_net_consumption": self.world_predictors["buildings_d_net_consumption"][:, idx:idx + self.prediction_depth] + district_d_plans,
        }
        return predictions

    def update_predictions(self, building_id, action_kwh, time_idx):
        self.world_predictors["buildings_net_consumption"][building_id, time_idx] += action_kwh

    def append_plan(self, plan, building_id):
        padded_plan = np.pad(np.array(plan), (0, self.prediction_depth - len(plan)), 'constant')
        self.plan_predictor[building_id] = np.concatenate((self.plan_predictor[building_id][1:2], padded_plan))




