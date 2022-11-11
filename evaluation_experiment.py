from utils import rb_evaluate
import numpy as np
import pandas as pd
from agents.battery_model_rb_agent_ import BatteryModelRBAgent
from agents.brute_force_day_agent import BruteForceLastAgent, BruteForceAgent
from agents.controller import RBLastAgentDecentralizedCoordinator, RBDecentralizedCoordinator
import random


class ParametersList:
    def __init__(self, n_experiments=1000):
        self.n_experiments = n_experiments
        self.counter = 0
        self.done = False
        self.depth_alpha_range = [1, 2]
        self.max_search_depth = [2, 3, 4, 5, 6, 7]
        self.d_action = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        self.default_action = [0.0]
        self.utility_weighting = []
        for i1 in range(1, 11):
            for i2 in range(1, 11):
                for i3 in range(1, 11):
                    for i4 in range(1, 11):
                        self.utility_weighting.append((1./i1, 1./i2, 1./i3, 1/i4))
        self.action_space_list = [(0, -0.05, 0.05, -1, 1),
                                  (0, -0.05, 0.05, -0.1, 0.1, -1, 1),
                                  (0, -0.05, 0.05, -0.1, 0.1, -0.3, 0.3, -1, 1),
                                  (0, -0.1, 0.1, -0.3, 0.3, -1, 1),
                                  (0, -0.05, -0.1, -0.3, -1, 1),
                                  ]

        self.memory = {}

    def sample_experiment(self):
        if self.counter == self.n_experiments:
            self.done = True
            return None
        depth_alpha = self.depth_alpha_range[0] + np.random.rand() * np.diff(self.depth_alpha_range)
        max_search_depth = random.sample(self.max_search_depth, 1)[0]
        search_depths = (np.arange(max_search_depth)**depth_alpha).astype(int)
        d_action = random.sample(self.d_action, 1)[0]
        utility_weighting = (1., 1., 1., 1.) if random.random() < 0.5 else random.sample(self.utility_weighting, 1)[0]
        action_space_list = None if random.random() < 0.8 else random.sample(self.action_space_list, 1)[0]
        params = {"search_depths": tuple(search_depths),
                  "d_action": d_action,
                  "plan_steps": search_depths[-1],
                  "default_action": 0.0,
                  "utility_weighting": utility_weighting,
                  "action_space_list": action_space_list
                  }
        experiment = tuple(params.values())
        if experiment not in self.memory.keys():
            self.memory[experiment] = params
            return params
        else:
            return None

    def sample_weight_experiment(self):
        utility_weighting = random.sample(self.utility_weighting, 1)[0]
        params = {"search_depths": (0, 1, 2, 3, 4, 5, 6),
                  "d_action": 0.2,
                  "max_search_time": 0.2,
                  "plan_steps": 1000,
                  "default_action": 0.0,
                  "random_order": False,
                  "utility_weighting": utility_weighting,
                  "action_space_list": None
                  }
        experiment = tuple(params.values())
        if experiment not in self.memory.keys():
            self.memory[experiment] = params
            return params
        else:
            return None


def main(n_experiments=10000):
    experiments_df = pd.DataFrame(columns=["utility", "price_cost", "emission_cost", "grid_cost", "search_depths",
                                           "d_action", "plan_steps", "default_action", "utility_weighting",
                                           "action_space_list"])

    param_list = ParametersList(n_experiments)
    counter = 0
    while not param_list.done:
        parameters = param_list.sample_weight_experiment()
        counter += 1
        if parameters is not None:
            print(counter, parameters)

            # Define the number of episodes being played and the schema
            class EvalConstants:
                episodes = 1
                compute_metric_interval = 1000
                schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'
                evaluation_days = episodes * 365
                sac_params = ()
                rule_based_params = parameters
                # rule_based_params = {}

            try:
                env, rl_coordinator, episode_metrics, evaluation_time = rb_evaluate(RBDecentralizedCoordinator,
                                                                                    BruteForceAgent,
                                                                                    EvalConstants,
                                                                                    verbose=True)

                episode_metrics = episode_metrics[0]
                episode_metrics["utility"] = np.mean(list(episode_metrics.values()))
                episode_metrics.update(parameters)
                experiments_df = experiments_df.append(episode_metrics, ignore_index=True)
                experiments_df.to_csv("brute_force_last_agent__evaluation_experiment.csv")
                param_list.counter += 1

            except Exception as e:
                print(e)


if __name__ == '__main__':
    main()




