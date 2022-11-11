import matplotlib.pyplot as plt
from utils import rb_evaluate
plt.rcParams.update({'font.size': 16})
from agents.brute_force_day_agent import BruteForceAgent, BruteForceLastAgent
from agents.controller import RBDecentralizedCoordinator, RBLastAgentDecentralizedCoordinator


# Define the number of episodes being played and the schema
class EvalConstants:
    # Environmental and simulation parameters
    episodes = 1
    compute_metric_interval = 100
    schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'
    evaluation_days = episodes * 365

    # Controller and agents' parameters
    rule_based_params = {"search_depths": [0, 1, 2, 4],
                         "max_search_time": 0.2,
                         "d_action": 0.2,
                         "utility_weighting": (1., 1., 1., 1.),
                         "random_order": True,
                         "action_space_list": None,     # [0, -0.05, 0.05, -0.1, 0.1, -0.3, 0.3, -1, 1]
                         "prediction_method": "IDX",
                         "agent_type": "RB-global",             # ("RB-local", "PLAN-local")
                         "last_agent_type": "RB-global",      # ("RB-local", "RB-global", "PLAN-local", "PLAN-global")
                        }


class EvalConstantsLast:
    # Environmental and simulation parameters
    episodes = 1
    compute_metric_interval = 100
    schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'
    evaluation_days = episodes * 365

    # Controller and agents' parameters
    rule_based_params = {"search_depths": [0, 1, 2, 4],
                         "max_search_time": 0.2,
                         "d_action": 0.2,
                         "utility_weighting": (1., 1., 1., 1.),
                         "action_space_list": None,     # [0, -0.05, 0.05, -0.1, 0.1, -0.3, 0.3, -1, 1]
                        }



if __name__ == '__main__':

    # verbos = True
    # env, rl_coordinator, episode_metrics, evaluation_time = rb_evaluate(RBDecentralizedCoordinator,
    #                                                                     BruteForceAgent,
    #                                                                     EvalConstants,
    #                                                                     verbose=verbos)

    verbos = True
    env, rl_coordinator, episode_metrics, evaluation_time = rb_evaluate(RBLastAgentDecentralizedCoordinator,
                                                                        BruteForceLastAgent,
                                                                        EvalConstantsLast,
                                                                        verbose=verbos)