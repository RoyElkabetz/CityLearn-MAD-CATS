import numpy as np
import copy as cp
from gym.spaces.box import Box
from agents.battery_model_rb_agent import BatteryModelRBAgent
from predictors.predictors_wrapper import DistrictPredictorWrapper


def dict_to_box(x):
    """
    Transforms a dictionary space into a Gym's Box space
    :param x: a dictionary space
    :return: Box space
    """
    return Box(low=x["low"], high=x["high"], shape=(x["shape"]), dtype=x["dtype"])


class RBDecentralizedCoordinator:
    def __init__(self, base_decision_maker, obs_dict, random_order=False, agent_type="RB-local",
                 last_agent_type="RB-local", *args, **kwargs):
        assert agent_type in ("RB-local", "RB-global", "PLAN-local")
        assert last_agent_type in ("RB-local", "RB-global", "PLAN-local", "PLAN-global")
        self.base_decision_maker = base_decision_maker
        self.obs_dict = obs_dict
        self.decision_makers = {}
        self.action_space = {}
        self.observation_space = {}
        self.agent_type = agent_type
        self.last_agent_type = last_agent_type
        self.num_buildings = None
        self.num_steps = 0
        self.num_episodes = 0
        self.random_order = random_order
        self.scores_and_metrics = {
            "metric_eval_step": [],
            "price_cost": [],
            "emission_cost": [],
            "grid_cost": [],
            "metric_value": [],
            "avg_metric_value": [],
            "best_metric_value": np.inf,
            "scores": {},
            "avg_scores": {},
            "current_scores": {},
            "current_avg_scores": {},
            "best_avg_scores": {},
            "rewards_memory": {},
            "avg_rewards_memory": {},
            "current_avg_rewards": {},
            "best_rewards": {},
        }
        self.observation_history = []

        self.init_decision_makers(*args, **kwargs)

    def init_decision_makers(self, *args, **kwargs):
        """
        Initialize decision makers for coordinator
        """
        observations = self.obs_dict["observation"]
        self.observation_space = [self.obs_dict["observation_space"][i] for i, _ in
                                  enumerate(self.obs_dict["observation_space"])]
        self.action_space = self.obs_dict["action_space"]
        self.num_buildings = len(observations)

        for agent_id in range(self.num_buildings):
            self.decision_makers[agent_id] = self.base_decision_maker(*args, **kwargs)
            self.decision_makers[agent_id].set_action_space(agent_id, dict_to_box(self.action_space[agent_id]))
            print(f"Initialize a rule-based decision maker for agent: {agent_id}.")

        # init score and metric collection
        self.scores_and_metrics['current_scores'] = [0] * self.num_buildings
        self.scores_and_metrics['best_avg_scores'] = [-np.inf] * self.num_buildings
        self.scores_and_metrics['current_avg_rewards'] = [0] * self.num_buildings
        self.scores_and_metrics['best_rewards'] = [-np.inf] * self.num_buildings

        for agent_id in range(self.num_buildings):
            self.scores_and_metrics['scores'][agent_id] = []
            self.scores_and_metrics['avg_scores'][agent_id] = []
            self.scores_and_metrics['rewards_memory'][agent_id] = []
            self.scores_and_metrics['avg_rewards_memory'][agent_id] = []

    def compute_action(self, observations):
        """
        Compute the joint action.
        """
        # save all observations for later plot and analysis
        self.observation_history.append(observations)

        prev_district_net_cons = 0.0
        predicted_rest_district_next_net_consumption = 0.
        rest_district_actions_sum_kwh = 0.

        actions = [[0.0]] * self.num_buildings
        if self.random_order:
            agents_order = self.get_random_order()
        else:
            agents_order = np.arange(self.num_buildings)

        for i, agent_id in enumerate(agents_order):
            prev_district_net_cons += observations[agent_id][23]
            action = [0]
            if i >= self.num_buildings - 1:
                if self.last_agent_type == "RB-local":
                    action = self.decision_makers[agent_id].compute_local_rb_action(observations[agent_id],
                                                                                    agent_id)
                elif self.last_agent_type == "RB-global":
                    action = self.decision_makers[agent_id].compute_global_rb_action(
                        observation=observations[agent_id],
                        agent_id=agent_id,
                        rest_district_net_cons_prediction=(rest_district_actions_sum_kwh +
                                                           predicted_rest_district_next_net_consumption),
                        prev_rest_district_net_cons=prev_district_net_cons)
                elif self.last_agent_type == "PLAN-local":
                    action = self.decision_makers[agent_id].compute_local_planner_action(observations[agent_id], agent_id)
                elif self.last_agent_type == "PLAN-global":
                    raise NotImplementedError
            else:
                if self.agent_type == "RB-local":
                    action = self.decision_makers[agent_id].compute_local_rb_action(observations[agent_id],
                                                                                    agent_id)
                elif self.agent_type == "RB-global":
                    action = self.decision_makers[agent_id].compute_global_rb_action(
                        observation=observations[agent_id],
                        agent_id=agent_id,
                        rest_district_net_cons_prediction=(rest_district_actions_sum_kwh +
                                                           predicted_rest_district_next_net_consumption),
                        prev_rest_district_net_cons=prev_district_net_cons)
                elif self.agent_type == "PLAN-local":
                    action = self.decision_makers[agent_id].compute_local_planner_action(observations[agent_id], agent_id)
                elif self.agent_type == "PLAN-global":
                    raise NotImplementedError

            rest_district_actions_sum_kwh += action[0] * self.decision_makers[agent_id].battery.capacity
            predicted_rest_district_next_net_consumption += \
            self.decision_makers[agent_id].predict_world()["building_net_consumption"][0]
            actions[agent_id] = action

        return actions

    def get_random_order(self):
        """
        Randomize the order of agents' action
        :return: random array of buildings' numbers
        """
        old_order = np.arange(self.num_buildings)
        return np.random.permutation(old_order)

    def collect_scores(self, rewards):
        """
        Collect rewards and compute scores
        """
        # compute scores and average scores
        self.num_steps += 1
        for agent_id in self.decision_makers.keys():
            # save rewards
            self.scores_and_metrics['rewards_memory'][agent_id].append(rewards[agent_id])
            self.scores_and_metrics['avg_rewards_memory'][agent_id].append(
                np.mean(self.scores_and_metrics['rewards_memory'][agent_id][-100:]))
            self.scores_and_metrics['current_avg_rewards'][agent_id] = cp.copy(
                self.scores_and_metrics['avg_rewards_memory'][agent_id][-1])

            # compute scores
            self.scores_and_metrics['current_scores'][agent_id] += rewards[agent_id]

    def init_score(self):
        """
        Zeroth the score in the beginning of a new episode.
        """
        self.scores_and_metrics['current_scores'] = [0] * self.num_buildings

    def reset_battery(self):
        # TODO verify this function working properly
        if isinstance(self.decision_makers[0], BatteryModelRBAgent):
            for agent_id in self.decision_makers.keys():
                self.decision_makers[agent_id].reset()

    def compute_avg_scores(self):
        """
        Compute the average scores over episodes
        """
        for agent_id in self.decision_makers.keys():
            self.scores_and_metrics['scores'][agent_id].append(self.scores_and_metrics['current_scores'][agent_id])
            self.scores_and_metrics['avg_scores'][agent_id].append(
                np.mean(self.scores_and_metrics['scores'][agent_id][-100:]))

    def collect_metrics(self, metrics):
        """
        Collect environmental metrics
        """
        self.scores_and_metrics["metric_eval_step"].append(self.num_steps)
        self.scores_and_metrics["price_cost"].append(metrics[0])
        self.scores_and_metrics["emission_cost"].append(metrics[1])
        self.scores_and_metrics["grid_cost"].append(metrics[2])
        self.scores_and_metrics["metric_value"].append(np.sum(metrics) / 3)
        self.scores_and_metrics["avg_metric_value"].append(np.mean(self.scores_and_metrics["metric_value"][-10:]))

    def print_scores_and_metrics(self, episodes_completed, num_steps):
        """
        Print score and metric state into screen
        """
        last_avg_scores = []
        for agent_id in self.decision_makers.keys():
            if len(self.scores_and_metrics['avg_scores'][agent_id]) > 0:
                last_avg_scores.append(np.round(self.scores_and_metrics['avg_scores'][agent_id][-1], 1))
            else:
                last_avg_scores.append(None)
        print(
            "| Episode: {:2d} | Steps: {:6d} | Price cost {:.4f} | Emission cost {:.4f} | Grid cost {:.4f} | Avg scores: {} | Best scores: {} |"
            .format(episodes_completed,
                    num_steps,
                    np.round(self.scores_and_metrics['price_cost'][-1], 4),
                    np.round(self.scores_and_metrics['emission_cost'][-1], 4),
                    np.round(self.scores_and_metrics['grid_cost'][-1], 4),
                    last_avg_scores,
                    np.round(self.scores_and_metrics['best_avg_scores'], 3)))


class RBWeighedDecentralizedCoordinator:
    def __init__(self, base_decision_maker, obs_dict, input_length: int = None, *args, **kwargs):
        self.base_decision_maker = base_decision_maker
        self.obs_dict = obs_dict
        self.decision_makers = {}
        self.action_space = {}
        self.observation_space = {}
        self.input_length = input_length
        self.num_buildings = None
        self.num_steps = 0
        self.num_episodes = 0
        self.world_predictor = None
        self.observation_history = []

        print("The RBWeighedDecentralizedCoordinator class is not working properly, should be debug !!!")

        self.scores_and_metrics = {
            "metric_eval_step": [],
            "price_cost": [],
            "emission_cost": [],
            "grid_cost": [],
            "metric_value": [],
            "avg_metric_value": [],
            "best_metric_value": np.inf,
            "scores": {},
            "avg_scores": {},
            "current_scores": {},
            "current_avg_scores": {},
            "best_avg_scores": {},
            "rewards_memory": {},
            "avg_rewards_memory": {},
            "current_avg_rewards": {},
            "best_rewards": {},

        }

        self.init_decision_makers(*args, **kwargs)

    def init_decision_makers(self, *args, **kwargs):
        """
        Initialize decision makers for the coordinator
        """
        observations = self.obs_dict["observation"]
        self.observation_space = [self.obs_dict["observation_space"][i] for i, _ in
                                  enumerate(self.obs_dict["observation_space"])]
        self.action_space = self.obs_dict["action_space"]
        self.num_buildings = len(observations)

        # initialize the world predictor
        self.world_predictor = DistrictPredictorWrapper(self.num_buildings, input_length=self.input_length,
                                                        prediction_depth=int(kwargs["search_depths"][-1] + 1))

        for agent_id in range(self.num_buildings):
            self.decision_makers[agent_id] = self.base_decision_maker(predictor=self.world_predictor,
                                                                      num_buildings=self.num_buildings,
                                                                      *args, **kwargs)
            self.decision_makers[agent_id].set_action_space(agent_id, dict_to_box(self.action_space[agent_id]))
            print(f"Initialize a rule-based decision maker for agent: {agent_id}.")



        # init score and metric collection
        self.scores_and_metrics['current_scores'] = [0] * self.num_buildings
        self.scores_and_metrics['best_avg_scores'] = [-np.inf] * self.num_buildings
        self.scores_and_metrics['current_avg_rewards'] = [0] * self.num_buildings
        self.scores_and_metrics['best_rewards'] = [-np.inf] * self.num_buildings

        for agent_id in range(self.num_buildings):
            self.scores_and_metrics['scores'][agent_id] = []
            self.scores_and_metrics['avg_scores'][agent_id] = []
            self.scores_and_metrics['rewards_memory'][agent_id] = []
            self.scores_and_metrics['avg_rewards_memory'][agent_id] = []

    def compute_action(self, observations):
        """
        Compute the joint action.
        """
        # save all observations for later plot and analysis
        self.observation_history.append(observations)

        actions = []
        for agent_id in self.decision_makers.keys():
            action = self.decision_makers[agent_id].compute_action(observations[agent_id], agent_id)
            actions.append(action)

            # update the world predictor with the agent's action
            plan_kwh = np.array(self.decision_makers[agent_id].plan) * self.decision_makers[agent_id].battery.capacity
            self.world_predictor.append_plan(building_id=agent_id, plan=plan_kwh)

        return actions

    def collect_scores(self, rewards):
        """
        Collect rewards and compute scores
        """
        # compute scores and average scores
        self.num_steps += 1
        for agent_id in self.decision_makers.keys():
            # save rewards
            self.scores_and_metrics['rewards_memory'][agent_id].append(rewards[agent_id])
            self.scores_and_metrics['avg_rewards_memory'][agent_id].append(
                np.mean(self.scores_and_metrics['rewards_memory'][agent_id][-100:]))
            self.scores_and_metrics['current_avg_rewards'][agent_id] = cp.copy(
                self.scores_and_metrics['avg_rewards_memory'][agent_id][-1])

            # compute scores
            self.scores_and_metrics['current_scores'][agent_id] += rewards[agent_id]

    def init_score(self):
        """
        Zeroth the score in the beginning of a new episode.
        """
        self.scores_and_metrics['current_scores'] = [0] * self.num_buildings

    def reset_battery(self):
        # TODO verify this function working properly
        if isinstance(self.decision_makers[0], BatteryModelRBAgent):
            for agent_id in self.decision_makers.keys():
                self.decision_makers[agent_id].reset()

    def compute_avg_scores(self):
        """
        Compute the average scores over episodes
        """
        for agent_id in self.decision_makers.keys():
            self.scores_and_metrics['scores'][agent_id].append(self.scores_and_metrics['current_scores'][agent_id])
            self.scores_and_metrics['avg_scores'][agent_id].append(
                np.mean(self.scores_and_metrics['scores'][agent_id][-100:]))

    def collect_metrics(self, metrics):
        """
        Collect environment metrics
        """
        self.scores_and_metrics["metric_eval_step"].append(self.num_steps)
        self.scores_and_metrics["price_cost"].append(metrics[0])
        self.scores_and_metrics["emission_cost"].append(metrics[1])
        self.scores_and_metrics["grid_cost"].append(metrics[2])
        self.scores_and_metrics["metric_value"].append(np.sum(metrics))
        self.scores_and_metrics["avg_metric_value"].append(np.mean(self.scores_and_metrics["metric_value"][-10:]))

    def print_scores_and_metrics(self, episodes_completed, num_steps):
        """
        Print score and metric state into screen
        """
        last_avg_scores = []
        for agent_id in self.decision_makers.keys():
            if len(self.scores_and_metrics['avg_scores'][agent_id]) > 0:
                last_avg_scores.append(np.round(self.scores_and_metrics['avg_scores'][agent_id][-1], 1))
            else:
                last_avg_scores.append(None)
        print(
            "| Episode: {:2d} | Steps: {:6d} | Price cost {:.4f} | Emission cost {:.4f} | Grid cost {:.4f} | Avg scores: {} | Best scores: {} |"
            .format(episodes_completed,
                    num_steps,
                    np.round(self.scores_and_metrics['price_cost'][-1], 4),
                    np.round(self.scores_and_metrics['emission_cost'][-1], 4),
                    np.round(self.scores_and_metrics['grid_cost'][-1], 4),
                    last_avg_scores,
                    np.round(self.scores_and_metrics['best_avg_scores'], 3)))



