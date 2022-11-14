import numpy as np
from battery.battery_model_new import Battery
from battery.battery_model import BatteryTransitionWrapper
from battery.tree_search import ucs
from agents.battery_model_rb_agent import BatteryModelRBAgent
from predictors.predictors_wrapper import PredictorsWrapper


class BruteForceAgent:
    def __init__(self, search_depths=(0, 1, 2, 3), d_action: float = 0.3, max_search_time: float = 0.2,
                 utility_weighting=(1., 1., 1., 1.), action_space_list=None, prediction_method="MLP",
                 *args, **kwargs):
        """
        A Rule-based brute force planning agent. The agent uses predictive models of the environment and its own
        behaviour. It uses a physical model of its battery (actions) as an MDP in order to plan ahead in the environment
        using a tree-search Uniform-Cost-Search (UCS) algorithm with a temporal utility approximation as the cost of
        transition between nodes.
        Args:
            d_action: action discretization in (0, 1)
            search_depths: a list of the tree depths (or prediction times) in which the algorithm should search
            *args: Battery physical model parameters
            **kwargs: Battery physical model parameters
        """
        self.action_space = {}
        self.agent_type = 'RB'
        self.rb_decision_makers = {"local": BatteryModelRBAgent(), "global": BatteryModelRBAgent()}

        self.prediction_method = prediction_method
        self.action_space_list = action_space_list
        self.d_action = d_action
        self.plan = []
        self.utility_weighting = utility_weighting
        self.search_depths = list(search_depths)
        self.max_search_time = max_search_time

        self.battery = Battery(*args, **kwargs)
        self.transition_model_object = BatteryTransitionWrapper
        self.rollout_tree = ucs

        self.agent_id = None
        self.history = None
        self.local_utility = None
        self.world_predictors = None
        self.month = None
        self.time_idx = None

        self.prediction_depth = int(self.search_depths[-1] + 1)
        self.non_search_depths = sorted(list(set(np.arange(self.prediction_depth)) - set(self.search_depths)))
        self.non_search_intervals = np.append(np.diff(self.search_depths), 1)
        self.action_factorization_by_depth = np.squeeze([i for i in self.non_search_intervals for _ in range(i)])

    def init_agent(self):
        """
        Init the agent. Delete history and init the WorldPredictors object.
        Returns: None

        """
        self.history = np.zeros((24, 28 + 1), dtype=float)
        self.world_predictors = PredictorsWrapper(self.agent_id, method=self.prediction_method)
        self.time_idx = 0
        self.rb_decision_makers["local"].set_action_space(agent_id=self.agent_id,
                                                          action_space=self.action_space[self.agent_id])
        self.rb_decision_makers["global"].set_action_space(agent_id=self.agent_id,
                                                           action_space=self.action_space[self.agent_id])

    def set_action_space(self, agent_id, action_space):
        """
        Set the agent's action space and init the agent.
        Args:
            agent_id: The ID of the agent (int - # building).
            action_space: the action space of the # agent.

        Returns:

        """
        self.action_space[agent_id] = action_space
        self.agent_id = agent_id
        self.init_agent()

    def compute_local_planner_action(self, observation, agent_id):
        """
        Gets observation return an action.
        Args:
            observation: observation of the environment.
            agent_id: The ID of the agent.

        Returns: [action]

        """
        self.record_observation(observation)
        return self.policy(observation, self.action_space[agent_id])

    def compute_local_rb_action(self, observation, agent_id):
        """
        Gets observation return an action.
        Args:
            observation: observation of the environment.
            agent_id: The ID of the agent.

        Returns: [action]
        """
        self.record_observation(observation)
        action = self.rb_decision_makers["local"].compute_local_action(observation, agent_id, self.predict_world())
        self.time_idx = np.mod(self.time_idx + 1, 8670)
        return action

    def compute_global_rb_action(self, observation, agent_id, rest_district_net_cons_prediction,
                                 prev_rest_district_net_cons):
        """
        Gets observation return an action.
        Args:
            observation: observation of the environment.
            agent_id: The ID of the agent.

        Returns: [action]
        """
        self.record_observation(observation)
        action = self.rb_decision_makers["global"].compute_global_action(observation, agent_id, self.predict_world(),
                                                                         rest_district_net_cons_prediction,
                                                                         prev_rest_district_net_cons)
        self.time_idx = np.mod(self.time_idx + 1, 8670)
        return action

    def record_observation(self, observation):
        """
        Saves world's observations in the history attribute.
        Args:
            observation: observation of the environment.

        Returns: None

        """

        self.month = int(observation[0]) - 1
        self.history[-1, 0] = self.time_idx
        indexed_observation = np.append(self.history[-1, 0], observation)
        self.history = np.concatenate((self.history[1:, :], indexed_observation.reshape(1, -1)), axis=0)

    def predict_world(self, consumption_only=False):
        """
        All variables are 1 dimensional of 'prediction_depth' elements.
        Returns: predictions and their derivatives
        """
        return self.world_predictors.predict(history=self.history, prediction_depth=self.prediction_depth,
                                             consumption_only=consumption_only)

    def rollout(self):
        """
        Computes the tree of states and actions in the environment with the utility value and soc for each node.
        Returns: The next action and the utility of its trajectory

        """
        # define an initial MDP state
        first_state = tuple([self.battery.norm_current_soc - self.battery.soc_history[-2],
                             self.battery.norm_current_soc,
                             self.battery.capacity, 0])

        # set the MDP's transition model using the physical model of the agent's battery
        transition_model = self.transition_model_object(self.action_space[self.agent_id], self.d_action,
                                                        self.predict_world(), self.month, self.utility_weighting,
                                                        action_space_list=self.action_space_list,
                                                        action_factorization_by_depth=
                                                        self.action_factorization_by_depth)

        # create a plan using tree search
        discrete_trajectory, _, future_utility = self.rollout_tree(first_state, transition_model, self.max_search_time,
                                                                   self.search_depths)

        # transform the plan's action trajectory from discrete actions the continues actions

        continues_trajectory = list(transition_model.map_action(discrete_trajectory) /
                                    transition_model.action_factorization_by_depth[0:len(discrete_trajectory)])
        return continues_trajectory, future_utility

    def policy(self, observation, action_space):
        """
        Computes the best next action given an observation
        Args:
            observation: observation of the environment.
            action_space: The agent's action space

        Returns: [action]

        """

        # override current battery soc with true soc from the environment
        electrical_storage_soc = observation[22]
        self.battery.override_soc_from_observation(electrical_storage_soc)

        # generate a new plan
        self.plan, future_utility = self.rollout()

        # pick the action
        self.time_idx = np.mod(self.time_idx + 1, 8670)
        action = self.plan.pop(0)
        action = np.array([action], dtype=action_space.dtype)
        assert action_space.contains(action)
        self.battery.update_soc(action[0])
        return action


class BruteForceWeighedAgent:
    def __init__(self, search_depths=(0, 1, 2, 3), d_action: float = 0.3, max_search_time: float = 0.2,
                 plan_steps: int = 1, default_action: float = 0., num_buildings=None,
                 utility_weighting=(1., 1., 1., 1.), action_space_list=None, predictor=None,
                 *args, **kwargs):
        """
        A Rule-based brute force planning agent. The agent uses predictive models of the environment and its own
        behaviour. It uses a physical model of its battery (actions) as an MDP in order to plan ahead in the environment
        using a tree-search Uniform-Cost-Search (UCS) algorithm with a temporal utility approximation as the cost of
        transition between nodes.
        Args:
            d_action: action discretization in (0, 1)
            search_depths: a list of the tree depths (or prediction times) in which the algorithm should search
            *args: Battery physical model parameters
            **kwargs: Battery physical model parameters
        """
        self.action_space = {}
        self.agent_type = 'RB'

        self.action_space_list = action_space_list
        self.d_action = d_action
        self.plan_steps = plan_steps
        self.plan = []
        self.default_action = default_action
        self.utility_weighting = utility_weighting
        self.search_depths = list(search_depths)
        self.max_search_time = max_search_time
        self.predictor = predictor
        self.num_buildings = num_buildings

        self.battery = Battery(*args, **kwargs)
        self.transition_model_object = BatteryTransitionWrapper
        self.rollout_tree = ucs

        self.agent_id = None
        self.history = None
        self.local_utility = None
        self.month = None
        self.time_idx = None

        self.prediction_depth = int(self.search_depths[-1] + 1)
        self.non_search_depths = sorted(list(set(np.arange(self.prediction_depth)) - set(self.search_depths)))
        self.non_search_intervals = np.append(np.diff(self.search_depths), 1)
        self.action_factorization_by_depth = np.squeeze([i for i in self.non_search_intervals for _ in range(i)])

    def init_agent(self):
        """
        Init the agent. Delete history and init the WorldPredictors object.
        Returns: None

        """
        self.history = np.zeros((24, 28+1), dtype=float)
        # if self.predictor is None, it means that the controller did not send a global predictor function to the agent,
        # therefore the agent needs to init its own local predictor.
        if self.predictor is None:
            self.predictor = PredictorsWrapper(self.agent_id)
        self.time_idx = 0

    def set_action_space(self, agent_id, action_space):
        """
        Set the agent's action space and init the agent.
        Args:
            agent_id: The ID of the agent (int - # building).
            action_space: the action space of the # agent.

        Returns:

        """
        self.action_space[agent_id] = action_space
        self.agent_id = agent_id
        self.init_agent()

    def compute_action(self, observation, agent_id):
        """
        Gets observation return an action.
        Args:
            observation: observation of the environment.
            agent_id: The ID of the agent.

        Returns: [action]

        """
        return self.policy(observation, self.action_space[agent_id])

    def record_observation(self, observation):
        """
        Saves world's observations in the history attribute.
        Args:
            observation: observation of the environment.

        Returns: None

        """

        self.month = int(observation[0])
        self.history[-1, 0] = self.time_idx
        indexed_observation = np.append(self.history[-1, 0], observation)
        self.history = np.concatenate((self.history[1:, :], indexed_observation.reshape(1, -1)), axis=0)

    def predict_world(self):
        """
        All variables are 1 dimensional of 'prediction_depth' elements.
        Returns: predictions and their derivatives
        """
        return self.predictor.predict_building(history=self.history, building_id=self.agent_id)

    def rollout(self):
        """
        Computes the tree of states and actions in the environment with the utility value and soc for each node.
        Returns: The next action and the utility of its trajectory

        """
        # define an initial MDP state
        first_state = tuple([self.battery.norm_current_soc - self.battery.soc_history[-2],
                             self.battery.norm_current_soc,
                             self.battery.capacity, 0])

        # set the MDP's transition model using the physical model of the agent's battery
        transition_model = self.transition_model_object(continuous_action_space=self.action_space[self.agent_id],
                                                        d_action=self.d_action,
                                                        predictions=self.predict_world(),
                                                        month=self.month-1,
                                                        agent_id=self.agent_id,
                                                        num_buildings=self.num_buildings,
                                                        utility_weighting=self.utility_weighting,
                                                        action_space_list=self.action_space_list,
                                                        action_factorization_by_depth=
                                                        self.action_factorization_by_depth)

        # create a plan using tree search
        discrete_trajectory, _, future_utility = self.rollout_tree(first_state, transition_model, self.max_search_time,
                                                                   self.search_depths)

        # transform the plan's action trajectory from discrete actions the continues actions

        continues_trajectory = list(transition_model.map_action(discrete_trajectory) /
                                    transition_model.action_factorization_by_depth[0:len(discrete_trajectory)])
        return continues_trajectory[:self.plan_steps], future_utility

    def policy(self, observation, action_space):
        """
        Computes the best next action given an observation
        Args:
            observation: observation of the environment.
            action_space: The agent's action space

        Returns: [action]

        """
        self.record_observation(observation)

        # # override current battery soc with true soc from the environment
        # electrical_storage_soc = observation[22]
        # self.battery.override_soc_from_observation(electrical_storage_soc)

        # generate a new plan
        self.plan, future_utility = self.rollout()

        # pick the action
        self.time_idx = np.mod(self.time_idx + 1, 8670)
        action = self.plan[0]
        action = np.array([action], dtype=action_space.dtype)
        assert action_space.contains(action)
        self.battery.update_soc(action[0])
        return action


