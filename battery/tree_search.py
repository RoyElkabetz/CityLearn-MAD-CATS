import numpy as np
import time
from battery.battery_model import BatteryTransitionWrapper


class Node:
    """
    A general node in a tree search class
    """
    def __init__(self, frame=None, state=None, action=None, reward=None, done=None, parent_state=None,
                 g_value=None, h_value=None, weight=None, limit=None, depth=None,
                 time_idx=None, transition_model_params=None):
        """

        Args:
            frame: str for printing of the states as a movie
            state: the nodes state
            action: the action led to the current state
            reward: the reward given on the transition
            done: True if the node is a goal node, otherwise False
            parent_state: the state of the parent node
            g_value: the cost of transitioning from parent_state to state
            h_value: an heuristic function to compute the cost from state to the end
            weight: a weight between g_value and h_value (in [0, 1])
            limit:
            depth: the node's depth
            time_idx: the environment time index / depth
            transition_model_params: environment's parameters (battery parameters)
        """
        self.frame = frame
        self.state = state
        self.action = action
        self.reward = reward
        self.done = done
        self.parent_state = parent_state
        self.g_value = g_value
        self.h_value = h_value
        if weight is not None:
            assert 0 <= weight and weight <= 1
        self.weight = weight
        self.limit = limit
        self.transition_model_params = transition_model_params
        self.transition_model_ = BatteryTransitionWrapper(**transition_model_params)
        self.transition_model_.init_state(self.state)
        self.depth = state[3]
        self.time_idx = time_idx

        # For weight=1 we get the Greedy-BFS algorithm, and for weight=0 we get the UCS algorithm
        if self.weight is not None:
            self.f_value = (1 - self.weight) * self.g_value + self.weight * self.h_value if (
                    self.g_value is not None and self.h_value is not None) else None
        else:
            self.f_value = self.g_value

    def __str__(self):
        return "frame:\n{}\nstate: {}\naction: {}\nreward: {}\ndone: {}" \
               "\nprevious state {}\ng(n): {}\nh(n): {}\nf(n): {}\nw: {}\ndepth: {}\n" \
               "".format(self.frame, self.state, self.action, self.reward, self.done,
                         self.parent_state, self.g_value, self.h_value, self.f_value, self.weight,
                         self.depth)

    def get_child(self, action, heuristic_function=None):
        """
        Given an action computes the child of the parent node
        Args:
            action: MDP action
            heuristic_function: a heuristic function to perform on the new state

        Returns: child node (Node class)

        """
        self.transition_model_.init_state(self.state)
        new_state, inst_u, done, info = self.transition_model_.step(action)
        next_g = inst_u

        child_node = Node(frame=None,
                          state=new_state,
                          action=action,
                          reward=inst_u,
                          done=done,
                          parent_state=self.state,
                          g_value=self.g_value + next_g if self.g_value is not None else None,
                          h_value=heuristic_function(new_state) if heuristic_function is not None else None,
                          weight=self.weight,
                          limit=self.limit - 1 if self.limit is not None else None,
                          depth=self.depth + 1 if self.depth is not None else None,
                          time_idx=self.time_idx + 1 if self.time_idx is not None else None,
                          transition_model_params=self.transition_model_params
                          )
        return child_node


class Frontier:
    def __init__(self, order_strategy='min_f', use_focal=False, epsilon=None):
        """
        A Frontier object that holds all nodes which are currently in the frontier of the search
        Args:
            order_strategy: the queue order method [FIFO, LIFO, min_f]
            use_focal: a focal queue for local search (default = not used)
            epsilon: the focal's epsilon
        """
        self.nodes = []
        self.nodes_values = []
        self.nodes_states = []
        self.epsilon = epsilon
        self.use_focal = use_focal

        assert order_strategy in ['LIFO', 'FIFO', 'min_f']
        self.order_strategy = order_strategy

        if use_focal:
            self.focal = []
            assert epsilon is not None and epsilon == np.abs(epsilon)

    def add_node(self, node):
        """
        Add a new node to frontier.
        Args:
            node: the node to be added to queue

        Returns: None

        """
        self.nodes.append(node)
        self.nodes_values.append(node.f_value)
        self.nodes_states.append(node.state)

    def pop_next(self):
        """
        Pop the next node to search.
        Returns: next node

        """
        if self.order_strategy == 'min_f':
            idx = np.argmin(self.nodes_values)
        if self.order_strategy == 'LIFO':
            idx = -1
        if self.order_strategy == 'FIFO':
            idx = 0
        next_node = self.nodes.pop(idx)
        _ = self.nodes_values.pop(idx)
        _ = self.nodes_states.pop(idx)
        return next_node

    def pop_node(self, node):
        """
        Pops and returns a specific node from frontier
        Args:
            node: the node to be popped

        Returns: the node

        """
        for i, n in enumerate(self.nodes):
            if n.state == node.state:
                _ = self.nodes_values.pop(i)
                _ = self.nodes_states.pop(i)
                next_node = self.nodes.pop(i)
                return next_node
        return None

    def update_focal(self):
        """
        Computes the indices of nodes in focal.
        Returns: None
        """
        f_min = np.min(self.nodes_values)
        if self.epsilon == 0:
            self.focal = [np.argmin(self.nodes_values)]
        else:
            idx_bool_focal = (np.array(self.nodes_values) <= (f_min * (1 + self.epsilon)))
            idx_focal = np.arange(len(self.nodes))
            self.focal = idx_focal[idx_bool_focal]

    def pop_focal(self):
        """
        Pop a node from focal.
        Returns: the next node from focal.

        """
        idx = np.random.choice(self.focal)
        next_node = self.nodes.pop(idx)
        _ = self.nodes_values.pop(idx)
        _ = self.nodes_states.pop(idx)
        return next_node

    def __len__(self):
        return len(self.nodes)

    def __contains__(self, node):
        return node.state in self.nodes_states

    def __str__(self):
        """
        Print function
        Returns: string to print

        """
        print_list = []
        if self.use_focal:
            print_focal_list = []
            for i, n in enumerate(self.nodes):
                print_list.append((self.nodes_states[i], self.nodes_values[i]))
                if i in self.focal:
                    print_focal_list.append((self.nodes_states[i], self.nodes_values[i]))
            return 'Frontier (state, value):\n{}\nFocal (state, value):\n{}\n'.format(print_list, print_focal_list)
        else:
            for i, n in enumerate(self.nodes):
                print_list.append((self.nodes_states[i], self.nodes_values[i]))
            return 'Frontier (state, value):\n{}'.format(print_list)


class Explored:
    def __init__(self):
        """
        An Explored object (hash table) that contains all the nodes already been searched.
        """
        self.nodes = {}

    def add_node(self, node):
        """
        Add node to explored dict.
        Args:
            node: the node to add

        Returns: None

        """
        self.nodes[node.state] = node

    def pop_node(self, node):
        """
        Pops and returns the node with node.state=state from explored
        Args:
            node: the node to pop

        Returns: the node

        """
        node_explored = self.nodes[node.state]
        self.nodes.pop(node.state)
        return node_explored

    def get_route(self, goal_node, frontier=None):
        """
        Extracts the route found by the algorithm given the goal node
        Args:
            goal_node: the end node of the route
            frontier: the frontier queue continued the nodes not been searched yet.

        Returns: the route to the goal node

        """
        if frontier is not None:
            for node in frontier.nodes:
                if node.state not in self.nodes:
                    self.nodes[node.state] = node
        frames = []
        actions_trajectory = []
        rewards = 0

        frames.append({'frame': goal_node.frame, 'state': goal_node.state, 'action': goal_node.action,
                       'reward': goal_node.reward})
        rewards += goal_node.reward
        actions_trajectory.append(goal_node.action)
        node = goal_node

        while node.action is not None:
            #             print('p: ', node.parent_state, ' s: ', node.state)
            node = self.nodes[node.parent_state]
            actions_trajectory.append(node.action)
            frames.append({'frame': node.frame, 'state': node.state, 'action': node.action, 'reward': node.reward})
            rewards += node.reward if node.reward is not None else 0
        actions_trajectory = actions_trajectory[::-1]
        return actions_trajectory[1:], frames[::-1], rewards

    def __len__(self):
        return len(self.nodes)

    def __contains__(self, node):
        return node.state in self.nodes.keys()

    def __str__(self):
        return 'Explored (states):\n{}'.format(list(self.nodes.keys()))


def get_info(node, explored, frontier):
    """
    A debug function for printing the search state to screen
    Args:
        node: current node
        explored: the explored object
        frontier: the frontier object

    Returns: None

    """
    states_seen = [node.state] + list(explored.nodes.keys()) + frontier.nodes_states
    different_states_seen = set(states_seen)

    text = "===============================================================\n" \
           "Current node: {}\n" \
           "Number of states seen: {}\n" \
           "Number of different states seen: {}\n" \
           "Number of states explored: {}\n" \
           "Number of different states explored: {}\n" \
           "Number of states in frontier {}\n" \
           "Number of different states in frontier {}\n" \
           "===============================================================".format(
                    node, len(states_seen), len(different_states_seen), len(explored.nodes),
                    len(set(list(explored.nodes.keys()))), len(frontier.nodes_states),
                    len(set(frontier.nodes_states)))
    print(text)


def ucs(first_state, transition_model, max_time_sec=0.2, search_action_depths: list = None):
    """
    A Uniform Cost Search algorithm implementation with a few variations
    Args:
        first_state: The first state from which to start the search.
        transition_model: The MDP transition model from which one can sample transitions.
        max_time_sec: The maximal time allowed for search. If a terminal state is not reached before this time, the
                      algorithm returns the trajectory to the current node being searched.
        search_action_depths: A list of depths in which the algorithm will use the transition model and will search
                              all the actions (means; all nodes in depths which are not on this list will have only a
                              single child).
        non_search_action_type: The type of action to take in search depths from fixed_action_depths. Should be one of the
                                next options:
                               'default': take a pre-defined constant action from 'fixed_action' variable.
                               'descending_integral': TODO create this feature
                               'ascending_integral': TODO create this feature
                               'constant_integral': take the action at the first node divided by the interval length at
                                                    each depth in the interval.
        non_search_action: Being used only when fixed_action_type=='default', then takes a constant pre-defined action
                           at all depth in the interval.

    Returns: the action trajectory (list) with the smallest path cost and the path value (float)

    """
    start_t = time.process_time()
    node = Node(state=first_state,
                transition_model_params={"continuous_action_space": transition_model.continuous_action_space,
                                         "d_action": transition_model.d_action,
                                         "predictions": transition_model.predictions,
                                         "month": transition_model.month,
                                         "utility_weighting": transition_model.utility_weighting,
                                         "action_space_list": transition_model.action_space_list,
                                         "num_buildings": transition_model.num_buildings,
                                         "agent_id": transition_model.agent_id,
                                         "action_factorization_by_depth":
                                             transition_model.action_factorization_by_depth},
                depth=0, g_value=0)
    frontier = Frontier(order_strategy='min_f')
    frontier.add_node(node)
    explored = Explored()

    while len(frontier) > 0:
        node = frontier.pop_next()

        ############## DEBUG ##############
        # uncomment to print debug info
        # get_info(node, explored, frontier)
        ###################################

        # if max_time_sec has passed, return route to current node
        if time.process_time() - start_t > max_time_sec:
            # raise TimeoutError(f"agent exceeds search time while searching in depth {node.depth}")
            return explored.get_route(node)

        if node.depth > transition_model.predictions_last_idx:
            return explored.get_route(node)

        # add node to explored set
        explored.add_node(node)

        # if the node depth is not in search_action_depths, take the parent node action, else search
        if node.depth not in search_action_depths:
            non_search_action = node.action
            child = node.get_child(non_search_action)
            if (child not in explored) and (child not in frontier):
                if child.done:
                    return explored.get_route(child)
                else:
                    frontier.add_node(child)
        else:
            for action in range(transition_model.action_space.n):
                child = node.get_child(action)
                if (child not in explored) and (child not in frontier):
                    if child.done:
                        return explored.get_route(child)
                    else:
                        frontier.add_node(child)
    return ([], None)

