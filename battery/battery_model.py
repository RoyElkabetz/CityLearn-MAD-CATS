import numpy as np
import matplotlib.pyplot as plt
from gym.spaces.discrete import Discrete
from rewards.agent_inst_u import get_inst_u
from rewards.agent_inst_u_with_last import get_inst_u_total
from gym.spaces.box import Box


class Battery:
    def __init__(self, initial_soc: float = 0, initial_capacity: float = 6.4, nominal_power: float = 5.0,
                 time_window: float = 1, max_power_in_curve=None,
                 max_power_out_curve=None, capacity_loss_coefficient: float = 1e-5,
                 power_efficiency_curve=None, efficiency: float = 0.9):
        """
        An Battery object implementation in power and energy units of kW and kWh
        :param initial_soc: Initial SoC (State of Charge) of the battery in units of energy of kWh.
        :param initial_capacity: Initial capacity of the battery in units of energy of kWh.
        :param nominal_power: The battery's nominal in/out power in units of power of kW.
        :param time_window: Time discretization window of the battery in units of hours, used for transforming
        power to energy and vice versa.
        :param max_power_in_curve: A 2D array of (normalized_soc, max_power_in) points,
        i.e. [[0., 0.6, 1.0], [1.0, 0.95, 0.43]], normalized_soc is unit-less in [0, 1].
        :param max_power_out_curve: A 2D array of (normalized_soc, max_power_out) points,
        i.e. [[0., 0.6, 1.0], [1.0, 0.95, 0.43]], normalized_soc is unit-less in [0, 1].
        :param capacity_loss_coefficient: Capacity loss coefficient used for capacity degradation in each soc update.
        :param power_efficiency_curve: A 2D array of (normalized_power, efficiency) points,
        i.e. [[0., 0.2, 0.6, 1.], [0.84, 0.87, 0.95, 0.91]], normalized_power is unit-less in [0, 1].
        :param efficiency: Power efficiency (used only if power_efficiency_curve is None)
        """
        assert 0 < initial_capacity < np.inf, f"The battery's capacity should be a positive finite number, " \
                                              f"instead got {initial_capacity}."
        assert 0 <= initial_soc <= initial_capacity, f"The battery's initial soc should be a positive number " \
                                                     f"bounded by the capacity, instead got {initial_soc}."
        assert 0 < nominal_power < np.inf, f"The battery's nominal power should be a positive finite number, " \
                                           f"instead got {nominal_power}."
        assert 0 < time_window < np.inf, f"The battery's time window should be a positive finite number, " \
                                         f"instead got {time_window}."
        assert 0 <= capacity_loss_coefficient < 1, f"The battery's capacity loss coefficient should be a positive " \
                                                   f"number in the [0, 1) interval, instead " \
                                                   f"got {capacity_loss_coefficient}."
        assert 0 < efficiency <= 1, f"The battery's efficiency should be a positive number in the interval (0, 1], " \
                                    f"instead got {efficiency}."

        self.initial_soc = initial_soc
        self.current_soc = initial_soc
        self.previous_soc = initial_soc
        self.norm_initial_soc = initial_soc / initial_capacity
        self.norm_current_soc = initial_soc / initial_capacity
        self.norm_previous_soc = initial_soc / initial_capacity
        self.initial_capacity = initial_capacity
        self.capacity = initial_capacity
        self.capacity_loss_coefficient = capacity_loss_coefficient
        self.nominal_power = nominal_power
        self.time_window = time_window
        self.max_power_in_curve = max_power_in_curve
        self.max_power_out_curve = max_power_out_curve
        self.power_efficiency_curve = power_efficiency_curve
        self.efficiency = efficiency
        self.efficiency_history = [efficiency]
        self.soc_history = [initial_soc]
        self.idial_soc_history = [initial_soc]
        self.norm_soc_history = [initial_soc / initial_capacity]
        self.capacity_history = [initial_capacity]

        if self.power_efficiency_curve is None:
            self.power_efficiency_curve = [[0., 0.3, 0.7, 0.8, 1.], [0.83, 0.83, 0.9, 0.9, 0.85]]
        if self.max_power_out_curve is None:
            self.max_power_out_curve = [[0., 0.8, 1.0], [1.0, 1.0, 0.2]]
            self.max_power_in_curve = [[0., 0.8, 1.0], [1.0, 1.0, 0.2]]

    def __str__(self):
        text = "== Battery state: ==\n" \
               "Normalized current soc: {}\n" \
               "Normalized previous soc: {}\n" \
               "Current soc: {}\n" \
               "Previous soc: {}\n" \
               "Capacity: {}\n" \
               "Capacity loss coefficient: {}\n" \
               "Nominal power: {}\n" \
               "====================\n".format(self.norm_current_soc, self.norm_previous_soc, self.current_soc,
                                             self.previous_soc, self.capacity, self.capacity_loss_coefficient,
                                             self.nominal_power)
        return text

    def hard_reset(self):
        """
        Resents the battery values to initial values and also erase history
        """
        self.norm_soc_history = []
        self.capacity_history = []
        self.idial_soc_history = []
        self.efficiency_history = []
        self.reset()

    def reset(self):
        """
        Resents the battery values to initial values while maintaining history
        """
        self.current_soc = self.initial_soc
        self.previous_soc = self.initial_soc
        self.soc_history.append(self.initial_soc)
        self.norm_current_soc = self.norm_initial_soc
        self.norm_previous_soc = self.norm_initial_soc
        self.norm_soc_history.append(self.norm_initial_soc)
        self.capacity = self.initial_capacity
        self.capacity_history.append(self.initial_capacity)
        self.idial_soc_history.append(self.initial_soc)

    @staticmethod
    def sample_curve(curve, x) -> float:
        """
        Given a curve described by a finite number of points, compute the linear interpolation of a given new point.
        :param curve: A 2D array of (x, f(x)) points, i.e. [[0., 0.6, 1.0], [1.0, 0.95, 0.43]].
        :param x: A new x point to find its y = f_{interpolation}(x)
        :return: y
        """
        idx = max(0, np.argmin(np.abs(np.array(curve[0]) - x)) - 1)
        slope = (curve[1][idx+1] - curve[1][idx]) / (curve[0][idx+1] - curve[0][idx])
        bias = curve[1][idx] - slope * curve[0][idx]
        return slope * x + bias

    @property
    def get_max_power_in(self) -> float:
        """
        Computes the battery's maximum power input in units of kW.
        :return: max_power_in
        """
        if self.max_power_in_curve is not None:
            return self.sample_curve(self.max_power_in_curve, self.current_soc / self.capacity) * self.nominal_power
        else:
            return self.nominal_power

    @property
    def get_max_power_out(self) -> float:
        """
        Computes the battery's maximum power output in units of kW.
        :return: max_power_out
        """
        if self.max_power_out_curve is not None:
            return self.sample_curve(self.max_power_out_curve, self.current_soc / self.capacity) * self.nominal_power
        else:
            return self.nominal_power

    @property
    def max_energy_in(self) -> float:
        """
        Computes the battery's maximum energy input.
        :return: max_energy_in = max_power_in * time_window
        """
        return self.get_max_power_in * self.time_window

    @property
    def max_energy_out(self) -> float:
        """
        Computes the battery's maximum energy output.
        :return: max_energy_out = max_power_out * time_window
        """
        return self.get_max_power_out * self.time_window

    def efficiency_per_energy(self, energy) -> float:
        """
        Computes the battery's efficiency for the given energy charge/discharge. This implementation assumes that
        efficiency is symmetric with respect to the energy sign (charge vs discharge).
        :param energy: Input/Output energy in kWh
        :return: efficiency
        """
        avg_power = np.abs(energy) / self.time_window
        return self.get_power_efficiency(avg_power)

    def get_power_efficiency(self, power) -> float:
        """
        Computes the power efficiency of the battery.
        :param power: Input power in kWh.
        :return: efficiency
        """
        if self.power_efficiency_curve is not None:
            return self.sample_curve(self.power_efficiency_curve, power / self.nominal_power) ** 0.5
        else:
            return self.efficiency

    def capacity_degrade(self) -> None:
        """
        Computes a single step energy dependent capacity degradation (symmetric in energy sign).
        :return: None
        """
        # self.capacity_loss_coefficient*self.capacity_history[0]*np.abs(self.energy_balance[-1])/(2*self.capacity)
        self.capacity -= self.capacity_loss_coefficient * self.initial_capacity * \
                         np.abs(self.idial_soc_history[-1] - self.idial_soc_history[-2]) / (2 * self.capacity)
        # self.capacity -= self.capacity_loss_coefficient * abs(self.current_soc - self.previous_soc) * \
        #                  self.capacity / self.initial_capacity
        self.capacity_history.append(self.capacity)

    def charge(self, energy, efficiency) -> None:
        """
        Charge the battery, update the current soc.
        :param energy: Input/Output energy in kWh
        :param efficiency: The efficiency of operation (energy dependent)
        :return: None
        """
        self.previous_soc = self.current_soc
        self.norm_previous_soc = self.norm_current_soc
        self.current_soc = min(self.current_soc + energy * efficiency, self.capacity) if energy >= 0 \
            else max(self.current_soc + energy / efficiency, 0)
        self.norm_current_soc = self.current_soc / self.capacity

        # save history
        self.efficiency_history.append(efficiency)
        self.soc_history.append(self.current_soc)
        self.idial_soc_history.append((self.current_soc - self.previous_soc) / efficiency if energy >= 0 else
                                      (self.current_soc - self.previous_soc) * efficiency)
        self.norm_soc_history.append(self.norm_current_soc)

    def simulate_charge_discharge(self, energy) -> (float, float, float, float):

        # compute the actual energy being charged (in limits) and the action needed
        action, energy = self.energy_to_action(energy)

        # get current efficiency
        efficiency = self.efficiency_per_energy(energy)

        # simulate the next soc
        next_soc = min(self.current_soc + energy * efficiency, self.capacity) if energy >= 0 \
            else max(self.current_soc + energy / efficiency, 0)

        return action, next_soc, energy, efficiency

    def energy_to_action(self, energy) -> (float, float):
        """
        Given an energy to charge/discharge, returns the needed action in [0, 1] and actual respective energy
        """
        # limit energy to battery's capacity
        energy = min(energy, self.capacity - self.current_soc) if energy >= 0 else -min(-energy, self.current_soc)

        # verify energy in smaller than nominal
        energy = min(energy, self.max_energy_in) if energy >= 0 else max(energy, -self.max_energy_out)

        # compute the action
        action = energy / self.capacity
        return action, energy

    def update_soc(self, action) -> None:
        """
        Updates the battery's SoC (State of Charge) given an action from the agent.
        :param action: The agent's action in [0, 1]
        :return: None
        """
        # compute energy from action
        energy = action * self.capacity

        # limit energy to battery's capacity
        energy = min(energy, self.capacity - self.current_soc) if energy >= 0 else -min(-energy, self.current_soc)

        # verify energy in smaller than nominal
        energy = min(energy, self.max_energy_in) if energy >= 0 else max(energy, -self.max_energy_out)

        # get current efficiency
        efficiency = self.efficiency_per_energy(energy)

        # charge battery
        self.charge(energy, efficiency)

        # degrade capacity
        self.capacity_degrade()


class BatteryTransitionWrapper(Battery):
    def __init__(self, continuous_action_space, d_action: float, predictions: dict,
                 month: int, utility_weighting=(1., 1., 1., 1.), action_space_list=None,
                 action_factorization_by_depth=None, agent_id=None, num_buildings=None, *args, **kwargs):
        """
        An MDP trees search wrapper for the battery's physical model (the MDP transition function).
        Args:
            continuous_action_space: the agent's continuous action space
            d_action: the action discretization in (0, 1)
            predictions: 24 hours predictions of the agent's and environment's behaviour
            month: the current month in the simulation (needed for utility computation)
            *args: Battery physical model parameters
            **kwargs: Battery physical model parameters
        """
        super().__init__(*args, **kwargs)
        if isinstance(continuous_action_space, dict):
            continuous_action_space = Box(low=continuous_action_space["low"],
                                          high=continuous_action_space["high"],
                                          shape=continuous_action_space["shape"],
                                          dtype=float)
        self.continuous_action_space = continuous_action_space
        self.d_action = d_action
        self.action_space_list = action_space_list
        self.predictions = predictions
        self.month = month
        self.agent_id = agent_id
        self.num_buildings = num_buildings
        self.utility_weighting = utility_weighting
        self.action_factorization_by_depth = action_factorization_by_depth

        self.predictions_last_idx = len(self.predictions["pricing"]) - 1
        self.action_space = None
        self.curr_state = None
        self.next_state = None

        self.init_action_space()

    def init_action_space(self):
        """
        Create a Discrete action space from a Box continuous action space and an action discretization step.
        Returns: None

        """
        if self.action_space_list is None:
            n = int((self.continuous_action_space.high - self.continuous_action_space.low) / self.d_action) + 1
        else:
            n = len(self.action_space_list)

        self.action_space = Discrete(n)

    def init_state(self, state):
        """
        Initialize the transition model with the current initial state of the MDP.
        Args:
            state: the current state of the MDP

        Returns: None

        """
        # unpack the state
        self.curr_state = state
        _, soc, capacity, _ = self.curr_state

        # reset the battery
        self.reset_battery_model(soc, capacity)

    def reset_battery_model(self, soc, capacity):
        """
        Delete the battery's history and reset the battery to a specific soc and capacity.
        Args:
            soc (float): initial soc (battery's state of charge \in [0, 1])
            capacity (float): battery's initial capacity

        Returns: None

        """
        if soc > 1:
            soc = 1
        assert 0 <= soc <= 1, f"Soc should be between [0,1], instead got {soc}."
        assert 0 < capacity, f"Capacity should be larger than 0 instead got, {capacity}."

        # set the initial soc in units of energy
        self.initial_soc = soc * capacity

        # set the capacity
        self.initial_capacity = capacity

        # set the unit-less initial soc, in [0, 1]
        self.norm_initial_soc = soc

        # reset the battery
        self.hard_reset()

    def map_action(self, action):
        """
        Maps an action index in discrete space to a continues action of the battery.
        ---
        Actions:
        idx --> continues action
        if n is odd:
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] --> da*[0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5]
        ---
        if n is even:
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] --> da*[-1, 1, -2, 2, -3, 3, -4, 4, -5, 5]
        ---

        Args:
            action (int): index of action in discrete action space

        Returns (float): an action in the original continues action space

        """
        if self.action_space_list is None:
            if np.mod(self.action_space.n, 2) == 1:
                con_action = np.ceil(np.array(action) / 2) * self.d_action * np.power(-1, np.array(action))
            else:
                con_action = np.ceil((np.array(action) + 1) / 2) * self.d_action * np.power(-1, np.array(action) + 1)
        else:
            con_action = np.array(self.action_space_list)[np.array(action)]

        return con_action

    def get_utility(self):
        """
        Compute the instantaneous utility given current environment predictions.
        Returns (float): instantaneous utility

        """
        # the hour of environment prediction in the day {0, 1, 2, \dots, 23}.
        hour = self.curr_state[3]   # the 0^th element of the prediction array is 1 hour prediction of the data

        # unpack the battery's soc and its derivative from the state
        soc = self.next_state[1] * self.next_state[2]
        curr_dsoc = self.next_state[0] * self.next_state[2]
        prev_dsoc = self.curr_state[0] * self.curr_state[2]
        ddsoc = curr_dsoc - prev_dsoc

        # compute the utility

        if "district_net_consumption" in self.predictions.keys():
            rest_net_consumption = np.delete(self.predictions["district_net_consumption"], self.agent_id, axis=0)
            rest_d_net_consumption = np.delete(self.predictions["district_d_net_consumption"], self.agent_id, axis=0)
            inst_u = get_inst_u_total(net_cons=self.predictions["building_net_consumption"][hour] + curr_dsoc,
                                      consumption_diff=self.predictions["building_d_net_consumption"][hour] + ddsoc,
                                      net_cons_rest=rest_net_consumption[:, hour],
                                      consumption_diff_rest=rest_d_net_consumption[:, hour],
                                      prc=self.predictions["pricing"][hour],
                                      carbon_ints=self.predictions["carbon"][hour],
                                      month=self.month,
                                      building_number=self.agent_id,
                                      out_of=self.num_buildings,
                                      weighting=self.utility_weighting)
        else:
            inst_u = get_inst_u(self.predictions["building_net_consumption"][hour] + curr_dsoc,
                                self.predictions["building_d_net_consumption"][hour] + ddsoc,
                                self.predictions["pricing"][hour],
                                self.predictions["carbon"][hour],
                                self.month,
                                weighting=self.utility_weighting)

        return inst_u

    def step(self, action):
        """
        Step forward the MDP transition model (battery model) with the given action.
        Args:
            action (int): index of action in discrete action space

        Returns: next_state, reward, info, done

        """
        depth = self.curr_state[3] + 1
        action = self.map_action(action)
        if self.action_factorization_by_depth is not None:
            action = action / self.action_factorization_by_depth[depth - 1]

        self.update_soc(action)
        self.next_state = tuple([self.norm_current_soc - self.norm_previous_soc,
                                 self.norm_current_soc,
                                 self.capacity,
                                 depth])
        inst_u = self.get_utility()
        return self.next_state, inst_u, None, False




def main():
    power_curve = [[0., 0.6, 1.0], [1.0, 0.95, 0.43]]
    efficiency_curve = [[0., 0.2, 0.6, 1.], [0.84, 0.87, 0.95, 0.91]]
    battery = Battery(initial_soc=0,
                      initial_capacity=4.5,
                      nominal_power=2.1,
                      time_window=1,
                      max_power_in_curve=power_curve,
                      max_power_out_curve=power_curve,
                      capacity_loss_coefficient=2e-5,
                      power_efficiency_curve=efficiency_curve,
                      efficiency=0.96)
    t_max = 20
    actions = 2 * (np.round(np.random.rand(t_max), 2) - 0.5)
    # actions = [1, 1, 1, 1, -1, 0.5, 0.2, -0.5, -0.2, 0.9]
    for _, a in enumerate(actions):
        battery.update_soc(a)

    t_max = len(actions)
    fig, axes = plt.subplots(4, 1, sharex='all')
    axes[0].plot(range(t_max), actions, 'o', label='actions')
    axes[0].set_ylabel('action')
    axes[0].set_ylim([min(-1.1, min(actions)), max(1.1, max(actions))])
    axes[0].legend()
    axes[0].grid()
    axes[1].plot(range(0, t_max + 1), battery.soc_history, '-o', label="soc")
    axes[1].set_ylabel('soc [kWh]')
    axes[1].set_ylim([-0.1, battery.initial_capacity])
    axes[1].legend()
    axes[1].grid()
    axes[2].plot(range(1, t_max + 1), np.diff(battery.soc_history), '-x', label="$\delta$soc")
    axes[2].set_ylabel('soc diff [kWh]')
    axes[2].set_ylim([-battery.initial_capacity - 0.1, battery.initial_capacity + 0.1])
    axes[2].legend()
    axes[2].grid()
    axes[3].plot(range(t_max + 1), battery.capacity_history, '-x', label="capacity")
    axes[3].set_ylabel('capacity [kWh]')
    axes[3].legend()
    axes[3].set_xlabel('time step [h]')
    axes[3].set_xticks(range(t_max + 1))
    axes[3].set_xticklabels([str(t) for t in range(t_max + 1)])
    axes[3].grid()
    plt.show()


if __name__ == '__main__':
    main()
