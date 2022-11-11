import numpy as np
import matplotlib.pyplot as plt


class Battery:
    def __init__(self, initial_soc: float = 0, initial_capacity: float = 6.4, nominal_power: float = 5.0,
                 capacity_power_curve=None, loss_coefficient: float = 1e-5, power_efficiency_curve=None,
                 efficiency: float = 0.9, efficiency_scaling: float = 0.5):
        """
        A Battery object implementation in power and energy units of kW and kWh
        :param initial_soc: Initial SoC (State of Charge) of the battery in units of energy of kWh.
        :param initial_capacity: Initial capacity of the battery in units of energy of kWh.
        :param nominal_power: The battery's nominal in/out power in units of power of kW.
        :param capacity_power_curve: A 2D array of (normalized_soc, max_power_in) points,
        i.e. [[0., 0.6, 1.0], [1.0, 0.95, 0.43]], normalized_soc is unit-less in [0, 1].
        :param loss_coefficient: Capacity loss coefficient used for capacity degradation in each soc update.
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
        assert 0 <= loss_coefficient < 1, f"The battery's capacity loss coefficient should be a positive " \
                                          f"number in the [0, 1) interval, instead " \
                                          f"got {loss_coefficient}."
        assert 0 < efficiency <= 1, f"The battery's efficiency should be a positive number in the interval (0, 1], " \
                                    f"instead got {efficiency}."

        assert 0 < efficiency_scaling <= 1, f"The battery's efficiency  scaling should be a positive number, " \
                                            f"instead got {efficiency_scaling}."

        self.initial_soc = initial_soc
        self.norm_current_soc = initial_soc / initial_capacity
        self.initial_capacity = initial_capacity
        self.capacity = initial_capacity
        self.loss_coefficient = loss_coefficient
        self.nominal_power = nominal_power
        self.capacity_power_curve = capacity_power_curve
        self.power_efficiency_curve = power_efficiency_curve
        self.efficiency = efficiency
        self.efficiency_scaling = efficiency_scaling
        self.efficiency_history = [efficiency]
        self.soc_history = [initial_soc, initial_soc]
        self.capacity_history = [initial_capacity]
        self.energy_balance = [0.0]

        if self.power_efficiency_curve is None:
            self.power_efficiency_curve = [[0., 0.3, 0.7, 0.8, 1.], [0.83, 0.83, 0.9, 0.9, 0.85]]
        if self.capacity_power_curve is None:
            self.capacity_power_curve = [[0., 0.8, 1.0], [1.0, 1.0, 0.2]]
        if efficiency_scaling is None:
            self.efficiency_scaling = 0.5

    def __str__(self):
        text = "== Battery state: ==\n" \
               "Normalized current SoC: {}\n" \
               "Initial SoC: {}\n" \
               "Capacity: {}\n" \
               "Capacity loss coefficient: {}\n" \
               "Nominal power: {}\n" \
               "====================\n".format(self.norm_current_soc, self.initial_soc, self.capacity,
                                               self.loss_coefficient, self.nominal_power)
        return text

    def hard_reset(self):
        """
        Resents the battery values to initial values and also erase history
        """
        self.efficiency_history = []
        self.soc_history = []
        self.capacity_history = []
        self.energy_balance = []
        self.reset()

    def reset(self):
        """
        Resents the battery values to initial values while maintaining history
        """
        # self.current_soc = self.initial_soc
        # self.previous_soc = self.initial_soc
        self.soc_history.append(self.initial_soc)
        self.norm_current_soc = self.initial_soc / self.initial_capacity
        # self.norm_previous_soc = self.norm_initial_soc
        # self.norm_soc_history.append(self.norm_initial_soc)
        self.capacity = self.initial_capacity
        self.capacity_history.append(self.initial_capacity)
        self.energy_balance = [0.0]
        self.efficiency_history = [self.efficiency]

    def override_soc_from_observation(self, soc: float) -> None:
        """
        Overides the battery's current soc with a new value.
        :param soc: The new value of the battery's current soc.
        """
        self.initial_soc = soc * self.capacity
        self.initial_capacity = self.capacity
        self.norm_current_soc = soc

        self.soc_history[-1] = self.initial_soc
        self.capacity_history[-1] = self.initial_capacity
        self.energy_balance[-1] = self.set_energy_balance()
        # self.reset()

    @staticmethod
    def sample_curve(curve, x) -> float:
        """
        Given a curve described by a finite number of points, compute the linear interpolation of a given new point.
        :param curve: A 2D array of (x, f(x)) points, i.e. [[0., 0.6, 1.0], [1.0, 0.95, 0.43]].
        :param x: A new x point to find its y = f_{interpolation}(x)
        :return: y
        """
        # idx = max(0, np.argmin(np.abs(np.array(curve[0]) - x)) - 1)
        # slope = (curve[1][idx+1] - curve[1][idx]) / (curve[0][idx+1] - curve[0][idx])
        # bias = curve[1][idx] - slope * curve[0][idx]
        # return slope * x + bias

        idx = max(0, np.argmax(x <= np.array(curve[0])) - 1)
        c0 = curve[0][idx]
        c1 = curve[1][idx]
        return c1 + (curve[1][idx + 1] - c1) * (x - c0) / (curve[0][idx + 1] - c0)

    @property
    def soc_init(self) -> float:
        r"""Latest state of charge after accounting for standby hourly losses."""

        return self.soc_history[-1] * (1 - self.loss_coefficient)

    def get_max_input_power(self) -> float:
        """
        Computes the battery's maximum power input in units of kW.
        :return: max_power_in
        """
        if self.capacity_power_curve is not None:
            return self.nominal_power * self.sample_curve(self.capacity_power_curve, self.soc_init / self.capacity)
        else:
            return self.nominal_power

    def get_max_output_power(self) -> float:
        """
        Computes the battery's maximum power output in units of kW.
        :return: max_power_out
        """
        return self.get_max_input_power()

    @property
    def max_energy_in(self) -> float:
        """
        Computes the battery's maximum energy input.
        """
        return self.get_max_input_power()

    @property
    def max_energy_out(self) -> float:
        """
        Computes the battery's maximum energy output.
        """
        return self.get_max_output_power()

    def get_current_efficiency(self, energy: float) -> float:
        """
        Computes the power efficiency of the battery.
        :param energy: Input power in kWh.
        :return: efficiency
        """
        if self.power_efficiency_curve is not None:
            return self.sample_curve(self.power_efficiency_curve, energy / self.nominal_power) \
                   ** self.efficiency_scaling
        else:
            return self.efficiency

    def charge(self, energy: float) -> None:
        """
        Charge the battery, update the current soc.
        :param energy: Input/Output energy in kWh
        :return: None
        """

        # energy = self.limit_energy(energy)
        # energy = min(energy, self.get_max_input_power()) if energy >= 0 else max(-self.get_max_output_power(), energy)
        # self.efficiency = self.get_current_efficiency(energy)

        # The initial State Of Charge (SOC) is the previous SOC minus the energy losses
        # soc = min(self.soc_init + energy * self.efficiency, self.capacity) if energy >= 0. \
        #     else max(0., self.soc_init + energy / self.efficiency)
        soc = energy + self.soc_init
        self.norm_current_soc = soc / self.capacity
        self.soc_history.append(soc)
        self.efficiency_history.append(self.efficiency)
        self.energy_balance.append(self.set_energy_balance())
        self.capacity -= self.loss_coefficient * self.capacity_history[0] * np.abs(self.energy_balance[-1]) \
                         / (2 * self.capacity)
        self.capacity_history.append(self.capacity)

    def limit_energy(self, energy: float) -> float:
        """
        Limits the energy to the maximum possible energy that can be charged or discharged.
        :param energy: Input/Output energy in kWh
        :return: limited_energy
        """
        if energy >= 0:  # charge
            energy = min(energy, self.get_max_input_power())  # Limit the energy due to power
            self.efficiency = self.get_current_efficiency(energy)
            energy = min(self.soc_init + energy * self.efficiency, self.capacity) - self.soc_init  # Limit the energy
            #                                                                                        due to capacity
        else:  # discharge
            energy = max(-self.get_max_output_power(), energy)
            self.efficiency = self.get_current_efficiency(energy)
            energy = max(0., self.soc_init + energy / self.efficiency) - self.soc_init
        return energy

    def set_energy_balance(self) -> float:
        """Calculate energy balance

        The energy balance is a derived quantity and is the product or quotient of the difference between consecutive
        SOCs and `efficiency` for discharge or charge events respectively thus, thus accounts for energy losses to
        environment during charging and discharge.
        """

        # actual energy charged/discharged irrespective of what is determined in the step function after
        # taking into account storage design limits e.g. maximum power input/output, capacity
        previous_soc = self.initial_soc if len(self.soc_history) < 2 else self.soc_history[-2]
        energy_balance = self.soc_history[-1] - previous_soc * (1.0 - self.loss_coefficient)
        energy_balance = energy_balance / self.efficiency if energy_balance >= 0 else energy_balance * self.efficiency
        return energy_balance

    def simulate_charge_discharge(self, energy) -> (float, float, float, float):

        # compute the actual energy being charged (in limits) and the action needed
        energy, action, efficiency = self.energy_to_action_and_efficiency(energy)

        # get current efficiency
        # efficiency = self.get_current_efficiency(energy)

        # simulate the next soc
        # soc = min(self.soc_init + energy * self.efficiency, self.capacity) if energy >= 0 \
        #     else max(0., self.soc_init + energy / self.efficiency)
        soc = energy + self.soc_init

        return action, soc, energy, efficiency

    def energy_to_action_and_efficiency(self, energy) -> (float, float, float):
        """
        Given an energy to charge/discharge, returns the needed action in [0, 1] and actual respective energy
        """

        # limit energy to battery's capacity
        energy = self.limit_energy(energy)

        # compute the action
        action = energy / self.capacity
        return energy, action, self.efficiency

    def update_soc(self, action) -> None:
        """
        Updates the battery's SoC (State of Charge) given an action from the agent.
        :param action: The agent's action in [0, 1]
        :return: None
        """
        # compute energy from action
        energy = action * self.capacity

        energy = self.limit_energy(energy)

        # get current efficiency
        # efficiency = self.get_current_efficiency(energy) <- implemented inside 'charge()' function

        # charge battery
        self.charge(energy)
        # self.charge(energy, efficiency)

        # degrade capacity
        # self.capacity_degrade() <- implemented inside 'charge()' function


def main():
    power_curve = [[0., 0.6, 1.0], [1.0, 0.95, 0.43]]
    efficiency_curve = [[0., 0.2, 0.6, 1.], [0.84, 0.87, 0.95, 0.91]]
    battery = Battery(initial_soc=0,
                      initial_capacity=4.5,
                      nominal_power=2.1,
                      capacity_power_curve=power_curve,
                      loss_coefficient=2e-5,
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
