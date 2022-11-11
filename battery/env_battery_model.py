from typing import Iterable, List, Union
import numpy as np
import uuid
np.seterr(divide='ignore', invalid='ignore')


class Environment:
    def __init__(self, seconds_per_time_step: float = None):
        """Initialize `Environment`.

        Attributes
        ----------
        seconds_per_time_step: float, default: 3600.0
           Number of seconds in 1 `time_step` and must be set to >= 1.
        """

        self.seconds_per_time_step = seconds_per_time_step
        self.__uid = uuid.uuid4().hex
        self.__time_step = None
        self.reset()

    @property
    def uid(self) -> str:
        r"""Unique environment ID."""

        return self.__uid

    @property
    def time_step(self) -> int:
        r"""Current environment time step."""

        return self.__time_step

    @property
    def seconds_per_time_step(self) -> float:
        r"""Number of seconds in 1 time step."""

        return self.__seconds_per_time_step

    @seconds_per_time_step.setter
    def seconds_per_time_step(self, seconds_per_time_step: float):
        if seconds_per_time_step is None:
            self.seconds_per_time_step = 3600.0
        else:
            assert seconds_per_time_step >= 1, 'seconds_per_time_step >= 1'
            self.__seconds_per_time_step = seconds_per_time_step

    def next_time_step(self):
        r"""Advance to next `time_step` value.

        Notes
        -----
        Override in subclass for custom implementation when advancing to next `time_step`.
        """

        self.__time_step += 1

    def reset(self):
        r"""Reset environment to initial state.

        Calls `reset_time_step`.

        Notes
        -----
        Override in subclass for custom implementation when reseting environment.
        """

        self.reset_time_step()

    def reset_time_step(self):
        r"""Reset `time_step` to initial state.

        Sets `time_step` to 0.
        """

        self.__time_step = 0


class Device(Environment):
    def __init__(self, efficiency: float = None, **kwargs):
        r"""Initialize `Device`.

        Parameters
        ----------
        efficiency : float, default: 1.0
            Technical efficiency. Must be set to > 0.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        super().__init__(**kwargs)
        self.efficiency = efficiency

    @property
    def efficiency(self) -> float:
        """Technical efficiency."""

        return self.__efficiency

    @efficiency.setter
    def efficiency(self, efficiency: float):
        if efficiency is None:
            self.__efficiency = 1.0
        else:
            assert efficiency > 0, 'efficiency must be > 0.'
            self.__efficiency = efficiency


class ElectricDevice(Device):
    def __init__(self, nominal_power: float, **kwargs):
        r"""Initialize `Device`.

        Parameters
        ----------
        nominal_power : float
            Electric device nominal power >= 0. If == 0, set to 0.00001 to avoid `ZeroDivisionError`.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        super().__init__(**kwargs)
        self.nominal_power = nominal_power

    @property
    def nominal_power(self) -> float:
        r"""Nominal power."""

        return self.__nominal_power

    @property
    def electricity_consumption(self) -> List[float]:
        r"""Electricity consumption time series."""

        return self.__electricity_consumption

    @property
    def available_nominal_power(self) -> float:
        r"""Difference between `nominal_power` and `electricity_consumption` at current `time_step`."""

        return None if self.nominal_power is None else self.nominal_power - self.electricity_consumption[self.time_step]

    @nominal_power.setter
    def nominal_power(self, nominal_power: float):
        if nominal_power is None or nominal_power == 0:
            self.__nominal_power = 0.00001
        else:
            assert nominal_power >= 0, 'nominal_power must be >= 0.'
            self.__nominal_power = nominal_power

    def update_electricity_consumption(self, electricity_consumption: float):
        r"""Updates `electricity_consumption` at current `time_step`.

        Parameters
        ----------
        electricity_consumption : float
            value to add to current `time_step` `electricity_consumption`. Must be >= 0.
        """

        assert electricity_consumption >= 0, 'electricity_consumption must be >= 0.'
        self.__electricity_consumption[self.time_step] += electricity_consumption

    def next_time_step(self):
        r"""Advance to next `time_step` and set `electricity_consumption` at new `time_step` to 0.0."""

        super().next_time_step()
        self.__electricity_consumption.append(0.0)

    def reset(self):
        r"""Reset `ElectricDevice` to initial state and set `electricity_consumption` at `time_step` 0 to = 0.0."""

        super().reset()
        self.__electricity_consumption = [0.0]


class StorageDevice(Device):
    def __init__(self, capacity: float, efficiency: float = None, loss_coefficient: float = None,
                 initial_soc: float = None, efficiency_scaling: float = None, **kwargs):
        r"""Initialize `StorageDevice`.

        Parameters
        ----------
        capacity : float
            Maximum amount of energy the storage device can store in [kWh]. Must be >= 0 and if == 0 or None, set to 0.00001 to avoid `ZeroDivisionError`.
        efficiency : float, default: 0.9
            Technical efficiency.
        loss_coefficient : float, default: 0.006
            Standby hourly losses. Must be between 0 and 1 (this value is often 0 or really close to 0).
        initial_soc : float, default: 0.0
            State of charge when `time_step` = 0. Must be >= 0 and < `capacity`.
        efficiency_scaling : float, default: 0.5
            `efficiency` exponent scaling for `efficienct` such that `efficiency` **= `efficiency_scaling`

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        self.efficiency_scaling = efficiency_scaling
        self.capacity = capacity
        self.loss_coefficient = loss_coefficient
        self.initial_soc = initial_soc
        super().__init__(efficiency=efficiency, **kwargs)

    @property
    def capacity(self) -> float:
        r"""Maximum amount of energy the storage device can store in [kWh]."""

        return self.__capacity

    @property
    def loss_coefficient(self) -> float:
        r"""Standby hourly losses."""

        return self.__loss_coefficient

    @property
    def initial_soc(self) -> float:
        r"""State of charge when `time_step` = 0 in [kWh]."""

        return self.__initial_soc

    @property
    def soc(self) -> List[float]:
        r"""State of charge time series in [kWh]."""

        return self.__soc

    @property
    def soc_init(self) -> float:
        r"""Latest state of charge after accounting for standby hourly lossses."""

        return self.__soc[-1] * (1 - self.loss_coefficient)

    @property
    def efficiency_scaling(self) -> float:
        r"""`efficiency` exponent scaling."""

        return self.__efficiency_scaling

    @property
    def energy_balance(self) -> List[float]:
        r"""Charged/discharged energy time series in [kWh]."""

        return self.__energy_balance

    @Device.efficiency.setter
    def efficiency(self, efficiency: float):
        efficiency = 0.9 if efficiency is None else efficiency
        Device.efficiency.fset(self, efficiency ** self.efficiency_scaling)

    @capacity.setter
    def capacity(self, capacity: float):
        if capacity is None or capacity == 0:
            self.__capacity = 0.00001
        else:
            assert capacity >= 0, 'capacity must be >= 0.'
            self.__capacity = capacity

    @loss_coefficient.setter
    def loss_coefficient(self, loss_coefficient: float):
        if loss_coefficient is None:
            self.__loss_coefficient = 0.006
        else:
            assert 0 <= loss_coefficient <= 1, 'initial_soc must be >= 0 and <= 1.'
            self.__loss_coefficient = loss_coefficient

    @initial_soc.setter
    def initial_soc(self, initial_soc: float):
        if initial_soc is None:
            self.__initial_soc = 0
        else:
            assert 0 <= initial_soc <= self.capacity, 'initial_soc must be >= 0 and <= capacity.'
            self.__initial_soc = initial_soc

    @efficiency_scaling.setter
    def efficiency_scaling(self, efficiency_scaling: float):
        if efficiency_scaling is None:
            self.__efficiency_scaling = 0.5
        else:
            self.__efficiency_scaling = efficiency_scaling

    def charge(self, energy: float):
        """Charges or discharges storage with respect to specified energy while considering `capacity` and `soc_init` limitations and, energy losses to the environment quantified by `efficiency`.

        Parameters
        ----------
        energy : float
            Energy to charge if (+) or discharge if (-) in [kWh].

        Notes
        -----
        If charging, soc = min(`soc_init` + energy*`efficiency`, `capacity`)
        If discharging, soc = max(0, `soc_init` + energy/`efficiency`)
        """

        # The initial State Of Charge (SOC) is the previous SOC minus the energy losses
        soc = min(self.soc_init + energy * self.efficiency, self.capacity) if energy >= 0 else max(0,
                                                                                                   self.soc_init + energy / self.efficiency)
        self.__soc.append(soc)
        self.__energy_balance.append(self.set_energy_balance())

    def set_energy_balance(self) -> float:
        r"""Calculate energy balance

        The energy balance is a derived quantity and is the product or quotient of the difference between consecutive SOCs and `efficiency`
        for discharge or charge events respectively thus, thus accounts for energy losses to environment during charging and discharge.
        """

        # actual energy charged/discharged irrespective of what is determined in the step function after
        # taking into account storage design limits e.g. maximum power input/output, capacity
        previous_soc = self.initial_soc if self.time_step == 0 else self.soc[-2]
        energy_balance = self.soc[-1] - previous_soc * (1.0 - self.loss_coefficient)
        energy_balance = energy_balance / self.efficiency if energy_balance >= 0 else energy_balance * self.efficiency
        return energy_balance

    def autosize(self, demand: Iterable[float], safety_factor: float = None):
        r"""Autosize `capacity`.

        Set `capacity` to the minimum capacity needed to always meet `demand`.

        Parameters
        ----------
        demand : Union[float, Iterable[float]], optional
            Heating emand in [kWh].
        safety_factor : float, default: 1.0
            The `capacity` is oversized by factor of `safety_factor`.

        Notes
        -----
        `capacity` = max(demand/`efficiency`)*safety_factor
        """

        safety_factor = 1.0 if safety_factor is None else safety_factor
        self.capacity = np.nanmax(demand) * safety_factor

    def reset(self):
        r"""Reset `StorageDevice` to initial state."""

        super().reset()
        self.__soc = [self.initial_soc]
        self.__energy_balance = [0.0]


class Battery(ElectricDevice, StorageDevice):
    def __init__(self, capacity: float, nominal_power: float, capacity_loss_coefficient: float = None,
                 power_efficiency_curve: List[List[float]] = None, capacity_power_curve: List[List[float]] = None,
                 **kwargs):
        r"""Initialize `Battery`.

        Parameters
        ----------
        capacity : float
            Maximum amount of energy the storage device can store in [kWh]. Must be >= 0 and if == 0 or None, set to 0.00001 to avoid `ZeroDivisionError`.
        nominal_power: float
            Maximum amount of electric power that the battery can use to charge or discharge.
        capacity_loss_coefficient : float, default: 0.00001
            Battery degradation; storage capacity lost in each charge and discharge cycle (as a fraction of the total capacity).
        power_efficiency_curve: list, default: [[0, 0.83],[0.3, 0.83],[0.7, 0.9],[0.8, 0.9],[1, 0.85]]
            Charging/Discharging efficiency as a function of the power released or consumed.
        capacity_power_curve: list, default: [[0.0, 1],[0.8, 1],[1.0, 0.2]]
            Maximum power of the battery as a function of its current state of charge.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super classes.
        """

        self.__efficiency_history = []
        self.__capacity_history = []
        super().__init__(capacity=capacity, nominal_power=nominal_power, **kwargs)
        self.capacity_loss_coefficient = capacity_loss_coefficient
        self.power_efficiency_curve = power_efficiency_curve
        self.capacity_power_curve = capacity_power_curve

    @StorageDevice.capacity.getter
    def capacity(self) -> float:
        r"""Current time step maximum amount of energy the storage device can store in [kWh]"""

        return self.capacity_history[-1]

    @StorageDevice.efficiency.getter
    def efficiency(self) -> float:
        """Current time step technical efficiency."""

        return self.efficiency_history[-1]

    @ElectricDevice.electricity_consumption.getter
    def electricity_consumption(self) -> List[float]:
        r"""Electricity consumption time series."""

        return self.energy_balance

    @property
    def capacity_loss_coefficient(self) -> float:
        """Battery degradation; storage capacity lost in each charge and discharge cycle (as a fraction of the total capacity)."""

        return self.__capacity_loss_coefficient

    @property
    def power_efficiency_curve(self) -> np.ndarray:
        """Charging/Discharging efficiency as a function of the power released or consumed."""

        return self.__power_efficiency_curve

    @property
    def capacity_power_curve(self) -> np.ndarray:
        """Maximum power of the battery as a function of its current state of charge."""

        return self.__capacity_power_curve

    @property
    def efficiency_history(self) -> List[float]:
        """Time series of technical efficiency."""

        return self.__efficiency_history

    @property
    def capacity_history(self) -> List[float]:
        """Time series of maximum amount of energy the storage device can store in [kWh]."""

        return self.__capacity_history

    @capacity.setter
    def capacity(self, capacity: float):
        StorageDevice.capacity.fset(self, capacity)
        self.__capacity_history.append(capacity)

    @efficiency.setter
    def efficiency(self, efficiency: float):
        efficiency = 0.9 if efficiency is None else efficiency
        StorageDevice.efficiency.fset(self, efficiency)
        self.__efficiency_history.append(efficiency)

    @capacity_loss_coefficient.setter
    def capacity_loss_coefficient(self, capacity_loss_coefficient: float):
        if capacity_loss_coefficient is None:
            capacity_loss_coefficient = 1e-5
        else:
            pass

        self.__capacity_loss_coefficient = capacity_loss_coefficient

    @power_efficiency_curve.setter
    def power_efficiency_curve(self, power_efficiency_curve: List[List[float]]):
        if power_efficiency_curve is None:
            power_efficiency_curve = [[0, 0.83], [0.3, 0.83], [0.7, 0.9], [0.8, 0.9], [1, 0.85]]
        else:
            pass

        self.__power_efficiency_curve = np.array(power_efficiency_curve).T

    @capacity_power_curve.setter
    def capacity_power_curve(self, capacity_power_curve: List[List[float]]):
        if capacity_power_curve is None:
            capacity_power_curve = [[0.0, 1], [0.8, 1], [1.0, 0.2]]
        else:
            pass

        self.__capacity_power_curve = np.array(capacity_power_curve).T

    def charge(self, energy: float):
        """Charges or discharges storage with respect to specified energy while considering `capacity` degradation and `soc_init` limitations, losses to the environment quantified by `efficiency`, `power_efficiency_curve` and `capacity_power_curve`.

        Parameters
        ----------
        energy : float
            Energy to charge if (+) or discharge if (-) in [kWh].

        Notes
        -----
        If charging, soc = min(`soc_init` + energy*`efficiency`, `max_input_power`, `capacity`)
        If discharging, soc = max(0, `soc_init` + energy/`efficiency`, `max_output_power`)
        """

        energy = min(energy, self.get_max_input_power()) if energy >= 0 else max(-self.get_max_output_power(), energy)
        self.efficiency = self.get_current_efficiency(energy)
        super().charge(energy)
        self.capacity = self.capacity - self.degrade()

    def get_max_output_power(self) -> float:
        r"""Get maximum output power while considering `capacity_power_curve` limitations if defined otherwise, returns `nominal_power`.

        Returns
        -------
        max_output_power : float
            Maximum amount of power that the storage unit can output [kW].
        """

        return self.get_max_input_power()

    def get_max_input_power(self) -> float:
        r"""Get maximum input power while considering `capacity_power_curve` limitations if defined otherwise, returns `nominal_power`.

        Returns
        -------
        max_input_power : float
            Maximum amount of power that the storage unit can use to charge [kW].
        """

        # The initial State Of Charge (SOC) is the previous SOC minus the energy losses
        if self.capacity_power_curve is not None:
            soc_normalized = self.soc_init / self.capacity
            # Calculating the maximum power rate at which the battery can be charged or discharged
            idx = max(0, np.argmax(soc_normalized <= self.capacity_power_curve[0]) - 1)
            max_output_power = self.nominal_power * (
                    self.capacity_power_curve[1][idx]
                    + (self.capacity_power_curve[1][idx + 1] - self.capacity_power_curve[1][idx]) * (
                                soc_normalized - self.capacity_power_curve[0][idx])
                    / (self.capacity_power_curve[0][idx + 1] - self.capacity_power_curve[0][idx])
            )
        else:
            max_output_power = self.nominal_power

        return max_output_power

    def get_current_efficiency(self, energy: float) -> float:
        r"""Get technical efficiency while considering `power_efficiency_curve` limitations if defined otherwise, returns `efficiency`.

        Returns
        -------
        efficiency : float
            Technical efficiency.
        """

        if self.power_efficiency_curve is not None:
            # Calculating the maximum power rate at which the battery can be charged or discharged
            energy_normalized = np.abs(energy) / self.nominal_power
            idx = max(0, np.argmax(energy_normalized <= self.power_efficiency_curve[0]) - 1)
            efficiency = self.power_efficiency_curve[1][idx] \
                         + (energy_normalized - self.power_efficiency_curve[0][idx]
                            ) * (self.power_efficiency_curve[1][idx + 1] - self.power_efficiency_curve[1][idx]
                                 ) / (self.power_efficiency_curve[0][idx + 1] - self.power_efficiency_curve[0][idx])
            efficiency = efficiency ** self.efficiency_scaling
        else:
            efficiency = self.efficiency

        return efficiency

    def degrade(self) -> float:
        r"""Get amount of capacity degradation.

        Returns
        -------
        capacity : float
            Maximum amount of energy the storage device can store in [kWh].

        Notes
        -----
        degradation = `capacity_loss_coef`*`capacity_history[0]`*abs(`energy_balance[-1]`)/(2*`capacity`)
        """

        # Calculating the degradation of the battery: new max. capacity of the battery after charge/discharge
        capacity_degrade = self.capacity_loss_coefficient * self.capacity_history[0] * np.abs(
            self.energy_balance[-1]) / (2 * self.capacity)
        return capacity_degrade

    def reset(self):
        r"""Reset `Battery` to initial state."""

        super().reset()
        self.__efficiency_history = self.__efficiency_history[0:1]
        self.__capacity_history = self.__capacity_history[0:1]


if __name__ == "__main__":
    params = {"capacity": 6.4,
              "nominal_power": 5.}
    battery = Battery(**params)