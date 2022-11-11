import pandas as pd
from battery.env_battery_model import Battery
from rewards.agent_inst_u import *


class BatteryModelRBAgent(Battery):
    def __init__(self, *args, **kwargs):
        super().__init__(capacity=6.4, nominal_power=5.0, *args, **kwargs)
        self.action_space = {}
        self.agent_net_no_battery = {}
        self.idx = {}
        self.max_price = 0.54
        self.min_price = 0.21
        self.agent_type = 'RB'
        self.agent_id = None
        self.data = []
        # self.data_history = {"month": [],
        #                      "price": [],
        #                      "carbon": [],
        #                      "load": [],
        #                      "solar": [],
        #                      "net_no_battery": [],
        #                      "soc": [],
        #                      "net_battery": [],
        #                      "actions": [],
        #                      "actions_kwh": []
        #                      }

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        data = np.concatenate((pd.read_csv(f"data/rb_agent_building_{str(agent_id)}.csv")["net_no_battery"].values[1:], np.array([0])))
        self.agent_net_no_battery[agent_id] = np.concatenate((data, data))
        self.idx[agent_id] = 0
        self.agent_id = agent_id

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        return self.policy_old_with_prediction(observation, self.action_space[agent_id])

    def policy(self, observation, action_space):
        """
        observation[0] = 'month',
        observation[1] = 'day_type',
        observation[2] = 'hour',
        observation[3] = 'outdoor_dry_bulb_temperature',
        observation[4] = 'outdoor_dry_bulb_temperature_predicted_6h',
        observation[5] = 'outdoor_dry_bulb_temperature_predicted_12h',
        observation[6] = 'outdoor_dry_bulb_temperature_predicted_24h',
        observation[7] = 'outdoor_relative_humidity',
        observation[8] = 'outdoor_relative_humidity_predicted_6h',
        observation[9] = 'outdoor_relative_humidity_predicted_12h',
        observation[10] = 'outdoor_relative_humidity_predicted_24h',
        observation[11] = 'diffuse_solar_irradiance',
        observation[12] = 'diffuse_solar_irradiance_predicted_6h',
        observation[13] = 'diffuse_solar_irradiance_predicted_12h',
        observation[14] = 'diffuse_solar_irradiance_predicted_24h',
        observation[15] = 'direct_solar_irradiance',
        observation[16] = 'direct_solar_irradiance_predicted_6h',
        observation[17] = 'direct_solar_irradiance_predicted_12h',
        observation[18] = 'direct_solar_irradiance_predicted_24h',
        observation[19] = 'carbon_intensity',
        observation[20] = 'non_shiftable_load',
        observation[21] = 'solar_generation',
        observation[22] = 'electrical_storage_soc',
        observation[23] = 'net_electricity_consumption',
        observation[24] = 'electricity_pricing',
        observation[25] = 'electricity_pricing_predicted_6h',
        observation[26] = 'electricity_pricing_predicted_12h',
        observation[27] = 'electricity_pricing_predicted_24h'
        """

        # unpack observation variables
        month = observation[0]
        hour = observation[2]
        carbon_intensity = observation[19]
        non_shiftable_load = observation[20]
        solar_generation = observation[21]
        electrical_storage_soc = observation[22]
        net_electricity_consumption = observation[23]
        electricity_pricing = observation[24]

        # print(electrical_storage_soc - self.soc_init / self.capacity)
        building_net_electricity_consumption = non_shiftable_load - solar_generation

        # if self.agent_id == 0:
        #     print(f'agent id: {self.agent_id}')
        #     print(f'env soc: {electrical_storage_soc}')
        #     print(f'building net: {building_net_electricity_consumption}')
        #     print(f'price {electricity_pricing}')
        #     print(self)

        if building_net_electricity_consumption < 0:
            if electricity_pricing < self.max_price:
                if np.abs(self.soc[-1] - self.capacity) < 0.01:
                    action = 0.
                else:
                    action = np.min([self.capacity - self.soc[-1],
                                     np.abs(building_net_electricity_consumption), self.nominal_power]) / self.capacity
            else:
                action = np.max([-self.soc[-1], building_net_electricity_consumption,
                                 -self.nominal_power]) / self.capacity
        else:
            action = np.min([self.capacity - self.soc[-1],
                             building_net_electricity_consumption, self.nominal_power]) / self.capacity
        self.charge(action)
        action = self.soc_init / self.capacity

        # if self.agent_id == 0:
        #     print(f'action: {action}')

        action = [action]
        assert action_space.contains(action)
        return action

    def save_history(self):
        pass

    def policy_old(self, observation, action_space):

        # observation variables
        month = observation[0]
        hour = observation[2]  # Hour index is 2 for all observations
        carbon_intensity = observation[19]
        non_shiftable_load = observation[20]
        solar_generation = observation[21]
        electrical_storage_soc = observation[22]  # in [0, 1]
        net_electricity_consumption = observation[23]
        electricity_pricing = observation[24]
        electricity_pricing_6h = observation[25]
        electricity_pricing_12h = observation[26]

        # update min and max observable electricity prices
        self.max_price = electricity_pricing if electricity_pricing > self.max_price else self.max_price
        self.min_price = electricity_pricing if electricity_pricing < self.min_price else self.min_price

        # physical quantities
        capacity = 6.4
        efficiency = 0.9
        capacity_loss_coefficient = 1e-05
        loss_coefficient = 0.0
        nominal_power = 5.0

        action = 0.0  # Default value
        if solar_generation >= non_shiftable_load:  # if having extra electricity generated enter
            if electricity_pricing < self.max_price:  # if the electricity price is smaller than the maximal price enter
                if electrical_storage_soc < 1:  # if the battery is not full enter
                    dsoc_kwh = min((solar_generation - non_shiftable_load), nominal_power)
                    action = min(dsoc_kwh / capacity / efficiency, 1)  # add the exact amount needed to fill the battery
                else:  # if the battery is full enter
                    action = 0  # do nothing - means, sell extra generated electricity back to the grid
                    dsoc_kwh = 0.
            else:  # if the electricity price is maximal enter
                action = -1  # full discharge the battery
                dsoc_kwh = -nominal_power

        else:  # if electricity load is higher than generation enter
            dsoc_kwh = max((solar_generation - non_shiftable_load),
                           -nominal_power)  # compute the amount of energy discharge needed for break-even
            action = max(dsoc_kwh / capacity / efficiency, -1)  # set action to discharge to break-even
            dsoc = max((solar_generation - non_shiftable_load), -nominal_power)

        # if self.agent_id == 0:
        #     action = 1
        #     print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        #     print(f"load - solar: {non_shiftable_load - solar_generation}")
        #     print(f"net: {net_electricity_consumption}")
        #     print(f"diff: {net_electricity_consumption - (non_shiftable_load - solar_generation)}")

        action = np.array([action], dtype=action_space.dtype)
        assert action_space.contains(action)


        return action

    def policy_old_with_prediction(self, observation, action_space):

        # observation variables
        month = observation[0]
        hour = observation[2]  # Hour index is 2 for all observations
        carbon_intensity = observation[19]
        non_shiftable_load = observation[20]
        solar_generation = observation[21]
        electrical_storage_soc = observation[22]  # in [0, 1]
        net_electricity_consumption = observation[23]
        electricity_pricing = observation[24]
        electricity_pricing_6h = observation[25]
        electricity_pricing_12h = observation[26]

        # predicted net consumption with no battery
        net_consumption_no_battery_next_hour = self.agent_net_no_battery[self.agent_id][self.idx[self.agent_id]]
        self.idx[self.agent_id] += 1

        # physical quantities
        capacity = 6.4
        efficiency = 0.9
        capacity_loss_coefficient = 1e-05
        loss_coefficient = 0.0
        nominal_power = 5.0

        action = 0.0  # Default value
        if net_consumption_no_battery_next_hour <= 0:  # if having extra electricity generated enter
            if electricity_pricing < self.max_price:  # if the electricity price is smaller than the maximal price enter
                if electrical_storage_soc < 1.:  # if the battery is not full enter
                    dsoc_kwh = min((np.abs(net_consumption_no_battery_next_hour)), nominal_power)
                    action = min(dsoc_kwh / capacity / efficiency, 1)  # add the exact amount needed to fill the battery
                else:  # if the battery is full enter
                    action = 0  # do nothing - means, sell extra generated electricity back to the grid
                    dsoc_kwh = 0.
            else:  # if the electricity price is maximal enter
                dsoc_kwh = 0.7 * min((np.abs(net_consumption_no_battery_next_hour)), nominal_power)
                action = min(dsoc_kwh / capacity / efficiency, 1)  # add the exact amount needed to fill the battery

                # action = -1  # full discharge the battery
                # dsoc_kwh = -nominal_power

        else:  # if electricity load is higher than generation enter
            dsoc_kwh = max((-net_consumption_no_battery_next_hour),
                           -nominal_power)  # compute the amount of energy discharge needed for break-even
            action = max(dsoc_kwh / capacity / efficiency, -1)  # set action to discharge to break-even
            dsoc = max((-net_consumption_no_battery_next_hour), -nominal_power)

        # if self.agent_id == 0:
        #     action = 1
        #     print("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        #     print(f"load - solar: {non_shiftable_load - solar_generation}")
        #     print(f"net: {net_electricity_consumption}")
        #     print(f"diff: {net_electricity_consumption - (non_shiftable_load - solar_generation)}")

        action = np.array([action], dtype=action_space.dtype)
        assert action_space.contains(action)

        return action

