import numpy as np


class BestRBCAgent:
    # Episode complete: 1 | Latest episode metrics: {'price_cost': 0.72501568351974, 'emmision_cost': 0.8688463349713167}
    def __init__(self):
        self.action_space = {}
        self.max_price = -np.inf
        self.min_price = np.inf
        self.soc = 0
        self.agent_type = 'RB'

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        return self.policy(observation, self.action_space[agent_id])

    def policy(self, observation, action_space):
        """
        observation[2]  = hour
        observation[19] = carbon_intensity
        observation[20] = non_shiftable_load
        observation[21] = solar_generation
        observation[22] = electrical_storage_soc
        observation[23] = net_electricity_consumption
        observation[24] = electricity_pricing
        observation[25] = electricity_pricing_6h
        observation[26] = electricity_pricing_12h
        """


        # observation variables
        hour = observation[2]  # Hour index is 2 for all observations
        carbon_intensity = observation[19]
        non_shiftable_load = observation[20]
        solar_generation = observation[21]
        electrical_storage_soc = observation[22]        # in [0, 1]
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
        if solar_generation >= non_shiftable_load:
            if electricity_pricing < self.max_price:
                if electrical_storage_soc < 1:
                    dsoc_kwh = min((solar_generation - non_shiftable_load), nominal_power)
                    action = min(dsoc_kwh / capacity / efficiency, 1)
                else:
                    action = 0
            else:
                action = -1
                dsoc_kwh = -nominal_power

        else:
            dsoc_kwh = max((solar_generation - non_shiftable_load), -nominal_power)
            action = max(dsoc_kwh / capacity / efficiency, -1)
            dsoc = max((solar_generation - non_shiftable_load), -nominal_power)

        action = np.array([action], dtype=action_space.dtype)
        assert action_space.contains(action)
        return action
