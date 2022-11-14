import numpy as np
from battery.battery_model_new import Battery


class BatteryModelRBAgent(Battery):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = {}
        self.max_price = np.inf
        self.min_price = 0.22
        self.agent_type = 'RB'
        self.agent_id = None

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space
        self.agent_id = agent_id

    def compute_local_action(self, observation, agent_id, prediction):
        """Get observation return action"""
        return self.policy_with_local_prediction(observation, self.action_space[agent_id], prediction)

    def compute_global_action(self, observation, agent_id, prediction, rest_district_net_cons_prediction, prev_rest_district_net_cons):
        """Get observation return action"""
        return self.policy_with_global_prediction(observation, self.action_space[agent_id], prediction, rest_district_net_cons_prediction, prev_rest_district_net_cons)

    def policy_with_local_prediction(self, observation, action_space, prediction):


        # override battery soc from observation
        electrical_storage_soc = observation[22]
        prev_net_consumption = observation[23]
        self.override_soc_from_observation(electrical_storage_soc)

        # predicted net consumption with no battery
        net_consumption_no_battery_next_hour = prediction["building_net_consumption"][0]
        d_net_consumption = net_consumption_no_battery_next_hour - prev_net_consumption
        electricity_pricing = prediction["pricing"][0]

        # physical quantities
        nominal_power = self.get_max_input_power()

        action = 0.0  # Default value
        if net_consumption_no_battery_next_hour <= 0:  # if having extra electricity generated enter
            if electricity_pricing < self.max_price:  # if the electricity price is smaller than the maximal price fill the battery
                if electrical_storage_soc < 1.:  # if the battery is not full, fill it up
                    dsoc_kwh = np.abs(net_consumption_no_battery_next_hour)
                    action = min(dsoc_kwh / self.capacity / self.get_current_efficiency(dsoc_kwh),
                                 nominal_power/self.capacity, 1-self.soc_init/self.capacity)  # add the exact amount needed to fill the battery
                else:  # if the battery is full enter
                    action = 0  # do nothing - means, sell extra generated electricity back to the grid
                    dsoc_kwh = 0.
            else:  # if the electricity price is maximal sell maximal energy possible from the battery on top of having extra generation
                # THIS NEVER HAPPENS IN THE SIMULATION !!!
                action = max(net_consumption_no_battery_next_hour / self.capacity /
                             self.get_current_efficiency(net_consumption_no_battery_next_hour),
                             -self.soc_init/self.capacity, -nominal_power/self.capacity)  #

        else:  # if electricity load is higher than generation
            # if electricity_pricing != self.min_price:
            # compute the amount of energy discharge needed from the battery to break-even
            action = max(-net_consumption_no_battery_next_hour / self.capacity /
                         self.get_current_efficiency(-net_consumption_no_battery_next_hour),
                         -self.soc_init/self.capacity,
                         -self.nominal_power/self.capacity)  # set action to discharge to break-even

        # if the change in net consumption is large due to the action, decrease action
        if np.abs(d_net_consumption) < np.abs(d_net_consumption + action * self.capacity):
            action *= 0.795

        action = np.array([action], dtype=action_space.dtype)
        assert action_space.contains(action)

        return action

    def policy_with_global_prediction(self, observation, action_space, prediction, rest_district_net_cons_prediction, prev_rest_district_net_cons):

        # override battery soc from observation
        electrical_storage_soc = observation[22]
        self.override_soc_from_observation(electrical_storage_soc)

        # predicted net consumption with no battery
        district_net_consumption_next_hour = prediction["building_net_consumption"][0] + rest_district_net_cons_prediction
        d_net_cons = district_net_consumption_next_hour - prev_rest_district_net_cons
        electricity_pricing = prediction["pricing"][0]

        # physical quantities
        nominal_power = self.get_max_input_power()

        action = 0.0  # Default value
        if district_net_consumption_next_hour <= 0:  # if having extra electricity generated enter
            if electricity_pricing < self.max_price:  # if the electricity price is smaller than the maximal price fill the battery
                if electrical_storage_soc < 1.:  # if the battery is not full, fill it up
                    dsoc_kwh = np.abs(district_net_consumption_next_hour)
                    action = min(dsoc_kwh / self.capacity / self.get_current_efficiency(dsoc_kwh),
                                 nominal_power / self.capacity,
                                 1 - self.soc_init / self.capacity)  # add the exact amount needed to fill the battery
                else:  # if the battery is full enter
                    action = 0  # do nothing - means, sell extra generated electricity back to the grid
                    dsoc_kwh = 0.
            else:  # if the electricity price is maximal sell maximal energy possible from the battery on top of having extra generation
                # THIS NEVER HAPPENS IN THE SIMULATION !!!
                action = max(district_net_consumption_next_hour / self.capacity /
                             self.get_current_efficiency(district_net_consumption_next_hour),
                             -self.soc_init / self.capacity, -nominal_power / self.capacity)  #

        else:  # if electricity load is higher than generation
            # if electricity_pricing != self.min_price:
            # compute the amount of energy discharge needed from the battery to break-even
            action = max(-district_net_consumption_next_hour / self.capacity /
                         self.get_current_efficiency(-district_net_consumption_next_hour),
                         -self.soc_init / self.capacity,
                         -self.nominal_power / self.capacity)  # set action to discharge to break-even

        # if the change in net consumption is large due to the action, decrease action
        # if np.abs(d_net_cons) < np.abs(d_net_cons + action * self.capacity):
        #     action *= 0.5

        action = np.array([action], dtype=action_space.dtype)
        assert action_space.contains(action)

        return action