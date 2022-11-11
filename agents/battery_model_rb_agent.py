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
        day_type = observation[1]
        hour = observation[2]
        outdoor_dry_bulb_temperature = observation[3]
        outdoor_dry_bulb_temperature_predicted_6h = observation[4]
        outdoor_dry_bulb_temperature_predicted_12h = observation[5]
        outdoor_dry_bulb_temperature_predicted_24h = observation[6]
        outdoor_relative_humidity = observation[7]
        outdoor_relative_humidity_predicted_6h = observation[8]
        outdoor_relative_humidity_predicted_12h = observation[9]
        outdoor_relative_humidity_predicted_24h = observation[10]
        diffuse_solar_irradiance = observation[11]
        diffuse_solar_irradiance_predicted_6h = observation[12]
        diffuse_solar_irradiance_predicted_12h = observation[13]
        diffuse_solar_irradiance_predicted_24h = observation[14]
        direct_solar_irradiance = observation[15]
        direct_solar_irradiance_predicted_6h = observation[16]
        direct_solar_irradiance_predicted_12h = observation[17]
        direct_solar_irradiance_predicted_24h = observation[18]
        carbon_intensity = observation[19]
        non_shiftable_load = observation[20]
        solar_generation = observation[21]
        electrical_storage_soc = observation[22]
        net_electricity_consumption = observation[23]
        electricity_pricing = observation[24]
        electricity_pricing_predicted_6h = observation[25]
        electricity_pricing_predicted_12h = observation[26]
        electricity_pricing_predicted_24h = observation[27]

        # update the state of the battery
        self.override_soc_from_observation(electrical_storage_soc)

        # # check that the agent's battery models soc is identical to soc from environment
        # assert self.norm_current_soc == electrical_storage_soc, f"Agent's battery model soc " \
        #                                                         f"is {self.norm_current_soc}, " \
        #                                                         f"while should be {electrical_storage_soc}"

        # add new electricity price to price set
        if electricity_pricing not in self.electricity_price_set:
            self.electricity_price_set.add(electricity_pricing)

        # update min and max observed electricity prices
        self.max_price = max(self.electricity_price_set)
        self.min_price = min(self.electricity_price_set)

        action = 0.0  # Default value

        # check for excess electricity generation
        excess_energy = solar_generation - non_shiftable_load
        if excess_energy > 0:
            if electricity_pricing < self.max_price:
                # compute the action needed for charging the excess energy (in limits)
                action, next_soc, energy_to_charge, charge_efficiency = self.simulate_charge_discharge(excess_energy)

            else:
                # compute the energy needed for max energy discharge
                action, next_soc, energy_to_charge, charge_efficiency = self.simulate_charge_discharge(
                    -self.max_energy_out)

        else:
            if self.norm_current_soc < 0.8:
                if electricity_pricing < 0.4:
                    # compute the energy to charge the battery if the electricity price is low
                    action, next_soc, energy_to_charge, charge_efficiency = self.simulate_charge_discharge(
                        self.max_energy_in)
                else:
                    action, next_soc, energy_to_charge, charge_efficiency = self.simulate_charge_discharge(0)
            else:
                action, next_soc, energy_to_charge, charge_efficiency = self.simulate_charge_discharge(excess_energy)

        # update the agent's battery models soc
        action = np.round(np.array([action], dtype=action_space.dtype), 4)
        self.update_soc(action[0])


        assert action_space.contains(action)
        return action

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

    def old_policy_with_global_prediction(self, observation, action_space, prediction, rest_district_net_cons_prediction):

        # override battery soc from observation
        electrical_storage_soc = observation[22]
        self.override_soc_from_observation(electrical_storage_soc)

        # predicted net consumption with no battery
        district_net_consumption_next_hour = prediction["building_net_consumption"][0] + rest_district_net_cons_prediction
        electricity_pricing = prediction["pricing"][0]

        # physical quantities
        nominal_power = self.get_max_input_power()

        action = 0.0  # Default value
        if district_net_consumption_next_hour <= 0:  # if having extra electricity generated enter
            dsoc_kwh = min((np.abs(district_net_consumption_next_hour)), nominal_power)
            action = min(dsoc_kwh / self.capacity / self.get_current_efficiency(dsoc_kwh), min(1 - self.soc_init / self.capacity, 0))  # add the exact amount needed to fill the battery
            if action == 0.:
                action = -0.1

        else:  # if electricity load is higher than generation enter
            dsoc_kwh = max((-district_net_consumption_next_hour),
                           -nominal_power)  # compute the amount of energy discharge needed for break-even
            action = max(dsoc_kwh / self.capacity / self.get_current_efficiency(dsoc_kwh), -(self.soc_init / self.capacity))  # set action to discharge to break-even
            if action == 0.:
                action = 0.1

        action = np.array([action], dtype=action_space.dtype)
        assert action_space.contains(action)

        return action

