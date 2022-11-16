import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from citylearn.citylearn import CityLearnEnv
from agents.brute_force_day_agent import BruteForceAgent, BruteForceWeighedAgent




####################################################################################################
##                                                                                                ##
##                                      VISUALIZATION TOOLS                                       ##
##                                                                                                ##
####################################################################################################


def preprocess_obs_history(controller, scale=False):
    obs_history = np.zeros((controller.num_buildings, len(controller.observation_history),
                            len(controller.observation_history[0][0])), dtype=float)
    for t, observation in enumerate(controller.observation_history):
        for i in range(controller.num_buildings):
            obs_history[i, t, :] = np.array(observation[i])
    if scale:
        for j in range(len(controller.observation_history[0][0])):
            max_val = np.max(obs_history[:, :, j])
            min_val = np.min(obs_history[:, :, j])
            obs_history[:, :, j] = (obs_history[:, :, j] - min_val) / (max_val - min_val)

    return obs_history # dimensions = (num_buildings, simulation_period, observation_len)


def observation_map():
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
    obs_str_list = [
        'month', 'day_type', 'hour', 'outdoor_dry_bulb_temperature',
        'outdoor_dry_bulb_temperature_predicted_6h', 'outdoor_dry_bulb_temperature_predicted_12h',
        'outdoor_dry_bulb_temperature_predicted_24h', 'outdoor_relative_humidity',
        'outdoor_relative_humidity_predicted_6h', 'outdoor_relative_humidity_predicted_12h',
        'outdoor_relative_humidity_predicted_24h', 'diffuse_solar_irradiance',
        'diffuse_solar_irradiance_predicted_6h', 'diffuse_solar_irradiance_predicted_12h',
        'diffuse_solar_irradiance_predicted_24h', 'direct_solar_irradiance',
        'direct_solar_irradiance_predicted_6h', 'direct_solar_irradiance_predicted_12h',
        'direct_solar_irradiance_predicted_24h', 'carbon_intensity', 'non_shiftable_load',
        'solar_generation', 'electrical_storage_soc', 'net_electricity_consumption',
        'electricity_pricing', 'electricity_pricing_predicted_6h', 'electricity_pricing_predicted_12h',
        'electricity_pricing_predicted_24h'
            ]

    obs_map = {item: i for i, item in enumerate(obs_str_list)}
    return obs_map


def plot_interval_results(controller, sim_period=(0, 200), name=None, agent_ids=(),
                          obs_params=None, scale=False, plot_no_op_consumption=False):
    """

    :param controller: The controller class
    :param sim_period:
    :param name:
    :param agent_ids:
    :param obs_params:
    :param scale:
    :param plot_no_op_consumption:
    :return:
    """

    import matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 18})
    colors = list(matplotlib._color_data.TABLEAU_COLORS.keys())
    obs_map = observation_map()
    obs_data = preprocess_obs_history(controller, scale=scale)
    sim_period = list(sim_period)

    # verify plot length
    time_step = obs_data.shape[1]
    sim_period[1] = np.min([sim_period[1], 8760, time_step])
    assert sim_period[1] > sim_period[0]

    # compute the no op consumption per building
    no_op_consumption = np.zeros((controller.num_buildings, time_step), dtype=float)
    for i in range(controller.num_buildings):
        no_op_consumption[i] = obs_data[i, :, obs_map["non_shiftable_load"]] - obs_data[i, :,
                                                                               obs_map["solar_generation"]]

    # plot
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 9))
    if len(agent_ids) == 0:
        # plot the district values
        if isinstance(obs_params, list) or isinstance(obs_params, tuple):
            for param in obs_params:
                ax[0].plot(range(sim_period[0], sim_period[1]),
                           np.sum(obs_data[:, sim_period[0]:sim_period[1], obs_map[param]], axis=0),
                           label="district " + param)
            if plot_no_op_consumption:
                ax[0].plot(range(sim_period[0], sim_period[1]),
                           np.sum(no_op_consumption[:, sim_period[0]:sim_period[1]], axis=0),
                           label="district net consumption no battery")

            if scale:
                ax[0].set_ylabel('a.u.')
            else:
                ax[0].set_ylabel('value')
        else:
            ax[0].plot(range(sim_period[0], sim_period[1]),
                       np.sum(obs_data[:, sim_period[0]:sim_period[1], obs_map[obs_params]], axis=0),
                       label="district " + obs_params)
            if scale:
                ax[0].set_ylabel('a.u.')
            else:
                ax[0].set_ylabel(obs_params)
    else:
        if isinstance(obs_params, list) or isinstance(obs_params, tuple):
            for param in obs_params:
                for i in agent_ids:
                    ax[0].plot(range(sim_period[0], sim_period[1]),
                               obs_data[i, sim_period[0]:sim_period[1], obs_map[param]],
                               label="building-" + str(i) + " " + param)
            if scale:
                ax[0].set_ylabel('a.u.')
            else:
                ax[0].set_ylabel('value')
        else:
            for i in agent_ids:
                ax[0].plot(range(sim_period[0], sim_period[1]),
                           obs_data[i, sim_period[0]:sim_period[1], obs_map[obs_params]],
                           label="building-" + str(i) + " " + obs_params)
            if scale:
                ax[0].set_ylabel('a.u.')
            else:
                ax[0].set_ylabel(obs_params)
    ax[0].set_xlabel('time (hours)')
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(controller.scores_and_metrics["metric_eval_step"], controller.scores_and_metrics["price_cost"],
               label="price cost")
    ax[1].plot(controller.scores_and_metrics["metric_eval_step"], controller.scores_and_metrics["emission_cost"],
               label="emission cost")
    ax[1].plot(controller.scores_and_metrics["metric_eval_step"], controller.scores_and_metrics["grid_cost"],
               label="grid cost")
    ax[1].plot(controller.scores_and_metrics["metric_eval_step"], controller.scores_and_metrics["metric_value"],
               label="metric value")
    ax[1].plot(controller.scores_and_metrics["avg_metric_value"], controller.scores_and_metrics["avg_metric_value"],
               label="avg metric value")
    ax[1].set_xlabel("time [hour]")
    ax[1].set_ylabel("score")
    ax[1].legend()
    ax[1].grid()

    if name is not None:
        name = 'figures/' + name + '_' + str(sim_period[0]) + '_' + str(sim_period[1]) + '.png'
        plt.savefig(name)
    plt.show()


####################################################################################################
##                                                                                                ##
##                                         EVALUATION TOOLS                                       ##
##                                                                                                ##
####################################################################################################


# Reformat the action space into a dictionary from a list
def action_space_to_dict(aspace):
    """ Only for box space """
    return {"high": aspace.high,
            "low": aspace.low,
            "shape": aspace.shape,
            "dtype": str(aspace.dtype)
            }


# Reset the environment and create the observation dictionary with all environment info
def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    building_info = env.get_building_information()
    building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                "building_info": building_info,
                "observation": observations}
    return obs_dict


def rb_evaluate(CoordinatorClass, AgentClass, eval_constants, verbose=True):
    print("Starting local evaluation")
    start_time = time.process_time()
    # Create a ne environment
    env = CityLearnEnv(schema=eval_constants.schema_path)

    # Reset the environment
    obs_dict = env_reset(env)
    observations = obs_dict["observation"]

    # Add coordinator
    rb_coordinator = CoordinatorClass(AgentClass, obs_dict, **eval_constants.rule_based_params)

    # Init local variables
    episodes_completed = 0
    num_steps = 0
    episode_metrics = []

    # Define the agents' training time
    evaluation_steps = 24 * eval_constants.evaluation_days
    done = False

    # Start the evaluation process
    while True:
        num_steps += 1

        # Take an action
        actions = rb_coordinator.compute_action(observations)

        # Step the environment and collect the observation, reward (user written reward), done and info
        observations_, rewards, done, info = env.step(actions)

        # step the observation
        observations = observations_

        # collect rewards and compute scores and average scores
        rb_coordinator.collect_scores(rewards)

        if num_steps % eval_constants.compute_metric_interval == 0:
            # evaluate the agents
            metrics_t = env.evaluate()

            # collect the metrics
            rb_coordinator.collect_metrics(metrics_t)

            # print scores and metrics
            if verbose:
                rb_coordinator.print_scores_and_metrics(episodes_completed, num_steps)

        # evaluate the last episode and reset the environment
        if done:
            episodes_completed += 1
            metrics_t = env.evaluate()
            metrics = {"price_cost": metrics_t[0],
                       "emission_cost": metrics_t[1],
                       "grid_cost": metrics_t[2]}
            if np.any(np.isnan(metrics_t)):
                raise ValueError("Episode metrics are nan, please contant organizers")
            episode_metrics.append(metrics)
            print(f"Episode complete: {episodes_completed} | Latest episode metrics: {metrics}", )

            # compute average scores
            rb_coordinator.compute_avg_scores()

            # Reset the environment
            done = False
            obs_dict = env_reset(env)
            rb_coordinator.init_score()
            rb_coordinator.reset_battery()
            observations = obs_dict["observation"]

        # terminate evaluation
        if num_steps == evaluation_steps:
            print(f"Evaluation process is terminated after {num_steps} steps.")
            break

    # print the episode mean evaluation score
    if len(episode_metrics) > 0:
        print("Average Price Cost:", np.mean([e['price_cost'] for e in episode_metrics]))
        print("Average Emission Cost:", np.mean([e['emission_cost'] for e in episode_metrics]))
        print("Average Grid Cost:", np.mean([e['grid_cost'] for e in episode_metrics]))
        for e in episode_metrics:
            print(f"Episode Utility: {np.mean(list(e.values()))}")
    return env, rb_coordinator, episode_metrics, time.process_time() - start_time