import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from citylearn.citylearn import CityLearnEnv
from agents.brute_force_day_agent import BruteForceAgent, BruteForceLastAgent




####################################################################################################
##                                                                                                ##
##                                      VISUALIZATION TOOLS                                       ##
##                                                                                                ##
####################################################################################################


def plot_interval_results(env, sim_period=(0, 200), name=None, agent_ids=()):
    #TODO if needed check that working
    import matplotlib
    import matplotlib.pyplot as plt
    colors = list(matplotlib._color_data.TABLEAU_COLORS.keys())
    if sim_period[1] > 8760:
        sim_period[1] -= 8760
    plt.figure(figsize=(16, 9))
    if len(agent_ids) == 0:
        plt.plot(env.net_electricity_consumption_without_storage_and_pv[sim_period[0]:sim_period[1]])
        plt.plot(-env.net_electricity_consumption_without_storage[sim_period[0]:sim_period[1]] +
                 env.net_electricity_consumption_without_storage_and_pv[sim_period[0]:sim_period[1]])
        plt.plot(env.net_electricity_consumption_without_storage[sim_period[0]:sim_period[1]], '--')
        plt.plot(env.net_electricity_consumption[sim_period[0]:sim_period[1]], '--')
        plt.xlabel('time (hours)')
        plt.ylabel('kW')
        plt.legend(['Electricity demand without storage and solar generation (kW)',
                    'Electricity solar generation (kW)',
                    'Electricity demand with PV generation and without storage(kW)',
                    'Electricity demand with PV generation and using an AI agent for storage control (kW)'])
        if name is not None:
            name = 'figures/' + name + '_' + str(sim_period[0]) + '_' + str(sim_period[1]) + '.png'
            plt.savefig(name)
    else:
        for i, agent_id in enumerate(agent_ids):
            c = colors[i]
            plt.plot(env.buildings[agent_id].net_electricity_consumption_without_storage_and_pv[sim_period[0]:sim_period[1]], '-x', c=c,
                     markersize=14)
            plt.plot(-env.buildings[agent_id].net_electricity_consumption_without_storage[sim_period[0]:sim_period[1]] +
                     env.buildings[agent_id].net_electricity_consumption_without_storage_and_pv[sim_period[0]:sim_period[1]], '.-', c=c)
            plt.plot(env.buildings[agent_id].net_electricity_consumption_without_storage[sim_period[0]:sim_period[1]], '-o', c=c)
            plt.plot(env.buildings[agent_id].net_electricity_consumption[sim_period[0]:sim_period[1]], '--', c=c)
            plt.xlabel('time (hours)')
            plt.ylabel('kW')
            if i == 0:
                plt.legend(['Electricity demand without storage and solar generation (kW)',
                            'Electricity solar generation (kW)',
                            'Electricity demand with PV generation and without storage(kW)',
                            'Electricity demand with PV generation and using an AI agent for storage control (kW)'])
        if name is not None:
            name = 'figures/' + name + '_' + str(sim_period[0]) + '_' + str(sim_period[1]) + '.png'
            plt.savefig(name)
    plt.show()


def plot_scores(rl_coordinator, train_constants, name=None):
    # TODO if needed check that working
    # plot agents' scores
    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex='all')
    for agent_id in rl_coordinator.decision_makers.keys():
        axes[0].plot(np.array(range(1, rl_coordinator.num_episodes + 1)) * 8760,
                     rl_coordinator.scores_and_metrics['avg_scores'][agent_id], '--o', label='agent-' + str(agent_id))
    axes[0].set_ylabel('Avg score')
    axes[0].grid()
    axes[0].legend()

    axes[1].plot(rl_coordinator.scores_and_metrics['metric_eval_step'],
                 rl_coordinator.scores_and_metrics['avg_metric_value'], '-o', label='average metric value')
    axes[1].hlines(2, rl_coordinator.scores_and_metrics['metric_eval_step'][0],
                   rl_coordinator.scores_and_metrics['metric_eval_step'][-1], label='No battery baseline', color='red')
    axes[1].set_ylabel('metric value')
    axes[1].legend()
    axes[1].grid()

    # plot environment metrics
    axes[2].plot(rl_coordinator.scores_and_metrics["metric_eval_step"],
                 rl_coordinator.scores_and_metrics["price_cost"], label='price cost')
    axes[2].plot(rl_coordinator.scores_and_metrics["metric_eval_step"],
                 rl_coordinator.scores_and_metrics["emission_cost"], label='emission cost')
    axes[2].set_xlabel('time (hours)')
    axes[2].set_ylabel('price')
    axes[2].legend()
    axes[2].grid()
    if name is not None:
        name = 'figures/' + name + train_constants.name + rl_coordinator.decision_makers[0].model_name + '.png'
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