# CityLearn Multi-Agent Smart-Grid Smart-Tree Smart-Search 

> !!! Disclaimer: This repository is currently under development, so be patient with bugs.
  If you find any, please let us know (the contact info is below) !!!
 
> This repository contains the code for our implementation of a solution for the
 [2022 CityLearn challenge](https://github.com/intelligent-environments-lab/CityLearn).


## Introduction

The imposed problem is a multi-agent scenario, where the agents are the buildings in the smart-grid.
There is a battery in each building, which can be used to store energy, and a solar panel, which produces energy.
Each building has its own energy consumption and production, and the goal is to minimize the utility,
which is a specified measure of the net energy consumption of the buildings in the grid, 
parts of which are global to the whole district and parts of which are local to each building.
The action-space is the amount of energy to be stored in the battery, and the observation space is the
energy consumption and production of the building, as well as additional global parameters such as the 
electricity price, the CO2 intensity per unit of electricity, the weather parameters, etc.

The crux of the problem is that:
- The actions affect the next step, so the net consumption of each building has to be predicted.
- The utility involves global parts, so the optimal action for one building depends on the actions of the other buildings.
- The natural periodicity of the net consumption is 24 hours, which even for planning using tree-search 
  algorithms with moderate branching factors is a lot of states to consider (e.g., `5**24=6e17`).


## Solution

The `i`'th building's net consumption at time `t` is made out of three key elements:

$$E^{(i,t)} = E_{Load}^{(i,t)} + E_{Solar}^{(i,t)} + E_{Storage}^{(i,t)}$$

The non-shiftable load $E_{Load}^{(i,t)}$ and the solar generation $E_{Solar}^{(i,t)} terms are given from the environment.
The storage $E_{Storage}^{(i,t)}$ is the action of the agent, which is the amount of energy to be charged/discharged from the battery.

We can therefore factorize the problem into two sub-environments, one for the agent and one for the "world", where the
actions only affect the agent's environment, and the time-evolution is dictated by the world's environment.

The actions affect the net electricity consumption via the equation above, and accordingly affect the utility function.



We implement and use:

- Uniform-Cost Search algorithm (a type of tree-search), which is a modified 
[Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm), to find the optimal action for each building.
- Various predictors to predict the net consumption of each building for the next time frame (1 or 24 hrs)
- Battery model to translate action to change in battery's state of charge (SoC).
- Local (instantaneous) estimation of the utility function to guide the search.
- Depth-selective search, where the search is performed only on specified levels of the tree, and the rest is 
  bridged by steps among which the action is uniformly divided.
- Some utility terms are global. Namely, they are effected by the actions of all the buildings in the district.
  We use decentralized controllers, where each building is taking actions to optimize local reward function.
  Therefore, a local estimation of the global utility is established and used to guide the search for each agent.
- Residual corrections between the sum of local trajectories and the global optimal behavior are taken care of by the last agent. 
  This is done by a simple heuristic, where it optimizes a global utility with the sum of net consumptions (modified with the planned actions). 

##### Etymology
Let's break down the name of the repository: 
- `CityLearn` is the name of the challenge, referring to the fact that there is a collective (city) learning goal.
- `Multi-Agent` is the type of problem, referring to the same thing basically.
- `Smart-Grid` is the domain of the problem, where the buildings have energy consumption and production, and a battery.
   So they can be smart, and use the grid in some optimal fashion.
- `Smart-Tree` refers to the uniform-cost tree-search algorithm we use to solve the problem, with its extra spices, e.g., 
  depth-selective search and non-uniform action space discretization.
- `Smart-Search` refers to the fact that we employ a battery model, and use mixed decision-makers for the decentralized controllers.


### Formulating the problem as an MDP



### Battery model

We reverse-engineered the battery model from the `CityLearn` environment, and used it as the MDP's (Markov decision process)
transition function for the planning.

The key parameters of the model are the battery's capacity, the battery's charging efficiency, and the battery's discharging efficiency.
SoC
capacity
nominal power

> copy from overleaf





### Local utility estimation

The utility is a function of the net consumption ($E$), that is only evaluated at the end of the year (episode).
However, the predictions and actions are made at each step, so we need to estimate the utility at each step.
For this purpose, we use an instantaneous utility estimation, which is an approximation of the utility function.

To motivate the construction of the instantaneous utility function, we first observe the utility function as a function of the net consumption:


There is a couple of estimators.
The first one uses for a single building, independent of the other buildings.
The second one uses the net consumption of the whole district, using the actions of the previous agents and 
no-op's as estimations for the missing next buildings.



### Hierarchical control scheme

We use a decentralised control setting for all agents, where each building has its own set of decision-makers.
The last agent is knowledgeable about the rest of the agents (net consumptions modified by the planned actions),
and can use this information to make better decisions.

<figure>
<img src="/"  width="900" 
alt="."/>
</figure>
> 

### Depth-selective search



## Alternative Rule-based solution
A set of rules defines the next move for each building independently (locally), based on the next hour prediction.
The rules were defined to "flatten" the net consumption curve (closing the temporal gap / phase-shift between 
peak production and peak demand), and by this to minimize the utility:
- If the next hour production is higher than the consumption, the battery is charged by the extra amount.
- If the next hour consumption is higher than the production, the battery is discharged by the missing amount.

On top of that, the rules treat the cases where the battery is fully charged or fully discharged.
We also penalize the battery charge, in hours when the carbon intensity is below its median,
as in such times the utility for using the grid power is relatively lower.

The rules are defined in two cases, for a single building and for a group of buildings.
The essence is the same, just that for the later case, the input is the net consumption of the group.

Additional tuning was done to the rules, to minimize the utility for the training set, and the parameters for the
single and group rules were found to be different.
An important hyperparameter is thus the number of buildings which use the group rules.

<figure>
<img src="/"  width="900" 
alt="."/>
</figure>

> Rule-based solution for a single building

## Tunable parameters

Controller parameters:
- `random_order`: Whether to choose the order of the buildings randomly each time-step or not.
- `prediction_method`: The method to use for predicting the net consumption of each building. Choose from:
  - `IDX`: Use the time and building indices for perfect prediction over the traning set.
  - `CSV`: Load the predictions from a CSV file.
  - `DOT`: Generate prediction by finding the maximal dot-product overlap of the past 24hr consumption and training data.
  - `MLP`: Predict with Multi-Layer Perceptron, using 24h history of net consumption and global variables.
- `agent_type`: The type of decision-maker to use for all agents except the last one (N-1 agents). Choose from:
  - `RB-local`: Use Rule-Based agents.
  - `PLAN-local`: Use the Uniform-Cost Search algorithm.
`last_agent_type`: The type of decision-maker to use for the last agent. Choose from:
  - `RB-local`: Use Rule-Based agent -- *egoistic* decision-maker using only individual net consumption.
  - `RB-global`: Use Rule-Based agent, but with global net consumption -- *altruistic* decision-maker using only collective district's net consumption.
  - `PLAN-local`: Use the Uniform-Cost Search algorithm.
  - `PLAN-global`: Use the Uniform-Cost Search algorithm, but with global net consumption. [not implemented]

Planner parameters:
- `search_depths`: list of depths to search to. The gaps are filled with constant action of the last searched depth.
- `max_serach_time`: time to terminate search if not finished, and keep to the best tarjectory found by this time.
- `d_action`: the action-space is discrete, so this is the step-size of the action-space.
- `acion_space_list`: list of actions to search over. If `None`, then the action-space is discretized to `d_action` steps.
- `utility_weighting`: re-weighting of the different terms in the local utility.



## Results analysis

### Decision-makers comparison

We compare the performance of the different decision-makers, and first observe how they affect the net-consumption
trajectory of an individual building and the whole district.

<figure>
<img src="/"  width="900" 
alt="."/>
</figure>

> A comparison of the decision-makers. The net consumption of a **single building** is shown,
  and the actions are taken by the different decision-makers.

<figure>
<img src="/"  width="900" 
alt="."/>
</figure>

> A comparison of the decision-makers. The net consumption of the **whole district** (sum of all building's) is shown,
  and the actions are taken by the different decision-makers: No-op, local Rule-Based,local RB with global RB,
  planners with (or w/o) last global RB. 
 
Utilities for these examples:
| Decision-maker | Utility (total) | Price cost | Emission cost | Grid cost |
|----------------|-----------------|------------|---------------|-----------|
| No-op          | 1.000           | 1.000      | 1.000         | 1.000     |
| Local RB       | 0.000           | 0.000      | 0.000         | 0.000     |
| Local RB + global RB | 0.000           | 0.000      | 0.000         | 0.000     |
| Local planner  | 0.000           | 0.000      | 0.000         | 0.000     |
| Local planner + global RB | 0.000           | 0.000      | 0.000         | 0.000     |


### Planner

#### Depth-selective search
`search_depths = {[0,1], [0,1,2,3,4]  [0,1,2,3,8]}`

#### utility weighting
`utility_weighting = {[1, 1, 1, 1], [1, 0, 0,0], [0, 1, 0, 0], [0, 0, 1, 1]}`

#### agents order shuffling
`random_order = {True, False}`

### Rule-based

w/o agents ramdomization.

| last_agent_type| last buliding (4) net consumption | whole district net consumption |
|--------------|-----------------------------------|-------------------------------|
| `RB-local`  | graph  [makes sense]              |                    graph |
| `RB-global` | graph                             |          graph[makes sense] |
 

### Summary

Synergizing imperfect planners with last RB agent is the best.




## Prerequisites
TODO: update requirements.txt or remove it.
Maybe it will also work without these, and it's enough to refer to the requirements.txt in the main repo.

| Library      | Version |
|--------------|---------|
| `python`     | 3.9.13  |
| `matplotlib` | 3.5.2   |
| `tqdm`       | 4.64.1  |

**plus** the CityLearn package itself, with its dependencies.
Note to get the 1.3.6 version, from:
[https://github.com/intelligent-environments-lab/CityLearn](https://github.com/intelligent-environments-lab/CityLearn)


## Files in the repository

TODO: complete!

| File/ folder name               | Purpose                                                          |
|---------------------------------|------------------------------------------------------------------|
| `main.py`                       | main script for locally evaluating the model on the trainig data |
| `utils.py`                      | utility functions for the main script                            |
| `evaluation_experiment.py`      | script for                                                       |
| `agents`                        | folder for the agents                                            |
| ├── `battery_model_rb_agent.py` |                                                                  |
| └── `controller.py`             |                                                                  |


## References
- **CityLearn**. [https://www.aicrowd.com/challenges/neurips-2022-citylearn-challenge](https://www.aicrowd.com/challenges/neurips-2022-citylearn-challenge)


## Contact

Gal Ness - [gness67@gmail.com](mailto:gness67@gmail.com)

Roy Elkabetz - [elkabetzroy@gmail.com](mailto:elkabetzroy@gmail.com)
