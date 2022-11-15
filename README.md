# CityLearn Multi-Agent Smart-Grid Smart-Tree Smart-Search
 
> This repository contains the code for our implementation of solution for the 2022 CityLearn challenge.

## Introduction

The imposed problem is multi-agent scenario, where the agents are the buildings in the smart grid.
There is a battery in each building, which can be used to store energy, and a solar panel, which produces energy.
Each buliding has its own energy consumption and production, and the goal is to minimize the utility,
which is a specified measure of the net energy consumption of the buildings in the grid, 
parts of which are global to the whole district and parts of which are local to each building.
The action space is the amount of energy to be stored in the battery, and the observation space is the
energy consumption and production of the building, as well as additional global parameters.

The crux of the problem is that:
- The action affects the next step, so the net consumption has to be predicted.
- The utility involves global parts, so the optimal action for one building depends on the actions of the other buildings.
- The natural periodicity of the net consumption is 24 hours, which even for moderate branching factors is a lot of states to consider (e.g., `5**24=6e17`).

## Solution
We use a tree search algorithm, which is a modified Dijkstra's algorithm, to find the optimal action for each building.

### Special tricks and spices

#### Battery model

#### Hierarchical control scheme

#### Local utility estimation
The utility is a function of the net consumption, which is only evaluated at the end of the year (episode).
However, the predictions and actions are made at each step, so we need to estimate the utility at each step.
For this purpose, we use an instantaneous utility estimation, which is an approximation of the utility function.

TODO: Explain each term in the utility estimation.

There is a couple of estimators.
The first one uses for a single building, independent of the other buildings.
The second one uses the net consumption of the whole district, using the actions of the previous agents and 
no-op's as estimations for the missing next buildings.

#### Adaptive depth search

### Alternative Role-based solution
A set of rules defines the next move for each build independently (locally), based on the next hour prediction.
The rules were defined to "flatten" the net consumption curve, and by this to minimize the utility:
- If the next hour production is higher than the consumption, the battery is charged by the extra amount.
- If the next hour consumption is higher than the production, the battery is discharged by the missing amount.

On top of that, the rules treat the cases where the battary is fully charged or fully discharged.
We also penalize the battery charge, in hours when the carbon intensity is below its median,
as in such times the utility for using the grid power is relatively lower.

The rules are defined in two cases, for a single building and for a group of buildings.
The essence is the same, just that for the later case, the input is the net consumption of the group.

Additional tuning was done to the rules, to minimize the utility for the training set, and the parameters for the
single and group rules were found to be different.
An important hyperparameter is thus the number of buildings which use the group rules.


## Analysis
<figure>
<img src="/"  width="900" 
alt="."/>
</figure>
>.

 
## Prerequisites
TODO: update requirements.txt or remove it

| Library      | Version |
|--------------|---------|
| `python`     | 3.9.13  |
| `numpy`      | 1.21.6  |
| `pytorch`    | 1.10.2  |
| `pandas`     | 1.3.5   |
| `matplotlib` | 3.5.2   |
| `tqdm`       | 4.64.1  |
| `pip`        | 22.2.2  |

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
- **CityLearn**. [https://citylearn.org/](https://citylearn.org/)


## Contact

Gal Ness - [gness67@gmail.com](mailto:gness67@gmail.com)

Roy Elkabetz - [elkabetzroy@gmail.com](mailto:elkabetzroy@gmail.com)