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

### Alternative simplified solutions

## Files in the repository

|File name       | Purpose                                                                                                         |
|-------------------|-----------------------------------------------------------------------------------------------------------------|
|`main.py`| main script for locally evaluating the model on the trainig data|