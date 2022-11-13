# CityLearn Multi-Agent Smart-Grid
 
This repository contains the code for our implementation of solution for the CityLearn challenge.
The imposed problem is multi-agent reinforcement learning, where the agents are the buildings in the smart grid.
The goal is to minimize a specified measure of the net energy consumption of the buildings in the grid, 
parts of which are global to the whole district and parts of which are local to each building.

## luren ipsum from copilot:
Here we offer a solution based on Dikestra's algorithm, which is a graph-based algorithm for finding the 
optimal path in the action graph of the buildings. The action graph is a graph where the nodes are the
possible actions of the buildings and the edges are the transitions between the actions. The algorithm
is implemented in the `dikestra.py` file. The `main.py` file contains the code for the training of the
agents and the `agent.py` file contains the code for the agents themselves. The `utils.py` file contains
the code for the environment and the `plot.py` file contains the code for plotting the results.
