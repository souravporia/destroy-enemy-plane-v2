# Destroy Enemy-Plane in a Base [Version 2]

## Project Overview
"Destroy Enemy-Plane in a Base [Version 2]" is a reinforcement learning (RL) project designed to tackle a grid-based optimization problem. The enemy base is represented by an NxN grid where an enemy plane is randomly placed with a specific orientation. The goal is to destroy the plane by targeting missiles at each square with the minimum possible number of launches. The task involves building an RL agent that can learn to solve this problem efficiently using algorithms like Policy Iteration (PI) and Monte Carlo (MC) Learning.

## Features
- **Dynamic Environment:** Supports any NxN grid size and various plane shapes.
- **Configurable:** Adjust environment dynamics and agent behaviors via the configuration file.
- **Visualization:** Heatmaps for state and reward visualizations.
- **Optimized Implementation:** Uses efficient data structures for faster convergence.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r /path/to/requirements.txt
```
## Configuration

The configuration file is stored as config/config.json. You can modify this file to change various parameters related to the environment and agent behaviors:

```json
{
    "environment": {
        "grid_size": [10, 10],
        "plane_dim" :  [[0, 0, 1, 0],
                        [1, 1, 1, 1],
                        [0, 0, 1, 0]],
        "reward" : {
            "destroy" : 10,
            "hit" : 1,
            "miss" : -1
        }
    },

    "agent": {
        "gamma": 0.9,
        "max_missiles" : 10,
        "algorithm" : "mc",
        "policy_iteration" : {
            "theta" : 0.5
        },
        "mc" : {
            "episode_length" : 50,
            "epsilon" : 0.2,
            "number_of_episodes" : 20000
        }
    },

    "visualization" : {
        "mc_episodes_inverval" : 10000
    }
}
```

### Configuration Parameters

-   **Environment Dynamics:**
    
    -   `grid_size`: The size of the grid (NxN).
    -   `plane_dim`: The shape and dimensions of the plane within the grid.
    -   `reward`: Reward structure for destroying, hitting, or missing the plane.
-   **Agent Behaviors:**
    
    -   `gamma`: Discount factor for future rewards.
    -   `max_missiles`: Maximum number of missile launches allowed.
    -   `algorithm`: Choice of RL algorithm (`mc` for Monte Carlo, `pi` for Policy Iteration).
    -   **Policy Iteration (PI) Specific:**
        -   `theta`: Convergence threshold for policy improvement.
    -   **Monte Carlo (MC) Specific:**
        -   `episode_length`: Maximum length of an episode.
        -   `epsilon`: Exploration factor.
        -   `number_of_episodes`: Number of episodes for training.
-   **Visualization:**
    
    -   `mc_episodes_interval`: Interval of episodes after which visualizations are updated for Monte Carlo.

### Note :
    The solution supports two algorithms: mc (Monte Carlo) and pi (Policy Iteration).
    Convergence in Monte Carlo will take longer without sufficient exploration (epsilon). The current configuration uses a 10x10 grid with a specific plane structure and near-optimal values.


## Running the Program

To execute the program, run **main.py** from the project root directory

```bash
python /path/to/main.py
```