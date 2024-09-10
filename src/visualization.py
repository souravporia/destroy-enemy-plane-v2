import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Heatmap() :
    def __init__(self, env, agent) :
        self.env = env
        self.agent = agent
        self.state_grids = []
        self.reward_grids = []
        self.iteration_numbers = []

    def visualize_grid(self, state_grids, reward_grids, iteration_nums):
        fig, ax = plt.subplots(1, 2)
        count = 0

        def update(frame):
            nonlocal count
            if(frame >= iteration_nums[count][0]) :
                count = count + 1

            ax[0].cla()
            ax[1].cla()

            sns.heatmap(state_grids[frame], ax=ax[0], annot=True, cmap="viridis", cbar=False, square=True, linewidths=0.5, linecolor='black')
            sns.heatmap(reward_grids[frame], ax=ax[1], annot=True, cmap="magma", cbar=False, square=True, linewidths=0.5, linecolor='black')

            ax[0].set_title('State Map', fontsize=12)
            ax[1].set_title('Cummulative Reward Map', fontsize=12)
            fig.suptitle(f'After Iteration Number : {iteration_nums[count][1]}', fontsize=16)

        anim = FuncAnimation(fig, update, frames=len(state_grids), repeat=False, interval=500)

        plt.tight_layout()
        plt.show()

    

    def visualize_mc(self, total_episodes, episodes_inverval) :
        
        for i in range(0, total_episodes, episodes_inverval) :
            state_grid, reward_grid = self.agent.action()
            self.state_grids = self.state_grids + state_grid
            self.reward_grids = self.reward_grids + reward_grid
            self.iteration_numbers.append([len(self.state_grids), i])
            self.agent.monte_carlo_es(episodes_inverval)

        state_grid, reward_grid = self.agent.action()
        self.state_grids = self.state_grids + state_grid
        self.reward_grids = self.reward_grids + reward_grid
        self.iteration_numbers.append([len(self.state_grids), total_episodes])

    def visualize_policy_iteration(self, theta) :
        count = 0
        while True:
            state_grid, reward_grid = self.agent.action()
            self.state_grids = self.state_grids + state_grid
            self.reward_grids = self.reward_grids + reward_grid
            self.iteration_numbers.append([len(self.state_grids), count])
            
            self.agent.policy_evaluation(theta)
            count = count + 1

            if self.agent.policy_improvement() :
                break

        state_grid, reward_grid = self.agent.action()
        self.state_grids = self.state_grids + state_grid
        self.reward_grids = self.reward_grids + reward_grid
        self.iteration_numbers.append([len(self.state_grids), count])
    
    def show(self) :
        self.visualize_grid(self.state_grids, self.reward_grids, self.iteration_numbers)
