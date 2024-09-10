import numpy as np
import random
from bitarray import frozenbitarray, bitarray
from bitarray.util import int2ba

class EnemyBase() :

    def __init__(self, config : dict) :
        
        self.grid_size = config['grid_size']
        grid_area = self.grid_size[0]*self.grid_size[1]

        self.plane = np.array(config['plane_dim'], dtype=np.int8)
        self.actions = range(grid_area)
        self.state_space = set()
        self.reward = config["reward"]

        # Generate State Space
        for i in range(4) :
            t_plane = np.rot90(self.plane, i)
            P, Q = t_plane.shape

            combinations = []
            pos_one = np.where(t_plane.flatten() == 1)[0]
            num_combination = 2**len(pos_one)

            for j in range(num_combination) :
                bits = int2ba(j, pos_one.shape[0])
                n_com = t_plane.flatten()
                n_com[pos_one] = bits.tolist()
                combinations.append(n_com.reshape((P, Q)))
            
            for comb in combinations :
                for y in range(self.grid_size[0] - P + 1) :
                    for x in range(self.grid_size[1] - Q + 1) :
                        n_grid = np.zeros(self.grid_size, dtype=np.int8)
                        n_grid[y : y + P, x : x + Q] = comb
                        grid_bit = frozenbitarray(n_grid.flatten().tolist())
                        self.state_space.add(grid_bit)
        self.state_space = list(self.state_space)

    def reset(self) :
        self.state = np.zeros(self.grid_size, dtype=np.int8)
        random_bit = random.randint(0, 3)
        t_plane = np.rot90(self.plane, random_bit)
        P, Q = t_plane.shape

        x = random.randint(0, self.grid_size[1] - Q)
        y = random.randint(0, self.grid_size[0] - P)

        self.state[y : y + P, x : x + Q] = t_plane
        self.state = frozenbitarray(self.state.ravel().tolist())
        return self.state

    def step(self, state : frozenbitarray, action : int) :
        done = False
        reward = self.reward['hit'] if(state[action] == 1) else self.reward['miss']
        new_state = bitarray(state)
        new_state[action] = 0
        if(reward > 0 and not new_state.any()) :
            done = True
            reward = self.reward['destroy']
        self.state = frozenbitarray(new_state)
        return self.state, reward, done










