import random
import numpy as np

class MissAgent:
    def __init__(self, env, config):
        self.env = env
        self.gamma = config['gamma']
        self.pi = config['policy_iteration']
        self.mc = config['mc']
        self.missile_count = config['max_missiles']
        self.policy = {state : random.choice(self.env.actions) for state in self.env.state_space}
        self.V = {state : 0 for state in self.env.state_space}

    
    def policy_evaluation(self, theta):
        while True :
            delta = 0
            for state in self.env.state_space :
                if not state.any():
                    continue

                v = self.V[state]
                t_state = state

                action = self.policy[t_state]
    
                next_state, r, _ = self.env.step(t_state, action)
                self.V[state] = r + self.gamma*self.V[next_state]
                delta = max(delta, abs(self.V[state]- v))

            if(delta < theta) :
                break

        return self.V

    def policy_improvement(self):
        policy_stable = True

        for state in self.env.state_space :
            if not state.any():
                continue
            new_actions = dict()
            old_action = self.policy[state]
            
            for action in self.env.actions :
                new_state, r, _ = self.env.step(state, action)
                total_value = r + self.gamma*self.V[new_state]
                new_actions[action] = total_value
            
            best_action = max(new_actions, key=new_actions.get)
            self.policy[state] = best_action

            if old_action != best_action:
                policy_stable = False
            
        return policy_stable
    
    def policy_iteration(self):
        while True:
            self.policy_evaluation(self.pi['theta'])
            if self.policy_improvement() :
                break
        return self.policy

    def monte_carlo_es(self, number_of_episodes):
        for _ in range(number_of_episodes) :
            curr_state = self.env.reset()
            curr_state = random.choice(self.env.state_space)
            action = random.choice(self.env.actions)
            sweeps = dict()
            rewards = []
            states_actions = []

            for _ in range(self.mc['episode_length']):
                new_state, reward, done = self.env.step(curr_state, action)
                rewards.append(reward)
                states_actions.append((curr_state, action))
                
                if done:
                    break

                curr_state = new_state
                if random.random() < self.mc['epsilon'] :
                    action = random.choice(self.env.actions)
                else :
                    action = self.policy[curr_state]

            G = 0
            for state_action in reversed(states_actions):
                G = self.gamma * G + rewards.pop()
                sweeps.setdefault(state_action, []).append(G)
            
            new_policy = {}
            for (state, action), return_list in sweeps.items():
                avg_return = sum(return_list) / len(return_list)
                
                if state not in new_policy or avg_return > new_policy[state][1] :
                    new_policy[state] = (action, avg_return)
        
            for state, (action, _) in new_policy.items() :
                self.policy[state] = action

        return self.policy
    
    def action(self) :
        state = self.env.reset()
        state_grids = []
        reward_grids = []

        reward_grid = np.zeros(self.env.grid_size, dtype=np.int8)
        for _ in range(self.missile_count) :
            action = self.policy[self.env.state]
            state = self.env.state
            
            _, reward, done = self.env.step(state, action)
            
            reward_grid.ravel()[action] += reward
            r_grid = reward_grid.copy()
            
            state_grid = np.reshape(np.array(state.tolist(), dtype=np.int8), self.env.grid_size)

            state_grids.append(state_grid)
            reward_grids.append(r_grid)

            if done :
                state_grids.append(np.reshape(np.array(self.env.state.tolist(), dtype=np.int8), self.env.grid_size))
                reward_grids.append(r_grid)
                break
            
        return state_grids, reward_grids
