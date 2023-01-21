# first/every visit monte carlo tabular q learning
import numpy as np
from utils import *


class TabularQ_MC:
    def __init__(self, action_space, observation_space, bins, q_table, **params):
        self.action_space = action_space
        self.observation_space = observation_space
        
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.min_epsilon = params['min_epsilon']
        self.lr = params['lr']
        self.gamma = params['gamma']
        self.first_visit = params['first_visit']
        
        self.bins = bins
        self.q_table = q_table
        
        self.memory = []
        
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_space.n)
        else:
            index = state_to_index(state, self.bins)
            q_values = self.q_table[index]
            action = np.argmax(q_values)
        return action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])
        
    def learn(self):
        if self.first_visit: # first visit monte carlo
            visited = set()
            for idx, i in enumerate(self.memory):
                state, action = i[0], i[1]
                state_index = state_to_index(state, self.bins) + tuple([action])
                if state_index not in visited:
                    rewards = np.array(self.memory)[idx:, 2]
                    G = get_total_discounted_rewards(rewards, self.gamma)
                    
                    self.q_table[state_index] = self.q_table[state_index] + self.lr * (G - self.q_table[state_index])
                    visited.add(state_index)
                    
        else: # every-visit monte carlo
            update_dict = {}
            for idx, i in enumerate(self.memory):
                state, action = i[0], i[1]
                state_index = state_to_index(state, self.bins) + tuple([action])
                rewards = np.array(self.memory)[idx:, 2]
                G = get_total_discounted_rewards(rewards, self.gamma)
                
                if state_index not in update_dict:
                    update_dict[state_index] = [1, G]
                else:
                    update_dict[state_index][0] += 1
                    update_dict[state_index][1] += G
                    
            for state, info in update_dict.items():
                G = info[1] / info[0] # averaged rewards over all visits to this state-action pair
                self.q_table[state] = self.q_table[state] + self.lr * (G - self.q_table[state])
                
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        self.memory = []