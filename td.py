# 1-step td tabular q learning
import numpy as np
from utils import *


class TabularQ_TD:
    def __init__(self, action_space, observation_space, bins, q_table, **params):
        self.action_space = action_space
        self.observation_space = observation_space
        
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.min_epsilon = params['min_epsilon']
        self.lr = params['lr']
        self.gamma = params['gamma']
        
        self.bins = bins # (n, bin_size) array where n is the observation space
        self.q_table = q_table # (bin_size,...,bin_size, action_space) array where there are n bin_size dimensions + 1 action dimension
        
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_space.n)
        else:
            index = state_to_index(state, self.bins)
            q_values = self.q_table[index]
            action = np.argmax(q_values)
        return action
    
    def learn(self, state, action, reward, next_state, done):
        state_index, next_state_index = state_to_index(state, self.bins), state_to_index(next_state, self.bins)
        state_index = state_index + tuple([action])
        self.q_table[state_index] = self.q_table[state_index] + self.lr * (reward + self.gamma * (1- done) * np.max(self.q_table[next_state_index]) - self.q_table[state_index])
        
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)