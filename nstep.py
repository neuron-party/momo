# n-step td tabular q learning
import numpy as np
import random
from utils import *


class nStepTD:
    def __init__(self, action_space, observation_space, bins, q_table, **params):
        self.action_space = action_space
        self.observation_space = observation_space
        
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.min_epsilon = params['min_epsilon']
        self.lr = params['lr']
        self.gamma = params['gamma']
        self.n = params['n']
        
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
        initial_state, initial_action = self.memory[0][0], self.memory[0][1]
        state_index = state_to_index(initial_state, self.bins)
        state_index = state_index + tuple([initial_action])
        
        n_step_next_state = self.memory[-1][3]
        n_step_next_state = state_to_index(n_step_next_state, self.bins)
        
        rewards = np.array(self.memory)[:, 2]
        G = get_total_discounted_rewards(rewards, self.gamma) # monte carlo total discounted rewrads
        G = G + np.max(self.q_table[n_step_next_state]) # n-step temporal difference error
        
        self.q_table[state_index] = self.q_table[state_index] + self.lr * (G - self.q_table[state_index])
        
        self.memory = []
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)