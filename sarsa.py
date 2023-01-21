import numpy as np
from utils import *


class SARSA:
    def __init__(self, action_space, observation_space, bins, q_table, **params):
        self.action_space = action_space
        self.observation_space = observation_space
        
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.min_epsilon = params['min_epsilon']
        self.lr = params['lr']
        self.gamma = params['gamma']
        
        self.bins = bins 
        self.q_table = q_table 
        
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
        # on policy learning, we derive the action the next state by following the current policy (epsilon-greedy)
        a_prime = self.get_action(next_state)
        next_state_index = next_state_index + tuple([a_prime])
        self.q_table[state_index] = self.q_table[state_index] + self.lr * (reward + self.gamma * (1- done) * self.q_table[next_state_index] - self.q_table[state_index])
        
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)