# dynaQ tabular q learning for integrated learning and planning
import numpy as np
import random
from utils import *


class DynaQ: # need prioritized sweeping as well
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
        self.model = {}
        
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
        self.q_table[state_index] = self.q_table[state_index] + self.lr * (reward + self.gamma * (1 - done) * np.max(self.q_table[next_state_index]) - self.q_table[state_index])
        self.model[state_index] = [reward, next_state_index]
        
        for i in range(self.n):
            # S <- random previously observed state
            # A <- random action previously taken in S
            # R, S' <- Model(S, A)
            # Q(S, A) <- Q(S, A) + a[R + max_a Q(S', a) - Q(S, A)]
            state_action_pair = random.choice(list(self.model.keys()))
            reward, next_state = self.model[state_action_pair]
            self.q_table[state_action_pair] = self.q_table[state_action_pair] + self.lr * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state_action_pair])
            
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)