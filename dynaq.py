# dynaQ tabular q learning for integrated learning and planning
import numpy as np
import random
from utils.utils import *


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
        self.queue = []
        self.pqueue = []
        self.theta = params['theta']
        self.prioritized_sweeping = params['prioritized_sweeping']
        
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
        td_error = reward + self.gamma * (1 - done) * np.max(self.q_table[next_state_index]) - self.q_table[state_index]
        self.q_table[state_index] = self.q_table[state_index] + self.lr * td_error
        self.model[state_index] = [reward, next_state_index]
        
        P = abs(td_error)
        if P > self.theta:
            priority = len(self.pqueue) - sum(P > np.array(self.pqueue))
            self.queue.insert(priority, state_index)
            self.pqueue.insert(priority, P)
        
        for i in range(self.n):
            if self.prioritized_sweeping:
                # S, A <- first(PQueue)
                # R, S' <- Model(S, A)
                # Q(S, A) <- Q(S, A) + a[R + max_a Q(S', a) - Q(S, A)]
                # Loop for all S_bar, A_bar predicted to lead to S:
                    # R_bar <- predicted rewards for S_bar, A_bar, S
                    # P <- |R_bar + gamma * max_a Q(S, a) - Q(S_bar, A_bar)|
                    # if P > theta, insert S_bar, A_bar into PQueue with priority P
                try:
                    state_action_pair = self.queue.pop(0)
                    self.pqueue.pop(0)
                except:
                    break
                reward, next_state = self.model[state_action_pair]
                self.q_table[state_action_pair] = self.q_table[state_action_pair] + self.lr * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state_action_pair])
                    
                S = state_action_pair[:len(state_action_pair) - 1]
                S_A_bar = [state for state, next_states in self.model.items() if next_states[1] == S]
                for state_action_pair in S_A_bar:
                    R_bar, _ = self.model[state_action_pair]
                    P = abs(R_bar + self.gamma * np.max(self.q_table[S]) - self.q_table[state_action_pair])
                    if P > self.theta:
                        priority = len(self.pqueue) - sum(P > np.array(self.pqueue))
                        self.queue.insert(priority, state_action_pair)
                        self.pqueue.insert(priority, P)
            else:
                # S <- random previously observed state
                # A <- random action previously taken in S
                # R, S' <- Model(S, A)
                # Q(S, A) <- Q(S, A) + a[R + max_a Q(S', a) - Q(S, A)]
                state_action_pair = random.choice(list(self.model.keys()))
                reward, next_state = self.model[state_action_pair]
                self.q_table[state_action_pair] = self.q_table[state_action_pair] + self.lr * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state_action_pair])
        
        assert len(self.queue) == len(self.pqueue)
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)