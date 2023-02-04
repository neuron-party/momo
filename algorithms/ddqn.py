import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from utils.replay import ExperienceReplay


class DDQN:
    def __init__(self, model, state_space, action_space, **params):
        self.state_space = state_space
        self.action_space = action_space
        
        self.lr = params['lr']
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.min_epsilon = params['min_epsilon']
        self.gamma = params['gamma']
        self.optimizer_update_freq = params['optimizer_update_freq']
        
        self.batch_size = params['batch_size']
        self.device = params['device']
        self.criterion = nn.MSELoss()
        
        self.replay = ExperienceReplay(params['replay_size'])
        self.network = model.to(self.device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.optimizer_steps = 0
        
    def remember(self, state, action, reward, next_state, done):
        self.replay.add((state, action, reward, next_state, done))
        
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.action_space)
        else:
            state = torch.tensor(state, dtype=torch.float32, device=self.device)
            state_values = self.network(state)
            # import pdb; pdb.set_trace()
            action = int(torch.argmax(state_values))
            
        return action
    
    def learn(self):
        states, actions, rewards, next_states, dones = self.replay.random_sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).reshape(-1, 1)
        rewards = torch.tensor(rewards, dtype=torch.int64, device=self.device).reshape(-1, 1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.int64, device=self.device).reshape(-1, 1)
        
        q_predicted = self.network(states)
        q_predicted = torch.gather(q_predicted, 1, actions)
        q_actual = self.target_network(next_states)
        optimal_actions = torch.argmax(q_actual, dim=1).reshape(-1, 1)
        q_actual = torch.gather(q_actual, 1, optimal_actions)
        q_actual = rewards + self.gamma * (1 - dones) * q_actual
        
        
        loss = self.criterion(q_predicted, q_actual)
        self.optimizer.zero_grad()
        loss.backward()
        
        # for param in self.network.parameters():
        #     param.grad.data.clamp_(-1, 1)
                
        self.optimizer.step()
        
        self.optimizer_steps += 1
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        if self.optimizer_steps % self.optimizer_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())