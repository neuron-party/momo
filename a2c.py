import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class A2C:
    '''
    1-step Actor Critic
    '''
    def __init__(self, model, observation_space, action_space, **params):
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.gamma = params['gamma']
        self.device = params['device']
        self.lr = params['lr']
        
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, -1) # 1 for batch size since we're doing 1-step
        with torch.no_grad():
            pi, _ = self.model(state)
            action = pi.sample()
        return action.detach().cpu().numpy()
    
    def learn(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        action = torch.tensor(action, dtype=torch.int64, device=self.device)
        reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        next_state = torch.tensor(state, dtype=torch.float32, device=self.device).view(1, -1)
        
        _, next_value_estimate = self.model(next_state)
        pi, value = self.model(state)
        
        advantage = reward + self.gamma * next_value_estimate * (1 - done) - value
        pi_log_probs = pi.log_prob(action)
        loss = -(advantage.detach() * pi_log_probs) # negative of the object function for gradient descent
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()