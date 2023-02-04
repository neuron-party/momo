# goin agane on the ppo
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools


class PPO3:
    def __init__(self, model, observation_space, action_space, **params):
        self.observation_space = observation_space
        self.action_space = action_space
        
        self.gamma = params['gamma']
        self.epochs = params['epochs']
        self.num_minibatches = params['num_minibatches']
        self.lambd = params['lambd']
        self.lr = params['lr']
        self.device = params['device']
        self.num_steps = params['num_steps']
        self.num_envs = params['num_envs']
        self.vf_clip = params['vf_clip']
        self.epsilon = params['epsilon']
        self.c1 = params['c1']
        self.c2 = params['c2']
        
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.num_updates = 0
        self.memory = []
        
    def remember(self, state, action, reward, next_state, done, log_probs_old, value_estimate_old):
        self.memory.append([state, action, reward, next_state, done, log_probs_old, value_estimate_old]) 
        # just add None None for the last two entries on the last step
        
    def get_action(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            pi, value = self.model(state)
            action = pi.sample()
            
            log_probs_old = pi.log_prob(action)
            value_estimate_old = value.view(-1, )
            
        return action.detach().cpu().numpy(), log_probs_old.detach().cpu().numpy(), value_estimate_old.detach().cpu().numpy()
    
    def _make_dataset(self):
        '''
        bootstrapping for advantage estimates
        delta <- R(s_t) + self.gamma * V(s_t+1) - V(s_t)
        '''
        assert len(self.memory) == self.num_steps
        dataset = []
        
        with torch.no_grad():
            next_state = torch.tensor(self.memory[-1][3], dtype=torch.float32, device=self.device)
            _, next_value_estimate = self.model(next_state)
            next_value_estimate = next_value_estimate.view(-1, )
            next_value_estimate = next_value_estimate.detach().cpu().numpy()
            
            advantage = 0
            for i in reversed(range(len(self.memory))): # does sending shit to the device like this take up more memory than using numpy arrays?
                if i == self.num_steps - 1:
                    next_values = next_value_estimate
                    terminal = 1 - self.memory[-1][4]
                else:
                    next_values = self.memory[i+1][6]
                    terminal = 1 - self.memory[i+1][4]
                delta = self.memory[i][2] + self.gamma * next_values * terminal - self.memory[i][6]
                advantage = delta + self.gamma * self.lambd * terminal * advantage
                value_target = self.memory[i][6] + advantage
                
                iterable = [[s, a, olp, ove, av, vt] for s, a, olp, ove, av, vt in 
                            zip(self.memory[i][0], self.memory[i][1], self.memory[i][5], self.memory[i][6], advantage, value_target)]
                
                # [state, action, old_log_probs, old_value_estimates, advantages, value_targets]
                dataset.append(iterable) 
        
        self.memory = []
        dataset = list(itertools.chain.from_iterable(dataset))
        assert len(dataset) == self.num_steps * self.num_envs
        return dataset
        
    def learn(self):
        dataset = self._make_dataset() # [256, 6, 64, ...]
        
        states = torch.tensor(np.stack([i[0] for i in dataset]), dtype=torch.float32, device=self.device) # [dataset_length, observation_shape]
        actions = torch.tensor([i[1] for i in dataset], dtype=torch.int64, device=self.device) # [dataset_length, ]
        old_log_probs = torch.tensor([i[2] for i in dataset], dtype=torch.float32, device=self.device) # [dataset_length, ]
        old_value_estimates = torch.tensor([i[3] for i in dataset], dtype=torch.float32, device=self.device) # [dataset_length, ]
        advantages = torch.tensor([i[4] for i in dataset], dtype=torch.float32, device=self.device) # [dataset_length, ]
        value_targets = torch.tensor([i[5] for i in dataset], dtype=torch.float32, device=self.device) # [dataset_length, ]
        
        # import pdb; pdb.set_trace()
        
        random_idxes = np.arange(len(dataset))
        
        for e in range(self.epochs):
            np.random.shuffle(random_idxes)
            minibatch_idxes = np.split(random_idxes, self.num_minibatches)
            
            for minibatch_idx in minibatch_idxes:
                s = states[minibatch_idx]
                a = actions[minibatch_idx]
                olp = old_log_probs[minibatch_idx]
                ove = old_value_estimates[minibatch_idx]
                adv = advantages[minibatch_idx]
                vt = value_targets[minibatch_idx]
                
                pi, value = self.model(s)
                value = value.view(-1, )
                log_probs = pi.log_prob(a)
                entropy = pi.entropy()
                
                loss = self._compute_loss(log_probs, olp, value, ove, adv, vt, entropy)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.num_updates += 1
                
    def _compute_loss(self, log_probs, old_log_probs, value_estimates, old_value_estimates, advantages, value_targets, entropy):
        probability_ratio = torch.exp(log_probs - old_log_probs)
        
        l_clip = torch.mean(
            torch.min(
                probability_ratio * advantages,
                torch.clamp(probability_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            )
        )
        if self.vf_clip:
            vf_unclipped = (value_estimates - value_targets) ** 2
            vf_clipped = old_value_estimates + torch.clamp(
                value_estimates - old_value_estimates, -self.vf_clip, self.vf_clip
            )
            vf_clipped = (vf_clipped - value_targets) ** 2
            vf = torch.mean(torch.max(vf_unclipped, vf_clipped))
        else:
            vf = (value_estimates - value_targets) ** 2
            vf = torch.mean(vf)
        
        entropy = torch.mean(entropy)
        
        loss = -l_clip + self.c1 * vf - self.c2 * entropy
        return loss