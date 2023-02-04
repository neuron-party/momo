import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools


class PPO:
    def __init__(self, model, observation_space, action_space, **params):
        self.observation_space = observation_space
        self.action_space = action_space

        self.device = params['device']
        self.gamma = params['gamma']
        self.lambd = params['lambd']
        self.epsilon = params['epsilon']
        self.num_envs = params['num_envs']
        self.num_epochs = params['num_epochs']
        self.clip_eps_vf = params['clip_eps_vf']
        self.update_interval = params['update_interval']
        self.num_minibatches = params['num_minibatches']
        self.c1 = params['c1']
        self.c2 = params['c2']
        self.lr = params['lr']

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.episodes = [[] for i in range(self.num_envs)]
        self.memory = []
        self.n_updates = 0

    def remember(self, state, action, reward, next_state, done):
        for i, (s, a, r, ns, d) in enumerate(zip(state, action, reward, next_state, done)):
            self.episodes[i].append([s, a, r, ns, d])
            if d: # episode terminated
                self.memory.append(self.episodes[i])
                self.episodes[i] = []

        self._make_dataset_if_ready()

    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pi, _ = self.model(state)
            actions = pi.sample()
        return actions.detach().cpu().numpy()

    def _make_dataset_if_ready(self):
        dataset_size = sum(len(episode) for episode in self.memory)
        if dataset_size >= self.update_interval:
            dataset = []
            for episode in self.memory:
                episode = self._kek(episode)
                dataset.append(episode)

        self.memory = []
        dataset = list(itertools.chain.from_iterable(dataset)) # whoever came up with this is a god
        self.learn(dataset)

    def _kek(self, episode):
        processed_episode = []
        advantage = 0 # advantage is computed with monte carlo?
        for mdp in reversed(episode):
            with torch.no_grad():
            states = torch.tensor(mdp[0], dtype=torch.float32, device=self.device)
            # import pdb; pdb.set_trace()
            actions = torch.tensor(mdp[1], dtype=torch.int64, device=self.device).view(-1, )
            rewards = torch.tensor(mdp[2], dtype=torch.float32, device=self.device).view(-1, )
            next_states = torch.tensor(mdp[3], dtype=torch.float32, device=self.device)
            dones = torch.tensor(mdp[4], dtype=torch.float32, device=self.device).view(-1, )

            pi_old, vs_pred_old = self.model(states.unsqueeze(0))
            log_probs_old = pi_old.log_prob(actions)

            _, next_vs_pred = self.model(next_states.unsqueeze(0))

            td_error = rewards + self.gamma * (1 - dones) * next_vs_pred - vs_pred_old
            advantage = td_error + self.gamma * self.lambd * advantage
            v_target = vs_pred_old + advantage

        processed_episode.append([states, actions, log_probs_old, vs_pred_old, advantage, v_target])

        return processed_episode

    def learn(self, dataset):
        batch_size = len(dataset) // self.num_minibatches
        for batch in _yield_minibatches(dataset, batch_size, self.num_epochs):
            states = torch.stack([i[0] for i in batch]).to(self.device) # [batch_size, state_dimensions]
            actions = torch.tensor([i[1] for i in batch], dtype=torch.int64, device=self.device) # [batch_size]
            log_probs_old = torch.tensor([i[2] for i in batch], dtype=torch.float32, device=self.device) # [batch_size]
            vs_pred_old = torch.tensor([i[3] for i in batch], dtype=torch.float32, device=self.device) # [batch_size]
            advantages = torch.tensor([i[4] for i in batch], dtype=torch.float32, device=self.device) # [batch_size]
            v_target = torch.tensor([i[5] for i in batch], dtype=torch.float32, device=self.device) # [batch_size] fix all these stupid ass reshapings

      # import pdb; pdb.set_trace()
      
            pi, value = self.model(states)
            value = value.view(-1, )
            log_probs = pi.log_prob(actions)
            entropy = pi.entropy()

            loss = self.compute_loss(log_probs, log_probs_old, value, vs_pred_old, v_target, advantages, entropy)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.n_updates += 1

    def compute_loss(self, log_probs, log_probs_old, value, vs_pred_old, v_target, advantages, entropy):
        probability_ratio = torch.exp(log_probs - log_probs_old)
        L_CLIP = torch.mean(
        torch.min(
                probability_ratio * advantages,
                torch.clamp(probability_ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            )
        )
        if self.clip_eps_vf:
        clipped_vs_pred = elementwise_clip(
            value, vs_pred_old - self.clip_eps_vf, vs_pred_old + self.clip_eps_vf
        )
        L_VF = torch.mean(
            torch.max(
              F.mse_loss(value, v_target, reduction='none'),
              F.mse_loss(clipped_vs_pred, v_target, reduction='none')
            )
        )
        else:
        L_VF = F.mse_loss(value, v_target)

        S = torch.mean(entropy)
        loss = -L_CLIP + self.c1 * L_VF - self.c2 * S
        return loss