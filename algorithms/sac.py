class SAC:
    '''
    Soft Actor Critic
    
    Value Function objective function: [V(s) - (Q(s, a) - log(pi(a|s))]^2; "soft" value function
    Q Function objective function: [Q(s, a) - (r + γ * V(s_t+1))]^2; "soft" q function
    
    Policy objective function: D_KL [pi(x|s) | exp(Q(s, x))]
    which can reparameterized as pi(a|s) * (log(pi(a|s) - Q(s, a)))
    
    Qualitatively: 
    * the value objective function is the Euclidean distance between the soft value functions approximated by the value function and the q function 
      approximators respectively.
    * the q objective function is the standard Bellman residual using the soft value function instead of max_a Q(s', a')
    * the policy objective function is the KL divergence between the stochastic actor and the Q function approximator
    
    Alternative formulation with only the Q approximator:
        our Q function objective function would then be: [Q(s, a) - (r + γ * (Q(s', a') - log(pi(a'|s')))]^2
    '''
    def __init__(self, observation_space, action_space, model, **params):
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = params['lr']
        self.gamma = params['gamma']
        self.update_target_freq = params['update_target_freq']
        self.batch_size = params['batch_size']
        self.device = torch.device('cuda:0')
        
        self.replay = ReplayBuffer(
            int(1e5),
            observation_space,
            action_space,
            torch.device('cuda:0')
        )
        
        self.vf = VApproximator(4, 2).to(self.device)
        self.vf_approximator_target = VApproximator(4, 2).to(self.device)
        self.q_approximator_1 = QApproximator(4, 2).to(self.device)
        self.q_approximator_2 = QApproximator(4, 2).to(self.device)
        self.policy = Actor(4, 2).to(self.device)
        
        self.vf_approximator_optimizer = torch.optim.Adam(self.vf.parameters(), lr=self.lr)
        self.q_approximator_1_optimizer = torch.optim.Adam(self.q_approximator_1.parameters(), lr=self.lr)
        self.q_approximator_2_optimizer = torch.optim.Adam(self.q_approximator_2.parameters(), lr=self.lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)
        
        # self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
        # self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        # self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
        self.alpha = 0.05
        
        self.n_updates = 0
        
    def remember(self, state, action, reward, next_state, done, info):
        self.replay.add(state, next_state, action, reward, done, [info])
        
    def get_action(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        with torch.no_grad():
            pi = self.policy(state)
            action = pi.sample()
        return action.detach().cpu().numpy()
    
    def learn(self):
        '''
        First time using stable baselines' replay buffer; 
            data.observations = states
            data.next_observations = next_states
            data.actions = actions
            data.rewards = rewards
            data.dones = dones
            
        The evaluation of the Q and V functions are at the actions chosen by the agent throughout the trajectory, but the targets for both are selected by 
        the stochastic actor at the time of update
        '''
        data = self.replay.sample(self.batch_size)
        # value function loss
        q_approximation_1, q_approximation_2 = self.q_approximator_1(data.observations), self.q_approximator_2(data.observations)
        q_approximation = torch.min(q_approximation_1, q_approximation_2)
        pi = self.policy(data.observations)
        stochastic_actions = pi.sample().reshape(-1, 1)
        log_probs = pi.log_prob(stochastic_actions.view(-1, ))
        log_probs = log_probs.reshape(-1, 1)
        
        # import pdb; pdb.set_trace()
        vf_target = torch.gather(q_approximation, 1, stochastic_actions) - (log_probs * self.alpha) # temperature scaling for the log probs?
        vf_approximation = self.vf(data.observations)
        
        vf_loss = F.mse_loss(vf_approximation, vf_target.detach())
        # vf_loss = torch.mean((vf_approximation - vf_target.detach()) ** 2)
        
        self.vf_approximator_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_approximator_optimizer.step()
        
        # q function loss
        with torch.no_grad():
            vf_approximation = self.vf_approximator_target(data.next_observations)
            q_target = data.rewards + self.gamma * vf_approximation * (1 - data.dones)
        
        q_approximation_1, q_approximation_2 = self.q_approximator_1(data.observations), self.q_approximator_2(data.observations)
        q_approximation_1 = torch.gather(q_approximation_1, 1, data.actions)
        q_approximation_2 = torch.gather(q_approximation_2, 1, data.actions)
        
        qf_loss_1 = F.mse_loss(q_approximation_1, q_target.detach())
        qf_loss_2 = F.mse_loss(q_approximation_2, q_target.detach())
        
        self.q_approximator_1_optimizer.zero_grad()
        qf_loss_1.backward()
        self.q_approximator_1_optimizer.step()
        self.q_approximator_2_optimizer.zero_grad()
        qf_loss_2.backward()
        self.q_approximator_2_optimizer.step()
        
        # policy loss
        policy_loss = self.alpha * log_probs - q_approximation.detach()
        policy_loss = torch.mean(policy_loss)
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.n_updates += 1
        if self.update_target_freq % self.n_updates == 0:
            self.vf_approximator_target.load_state_dict(self.vf.state_dict())