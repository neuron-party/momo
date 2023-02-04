import os
import json
from easydict import EasyDict as edict
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import wandb
import numpy as np
import gym
import procgen

from algorithms.ppo4 import PPO4
from networks import ImpalaCNN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-config", type=str, default='../configs/ppo4.json')
    parser.add_argument("--logging-config", type=str, default='../configs/logging.json')
    parser.add_argument("--env-config", type=str, default='../configs/procgen_vec.json')
    args = parser.parse_args()
    return args

def main(args):
    with open(args.agent_config, 'r') as f:
        agent_config = edict(json.load(f))
    with open(args.logging_config, 'r') as f:
        logging_config = edict(json.load(f))
    with open(args.env_config, 'r') as f:
        env_config = edict(json.load(f))

        
    # logging setup
    if logging_config.log:
        wandb.init(
            project=logging_config.project,
            sync_tensorboard=True,
            name=logging_config.name,
            monitor_gym=True
        )
    writer = SummaryWriter(f'runs/{logging_config.name}')

    
    # environment setup
    env = procgen.ProcgenEnv(
        num_envs=agent_config.num_envs, 
        env_name=env_config.env_name, 
        num_levels=env_config.num_levels, 
        start_level=env_config.start_level,
        distribution_mode=env_config.distribution_mode
    )
    env = gym.wrappers.TransformObservation(env, lambda obs: obs['rgb'])
    env.single_action_space = env.action_space
    env.single_observation_space = env.observation_space['rgb']
    env.is_vector_env = True
    env = gym.wrappers.RecordEpisodeStatistics(env)
    
    # eval env (just modify this part myself)
    eval_env = procgen.ProcgenEnv(
        num_envs=1,
        env_name=env_config.env_name,
        num_levels=500,
        start_level=env_config.start_level + env_config.num_levels,
        distribution_mode=env_config.distribution_mode
    )
    eval_env = gym.wrappers.TransformObservation(eval_env, lambda obs: obs['rgb'])
    eval_env.single_action_space = eval_env.action_space
    eval_env.single_observation_space = eval_env.observation_space['rgb']
    eval_env.is_vector_env = True # preserve the batch dimension for easy of inference
    eval_env = gym.wrappers.RecordEpisodeStatistics(eval_env)
    
    
    # agent setup
    model = ImpalaCNN(env.single_observation_space, env.single_action_space.n)
    agent = PPO4(env.single_observation_space, env.single_action_space, model, **agent_config)
    
    
    # training loop
    state = env.reset()
    eval_state = eval_env.reset()
    total_global_steps = 0
    max_global_steps = env_config.max_global_steps
    
    while total_global_steps < max_global_steps:
        # training curve
        for i in range(agent_config.num_steps):
            action, log_probs, value_estimate = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done, log_probs, value_estimate, i)
            state = next_state
            
            total_global_steps += agent_config.num_envs
            for item in info:
                if 'episode' in item.keys():
                    writer.add_scalar('charts/episodic_return_train', item['episode']['r'], total_global_steps)
        
        agent.learn(state)
        torch.save({'model': agent.model.state_dict(), 'optimizer': agent.optimizer.state_dict()}, agent.save_dir + '.pth')
        
        # testing curve; play through one episode and record the return
        eval_done = False
        while not eval_done:
            eval_action, _, _ = agent.get_action(eval_state)
            eval_next_state, eval_reward, eval_done, eval_info = eval_env.step(eval_action)
            eval_state = eval_next_state
            
            for item in eval_info:
                if 'episode' in item.keys():
                    writer.add_scalar('charts/episodic_return_test', item['episode']['r'], total_global_steps)
        
    # final save
    torch.save({'model': agent.model.state_dict(), 'optimizer': agent.optimizer.state_dict()}, agent.save_dir + '_final.pth')
    
if __name__ == '__main__':
    args = parse_args()
    main(args)