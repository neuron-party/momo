import random
import numpy as np


def state_to_index(state, bins):
    assert len(state) == len(bins)
    index = []
    for i in range(len(state)):
        idx = np.digitize(state[i], bins[i]) - 1
        index.append(idx)
    return tuple(index)


def get_total_discounted_rewards(rewards, gamma):
    gammas = np.array([gamma ** i for i in range(len(rewards))])
    tdr = sum(rewards * gammas)
    return tdr


def discretize_continuous_environment(env, samples, bin_size):
    states = []
    for i in range(samples):
        state = env.reset()
        done = False
        while not done:
            action = np.random.randint(0, env.action_space.n)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            states.append(state)
            
    states = np.array(states)
    bins = []
    for i in range(states.shape[-1]):
        bins.append(
            np.linspace(np.min(states[:, i]), np.max(states[:, i]), bin_size)
        )
    return np.array(bins)