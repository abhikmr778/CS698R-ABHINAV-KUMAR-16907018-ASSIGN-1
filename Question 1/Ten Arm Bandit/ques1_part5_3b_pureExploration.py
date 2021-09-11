import gym
import custom_bandits
import numpy as np
import matplotlib.pyplot as plt
from agents import pureExploration


if __name__ == "__main__":
    sigma = 1
    SEED = 0
    maxEpisodes = 20
    env = gym.make('tenArmGaussian_bandits-v0', sigma_square=sigma, seed=SEED)
    env.reset()
    Q_estimates, action_history, optimal_action_history, reward_history, regret_history = pureExploration(env,maxEpisodes)
    print(f'--------------------SEED: {SEED}------------------------')
    print(f'True Q values: {env.q_value}')
    print(f'Final Q Estimates: {Q_estimates[-1,:]}')
    print(f'Action with highest q_value: {np.argmax(env.q_value)}')
    print(f'Q_val |Q[actionTaken]| actionTaken | reward')
    print('--------------------------------------------')
    for i in range(maxEpisodes-20, maxEpisodes):
        print(f'Q[{i}]| {Q_estimates[i][int(action_history[i])]:.2f} | action: {action_history[i]} | reward: {reward_history[i]}')