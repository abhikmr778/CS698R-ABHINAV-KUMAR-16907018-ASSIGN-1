import gym
import custom_bandits
import numpy as np
import matplotlib.pyplot as plt
from agents import pureExploration

if __name__ == "__main__":
    
    # parameters
    alpha = 0.8
    beta = 0.8
    SEED = 5
    maxEpisodes = 20
    
    # create env
    env = gym.make('twoArm_bandits-v0', alpha=alpha, beta=beta, seed=SEED)
    env.reset()
    # run agent
    Q_estimates, action_history, optimal_action_history, reward_history, regret_history = pureExploration(env,maxEpisodes)
        
    print(f'-------------------SEED: {SEED}-------------------')
    print(f'Q_val|Q[a=0]|Q[a=1]| actionTaken | reward')
    print('--------------------------------------------')
    for i in range(maxEpisodes-20, maxEpisodes):
        print(f'Q[{i}] | {Q_estimates[i][0]:.2f} | {Q_estimates[i][1]:.2f} | action: {action_history[i]} | reward: {reward_history[i]}')