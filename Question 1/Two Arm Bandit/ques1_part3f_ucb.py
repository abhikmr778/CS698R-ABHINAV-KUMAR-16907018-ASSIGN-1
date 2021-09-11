import gym
import custom_bandits
import numpy as np
import matplotlib.pyplot as plt
from utils import smooth_array
from agents import UCB

if __name__ == "__main__":
    
    # parameters
    alpha = 0.8
    beta = 0.8
    SEED = 0
    c = [0.1, 1, 5, 10]
    noOfc = 4
    maxEpisodes = 1000
    reward_history = np.zeros((noOfc, maxEpisodes))
    
    # for every c[i]
    for i in range(noOfc):
        # create env
        env = gym.make('twoArm_bandits-v0', alpha=alpha, beta=beta, seed=SEED)
        env.reset()
        # run agent for c[i]
        Q_estimates, action_history, optimal_action_history, reward_history[i], regret_history = UCB(env,maxEpisodes,c[i])

    # compare results of all c[i]
    episodes = [i for i in range(maxEpisodes)]
    
    plt.figure(figsize=(12,8))
    plt.rcParams.update({'font.size': 14})
    for i in range(noOfc):
        plt.plot(episodes, smooth_array(reward_history[i],50), label='c = '+str(c[i]))
    plt.title('UCB Agent for Two Armed Bandit')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('q1p3f_reward.jpg', dpi=300)
    plt.savefig('q1p3f_reward.svg')
    plt.show()