import gym
import custom_bandits
import numpy as np
import matplotlib.pyplot as plt
from utils import smooth_array
from agents import decayingEpsilonGreedy

if __name__ == "__main__":
    
    alpha = 0.8
    beta = 0.8
    SEED = 0
    maxEpisodes = 1000
    decay_type = ['lin','exp']
    noOfDecays = 2
    reward_history = np.zeros((noOfDecays, maxEpisodes))
    max_epsilon = 1
    min_epsilon = 1e-6
    decay_till = maxEpisodes/2
    for i in range(noOfDecays):
        env = gym.make('twoArm_bandits-v0', alpha=alpha, beta=beta, seed=SEED)
        env.reset()
        Q_estimates, action_history, optimal_action_history, reward_history[i], regret_history = decayingEpsilonGreedy(env,maxEpisodes,decay_till,max_epsilon,min_epsilon,decay_type[i])
    
    episodes = [i for i in range(maxEpisodes)]
    
    plt.figure(figsize=(12,8))
    plt.rcParams.update({'font.size': 14})
    for i in range(noOfDecays):
        plt.plot(episodes, smooth_array(reward_history[i],50), label='Decay type = '+str(decay_type[i]))
    plt.title('Decaying Epsilon Greedy Agent for Two Armed Bandit')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('q1p3d_reward.jpg', dpi=300)
    plt.savefig('q1p3d_reward.svg')
    plt.show()
    
    # print('--------------------------------------------')
    # print('Q_val |Q[a=0]|Q[a=1]| actionTaken | reward')
    # # print('--------------------------------------------')
    # for i in range(maxEpisodes-20, maxEpisodes):
    #     print(f'Q[{i}]| {Q_estimates[i][0]:.2f} | {Q_estimates[i][1]:.2f} | action: {action_history[i]} | reward: {reward_history[i]}')