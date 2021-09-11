import gym
import custom_bandits
import numpy as np
import matplotlib.pyplot as plt
from utils import smooth_array
from agents import epsilonGreedy

if __name__ == "__main__":
    alpha = 0.8
    beta = 0.8
    SEED = 0
    epsilon = [0.1, 0.01, 0, 1]
    noOfEps = 4
    maxEpisodes = 1000
    reward_history = np.zeros((noOfEps,maxEpisodes))
    for i in range(noOfEps):
        env = gym.make('twoArm_bandits-v0', alpha=alpha, beta=beta, seed=SEED)
        # env = TwoArmBandit(alpha, beta, SEED)
        env.reset()
        Q_estimates, action_history, optimal_action_history, reward_history[i], regret_history = epsilonGreedy(env,maxEpisodes,epsilon[i])
    
    episodes = [i for i in range(maxEpisodes)]
    
    plt.figure(figsize=(12,8))
    plt.rcParams.update({'font.size': 14})
    for i in range(noOfEps):
        plt.plot(episodes, smooth_array(reward_history[i],50), label='Epsilon = '+str(epsilon[i]))
    plt.title('Epsilon Greedy Agent for Two Armed Bandit')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('q1p3c_reward.jpg', dpi=300)
    plt.savefig('q1p3c_reward.svg')
    plt.show()

    # print(f'-------------------SEED: {SEED}-------------------')
    # print(f'Q_val|Q[a=0]|Q[a=1]| actionTaken | reward')
    # # print('--------------------------------------------')
    # for i in range(maxEpisodes-20, maxEpisodes):
    #     print(f'Q[{i}] | {Q_estimates[i][0]:.2f} | {Q_estimates[i][1]:.2f} | action: {action_history[i]} | reward: {reward_history[i]}')