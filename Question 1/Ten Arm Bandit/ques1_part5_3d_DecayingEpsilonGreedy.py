import gym
import custom_bandits
import numpy as np
import matplotlib.pyplot as plt
from utils import smooth_array
from agents import decayingEpsilonGreedy


if __name__ == "__main__":
    
    # parameters
    sigma = 1
    SEED = 0
    maxEpisodes = 1000
    decay_type = ['lin','exp']
    noOfDecays = 2
    max_epsilon = 1
    min_epsilon = 1e-6
    decay_till = maxEpisodes/2
    reward_history = np.zeros((noOfDecays, maxEpisodes)) # will store rewards for both decay types

    # for every decay type
    for i in range(noOfDecays):
        # create env
        env = gym.make('tenArmGaussian_bandits-v0', sigma_square=sigma, seed=SEED)
        env.reset()
        # run agent
        Q_estimates, action_history, optimal_action_history, reward_history[i], regret_history = decayingEpsilonGreedy(env,maxEpisodes,decay_till,max_epsilon,min_epsilon, decay_type[i])
    
    print(env.q_value)
    print(Q_estimates[-1])

    # plot and compare results of linear and exponential
    episodes = [i for i in range(maxEpisodes)]
    plt.figure(figsize=(12,8))
    plt.rcParams.update({'font.size': 14})
    ax = plt.subplot(111)
    for i in range(noOfDecays):
        ax.plot(episodes, smooth_array(reward_history[i],50), label='Decay type = '+str(decay_type[i]), linewidth=2)
    plt.ylabel('Reward')
    plt.xlabel('Episodes')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Decaying Epsilon Greedy Agent for Ten-Armed Gaussian Bandit')
    plt.savefig('q1p5d.jpg', dpi=300)
    plt.savefig('q1p5d.svg')
    plt.show()

    print(f'--------------------SEED: {SEED}------------------------')
    print(f'True Q values: {env.q_value}')
    print(f'Final Q Estimates: {Q_estimates[-1,:]}')
    print(f'Action with highest q_value: {np.argmax(env.q_value)}')
    print(f'Q_val |Q[actionTaken]| actionTaken | reward')
    print('--------------------------------------------')
    for i in range(maxEpisodes-20, maxEpisodes):
        print(f'Q[{i}]| {Q_estimates[i][int(action_history[i])]:.2f} | action: {action_history[i]}')