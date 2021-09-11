import gym
import custom_bandits
import numpy as np
import matplotlib.pyplot as plt
from utils import smooth_array
from agents import UCB


if __name__ == "__main__":
    sigma = 1
    SEED = 0
    c = [0.1, 1, 5, 10]
    noOfc = 4
    maxEpisodes = 1000
    reward_history = np.zeros((noOfc, maxEpisodes))
    for i in range(noOfc):
        env = gym.make('tenArmGaussian_bandits-v0', sigma_square=sigma, seed=SEED)
        env.reset()
        Q_estimates, action_history, optimal_action_history, reward_history[i], regret_history = UCB(env,maxEpisodes,c[i])

    print(env.q_value)
    print(Q_estimates[-1])

    episodes = [i for i in range(maxEpisodes)]
    plt.figure(figsize=(12,8))
    plt.rcParams.update({'font.size': 14})
    ax = plt.subplot(111)
    for i in range(noOfc):
        ax.plot(episodes, smooth_array(reward_history[i],50), label='c = '+str(c[i]), linewidth=2)
    plt.ylabel('Reward')
    plt.xlabel('Episodes')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('UCB Agent for Ten-Armed Gaussian Bandit')
    plt.savefig('q1p5f.jpg', dpi=300)
    plt.savefig('q1p5f.svg')
    plt.show()
    
    print(f'--------------------SEED: {SEED}------------------------')
    print(f'True Q values: {env.q_value}')
    print(f'Final Q Estimates: {Q_estimates[-1,:]}')
    print(f'Action with highest q_value: {np.argmax(env.q_value)}')
    print(f'Q_val |Q[actionTaken]| actionTaken | reward')
    print('--------------------------------------------')
    for i in range(maxEpisodes-20, maxEpisodes):
        print(f'Q[{i}]| {Q_estimates[i][int(action_history[i])]:.2f} | action: {action_history[i]}')