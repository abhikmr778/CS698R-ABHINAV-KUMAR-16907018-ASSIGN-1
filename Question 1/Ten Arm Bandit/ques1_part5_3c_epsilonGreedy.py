import gym
import custom_bandits
import numpy as np
import matplotlib.pyplot as plt
from agents import epsilonGreedy

if __name__ == "__main__":

    # parameters
    sigma = 1
    SEED = 0
    epsilon = 0.1
    maxEpisodes = 1000

    # create env
    env = gym.make('tenArmGaussian_bandits-v0', sigma_square=sigma, seed=SEED)
    env.reset()
    
    # run the agent
    Q_estimates, action_history, optimal_action_history, reward_history, regret_history = epsilonGreedy(env,maxEpisodes,epsilon)

    # check values
    print(env.q_value)
    print(Q_estimates[-1])

    # plotting the estimates vs true value
    episodes = [i for i in range(maxEpisodes)]
    plt.figure(figsize=(12,8))
    plt.rcParams.update({'font.size': 16})
    ax = plt.subplot(111)
    for i in range(env.action_space.n):
        if i%2==1:
            ax.plot(episodes, Q_estimates[:,i], label='Q(a='+str(i)+')', linewidth=2)
            ax.plot(episodes, [env.q_value[i] for j in range(maxEpisodes)], '--', label='Q*(a='+str(i)+')')
    plt.ylabel('Q function Estimates')
    plt.xlabel('Episodes')
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Epsilon Greedy Agent for Ten-Armed Gaussian Bandit for epsilon='+str(epsilon))
    plt.savefig('q1p5c.jpg', dpi=300)
    plt.savefig('q1p5c.svg')
    plt.show()

    print(f'--------------------SEED: {SEED}------------------------')
    print(f'True Q values: {env.q_value}')
    print(f'Final Q Estimates: {Q_estimates[-1,:]}')
    print(f'Action with highest q_value: {np.argmax(env.q_value)}')
    print(f'Q_val |Q[actionTaken]| actionTaken | reward')
    print('--------------------------------------------')
    for i in range(maxEpisodes-20, maxEpisodes):
        print(f'Q[{i}]| {Q_estimates[i][int(action_history[i])]:.2f} | action: {action_history[i]} | reward: {reward_history[i]}')