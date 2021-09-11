import gym
import custom_bandits
import numpy as np
import matplotlib.pyplot as plt
from utils import smooth_array
from ques1_part3c_epsilonGreedy import epsilonGreedy
def decayingEpsilonGreedy(env, maxEpisodes, decay_till, max_epsilon=1, min_epsilon=1e-6, decay_type='exp'):
        Q = np.zeros(env.action_space.n)
        N = np.zeros(env.action_space.n)
        Q_est = np.zeros((maxEpisodes, env.action_space.n))
        a_history = np.zeros(maxEpisodes)
        r_history = np.zeros(maxEpisodes)
        epsilon_history = np.zeros(maxEpisodes)
        regret_history = np.zeros(maxEpisodes)

        optimal_action = np.argmax(env.q_value)
        optimal_action_history = np.zeros(maxEpisodes)

        epsilon = max_epsilon
        min_epsilon = min_epsilon # close to 0
        exp_decay_rate = np.power(min_epsilon,(1/(decay_till)))
        lin_decay_rate = (epsilon - min_epsilon)/decay_till

        for i in range(maxEpisodes):
            env.reset()
            if np.random.uniform() < epsilon:
                a = np.random.randint(0,env.action_space.n,1)[0]
            else:
                a = np.argmax(Q)
            _, R, _, _ = env.step(a)
            N[a] = N[a] + 1
            # print(N, a, R)
            Q[a] = Q[a] + (R-Q[a])/N[a]
            Q_est[i] = Q
            a_history[i] = a
            r_history[i] = R

            if epsilon > min_epsilon:
                if decay_type == 'exp':
                    epsilon = epsilon * exp_decay_rate
                elif decay_type == 'lin':
                    epsilon = max(epsilon - lin_decay_rate, min_epsilon)
                epsilon_history[i] = epsilon

            if i==0:
                regret_history[i] = env.q_value[optimal_action] - env.q_value[a]
            else:
                regret_history[i] = regret_history[i-1] + (env.q_value[optimal_action] - env.q_value[a])
            
            if a==optimal_action:
                optimal_action_history[i] = N[a]/(i+1)

        return Q_est, a_history, optimal_action_history, r_history, regret_history

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