import gym
import custom_bandits
import numpy as np
import matplotlib.pyplot as plt
from utils import smooth_array
def UCB(env, maxEpisodes, c):
        Q = np.zeros(env.action_space.n)
        N = np.zeros(env.action_space.n)
        Q_est = np.zeros((maxEpisodes, env.action_space.n))
        a_history = np.zeros(maxEpisodes)
        r_history = np.zeros(maxEpisodes)
        regret_history = np.zeros(maxEpisodes)

        optimal_action = np.argmax(env.q_value)
        optimal_action_history = np.zeros(maxEpisodes)

        for i in range(maxEpisodes):
            env.reset()
            if i < env.action_space.n:
                a = i
            else:
                U = c * np.sqrt(np.log(i)/N)
                a = np.argmax(Q+U)
            _, R, _, _ = env.step(a)
            N[a] = N[a] + 1
            # print(N, a, R)
            Q[a] = Q[a] + (R-Q[a])/N[a]
            Q_est[i] = Q
            a_history[i] = a
            r_history[i] = R
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
    c = [0.1, 1, 5, 10]
    noOfc = 4
    maxEpisodes = 1000
    reward_history = np.zeros((noOfc, maxEpisodes))
    
    for i in range(noOfc):
        env = gym.make('twoArm_bandits-v0', alpha=alpha, beta=beta, seed=SEED)
        env.reset()
        Q_estimates, action_history, optimal_action_history, reward_history[i], regret_history = UCB(env,maxEpisodes,c[i])


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