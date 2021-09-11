import gym
import custom_bandits
import numpy as np
import matplotlib.pyplot as plt

def epsilonGreedy(env, maxEpisodes, epsilon):
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
            if i==0:
                regret_history[i] = env.q_value[optimal_action] - env.q_value[a]
            else:
                regret_history[i] = regret_history[i-1] + (env.q_value[optimal_action] - env.q_value[a]) 

            if a==optimal_action:
                optimal_action_history[i] = N[a]/(i+1)

        return Q_est, optimal_action_history, r_history, regret_history

if __name__ == "__main__":
    sigma = 1
    SEED = 0
    epsilon = 0.1
    maxEpisodes = 1000
    env = gym.make('tenArmGaussian_bandits-v0', sigma_square=sigma, seed=SEED)
    env.reset()
    Q_estimates, action_history, reward_history, regret_history = epsilonGreedy(env,maxEpisodes,epsilon)

    print(env.q_value)
    print(Q_estimates[-1])

    episodes = [i for i in range(maxEpisodes)]
    plt.plot(episodes, Q_estimates)
    plt.show()

    print('--------------------------------------------')
    print(f'actionTaken | reward')
    # print('--------------------------------------------')
    for i in range(maxEpisodes-20, maxEpisodes):
        print(f' action: {action_history[i]} | reward: {reward_history[i]}')