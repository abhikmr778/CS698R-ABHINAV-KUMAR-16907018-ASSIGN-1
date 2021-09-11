import gym
import custom_bandits
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

def softmaxExploration(env, maxEpisodes, tau=100, decay_type='lin'):
        Q = np.zeros(env.action_space.n, dtype=np.float64)
        N = np.zeros(env.action_space.n)
        Q_est = np.zeros((maxEpisodes, env.action_space.n))
        a_history = np.zeros(maxEpisodes)
        r_history = np.zeros(maxEpisodes)
        tau_history = np.zeros(maxEpisodes)
        regret_history = np.zeros(maxEpisodes)

        optimal_action = np.argmax(env.q_value)
        optimal_action_history = np.zeros(maxEpisodes)

        # want to decay till maxEpisodes/2
        decay_till = maxEpisodes/4
        min_tau = 0.005
        exp_decay_rate = np.power(min_tau,(1/(decay_till)))
        lin_decay_rate = (tau - min_tau)/decay_till
        
        for i in range(maxEpisodes):
            env.reset()
            probs = softmax(Q/tau)
            a = np.random.choice(a = env.action_space.n, size = 1, p = probs)[0]
            _, R, _, _ = env.step(a)
            N[a] = N[a] + 1
            # print(N, a, R)
            Q[a] = Q[a] + (R-Q[a])/N[a]
            Q_est[i] = Q
            a_history[i] = a
            r_history[i] = R

            if tau > min_tau:
                if decay_type == 'exp':
                    tau = tau * exp_decay_rate
                elif decay_type == 'lin':
                    tau = max(tau - lin_decay_rate, min_tau)
                tau_history[i] = tau
            
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
    tau = 1e5
    decay_type = 'lin'
    maxEpisodes = 1000
    env = gym.make('tenArmGaussian_bandits-v0', sigma_square=sigma, seed=SEED)
    env.reset()
    Q_estimates, action_history, reward_history, regret_history = softmaxExploration(env,maxEpisodes,tau,decay_type)
    
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