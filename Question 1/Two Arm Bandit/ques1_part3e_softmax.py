import gym
import custom_bandits
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from utils import smooth_array
def softmaxExploration(env, maxEpisodes, decay_till, max_tau=100, min_tau = 0.005, decay_type='lin'):
        Q = np.zeros(env.action_space.n)
        N = np.zeros(env.action_space.n)
        Q_est = np.zeros((maxEpisodes, env.action_space.n))
        a_history = np.zeros(maxEpisodes)
        r_history = np.zeros(maxEpisodes)
        tau_history = np.zeros(maxEpisodes)
        regret_history = np.zeros(maxEpisodes)

        optimal_action = np.argmax(env.q_value)
        optimal_action_history = np.zeros(maxEpisodes)

    
        exp_decay_rate = np.power(min_tau/max_tau,(1/(decay_till)))
        lin_decay_rate = (max_tau - min_tau)/decay_till
        tau = max_tau
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

        return Q_est, a_history, optimal_action_history, r_history, regret_history

if __name__ == "__main__":
    
    alpha = 0.8
    beta = 0.8
    SEED = 0
    max_tau = 1e5
    min_tau = 0.005 
    decay_type = ['lin','exp']
    noOfDecays = 2
    maxEpisodes = 1000
    decay_till = maxEpisodes/2
    reward_history = np.zeros((noOfDecays, maxEpisodes))

    for i in range(noOfDecays):
        env = gym.make('twoArm_bandits-v0', alpha=alpha, beta=beta, seed=SEED)
        env.reset()
        Q_estimates, action_history, optimal_action_history, reward_history[i], regret_history = softmaxExploration(env,maxEpisodes, decay_till, max_tau, min_tau, decay_type[i])
    
    episodes = [i for i in range(maxEpisodes)]
    plt.figure(figsize=(12,8))
    plt.rcParams.update({'font.size': 14})
    for i in range(noOfDecays):
        plt.plot(episodes, smooth_array(reward_history[i],50), label='Decay type = '+str(decay_type[i]))
    plt.title('Softmax Agent for Two Armed Bandit')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('q1p3e_reward.jpg', dpi=300)
    plt.savefig('q1p3e_reward.svg')
    plt.show()