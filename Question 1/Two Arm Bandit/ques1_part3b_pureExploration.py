import gym
import custom_bandits
import numpy as np
import matplotlib.pyplot as plt

def pureExploration(env, maxEpisodes):
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
            a = np.random.randint(0,env.action_space.n,1)[0]
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
    SEED = 5
    maxEpisodes = 20
    env = gym.make('twoArm_bandits-v0', alpha=alpha, beta=beta, seed=SEED)
    # env = TwoArmBandit(alpha, beta, SEED)
    env.reset()
    Q_estimates, action_history, optimal_action_history, reward_history, regret_history = pureExploration(env,maxEpisodes)
    
    # plt.plot(regret_history)
    # plt.show()
    
    print(f'-------------------SEED: {SEED}-------------------')
    print(f'Q_val|Q[a=0]|Q[a=1]| actionTaken | reward')
    # print('--------------------------------------------')
    for i in range(maxEpisodes-20, maxEpisodes):
        print(f'Q[{i}] | {Q_estimates[i][0]:.2f} | {Q_estimates[i][1]:.2f} | action: {action_history[i]} | reward: {reward_history[i]}')