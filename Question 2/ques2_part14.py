import gym
import random_walk
import numpy as np
import matplotlib.pyplot as plt
from ques2_part1 import generateTrajectory
from ques2_part2 import decayAlpha
from true_value_randomWalk import policyEvaluation
from utils import smooth_array


def TemporalDifferencePrediction(env, policy, gamma, alpha,final_alpha,decayType, maxSteps, noEpisodes):
    v_est = np.zeros(env.observation_space.n)
    v_r = np.zeros((noEpisodes, env.observation_space.n))
    
    td_hist_v3 = []

    alphas = decayAlpha(initialValue=alpha, finalValue=final_alpha, maxSteps=maxSteps, decayType=decayType)
    for e in range(noEpisodes):
        if e < maxSteps:
            alpha = alphas[e]
        else:
            alpha = alphas[-1]
        
        s = env.reset()
        done = False
        while not done:
            a = policy[s]
            s_, r, done, _ = env.step(a)
            td_target = r
            if not done:
                td_target += gamma*v_est[s_]
            if s==3:
                td_hist_v3.append(td_target)
            td_error = td_target - v_est[s]
            v_est[s] += alpha*td_error
            s = s_
        
        v_r[e] = v_est

    return v_est, v_r, td_hist_v3


if __name__ == "__main__":
    SEED = 0
    env = gym.make('random_walk-v0', seed=SEED)
    
    gamma = 1
    # True value
    left_pi = {
        0:1, # going left with prob 1
        1:0 # going right with prob 0
    }
    policyEvaluator = policyEvaluation(left_pi,gamma=gamma, theta=1e-10, max_iterations=1000)
    true_V_est = policyEvaluator.evaluate(env)
    
    left_policy = np.array([0 for i in range(env.observation_space.n)])
    alpha = 0.5
    final_alpha = 0.01
    decayType = 'exp'
    maxSteps = 250
    noEpisodes = 500
    # TD
    _, _, td_hist_v3 = TemporalDifferencePrediction(env, left_policy, gamma, alpha,final_alpha,decayType, maxSteps, noEpisodes)

    
    episodes = [i for i in range(len(td_hist_v3))]
    plt.figure(figsize=(12,8))
    # for i in range(1,env.observation_space.n-1):
        # smooth_v = smooth_array(v_history_td[:,i], 10)
    plt.plot(episodes, td_hist_v3,'go')
    plt.plot(episodes, [true_V_est[3] for j in range(len(td_hist_v3))], 'k--', label='true state-value V(3)')
    plt.title('TD target sequence')
    plt.xlabel('Estimate sequence number')
    plt.ylabel('Target Value')
    plt.legend()
    plt.savefig('Q2P14.svg')
    plt.savefig('Q2P14.jpg', dpi=300)
    plt.show()
