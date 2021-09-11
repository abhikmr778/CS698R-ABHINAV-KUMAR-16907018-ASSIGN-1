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
            td_error = td_target - v_est[s]
            v_est[s] += alpha*td_error
            s = s_
        
        v_r[e] = v_est

    return v_est, v_r


if __name__ == "__main__":
    SEED = 0
    env = gym.make('random_walk-v0', seed=SEED)
    
    # True value
    left_pi = {
        0:1, # going left with prob 1
        1:0 # going right with prob 0
    }
    policyEvaluator = policyEvaluation(left_pi,gamma=0.99, theta=1e-10, max_iterations=1000)
    true_V_est = policyEvaluator.evaluate(env)
    
    left_policy = np.array([0 for i in range(env.observation_space.n)])
    gamma = 0.99
    alpha = 0.5
    final_alpha = 0.01
    decayType = 'exp'
    maxSteps = 250
    noEpisodes = 500
    # TD
    v_est_td, v_history_td = TemporalDifferencePrediction(env, left_policy, gamma, alpha,final_alpha,decayType, maxSteps, noEpisodes)

    
    episodes = [i for i in range(noEpisodes)]
    plt.figure(figsize=(12,8))
    plt.rcParams.update({'font.size': 14})
    ax = plt.subplot(111)
    for i in range(env.observation_space.n):
        ax.plot(episodes, smooth_array(v_history_td[:,i], 10), label='V(s='+str(i)+')', linewidth=2)
        ax.plot(episodes, [true_V_est[i] for j in range(noEpisodes)], 'k--', label = 'V*(s='+str(i)+')', linewidth=2)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Time Steps')
    plt.ylabel('State-value estimates')
    plt.title('Temporal Difference Algorithm')
    plt.savefig('q2p4.jpg', dpi=300)
    plt.savefig('q2p4.svg')
    plt.show()
