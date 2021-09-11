import gym
import random_walk
import numpy as np
from utils import smooth_array
import matplotlib.pyplot as plt
from ques2_part1 import generateTrajectory
from ques2_part2 import decayAlpha
from true_value_randomWalk import policyEvaluation


def MonteCarloPrediction(env, policy, gamma, alpha, final_alpha, decayType,  maxSteps, noEpisodes, firstVisit):
    v_est = np.zeros(env.observation_space.n)
    v_r = np.zeros((noEpisodes, env.observation_space.n))
    
    alphas = decayAlpha(initialValue=alpha, finalValue=final_alpha, maxSteps=maxSteps, decayType=decayType)
    for e in range(noEpisodes):
        if e < maxSteps:
            alpha = alphas[e]
        else:
            alpha = alphas[-1]
        t = generateTrajectory(env=env, policy=policy, maxSteps=maxSteps)
        visited = np.zeros(env.observation_space.n)

        for i, (s,a,s_,r,d) in enumerate(t):
            if visited[s] and firstVisit:
                continue
            
            G = 0
            for j in range(i, len(t)):
                G += np.power(gamma, j-i) * t[j][3]
            
            v_est[s] += alpha * (G - v_est[s])
            visited[s] += 1

        
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
    # FVMC
    firstVisit = True
    v_est_fvmc, v_history_fvmc = MonteCarloPrediction(env, left_policy, gamma, alpha,final_alpha,decayType, maxSteps, noEpisodes, firstVisit)

    # EVMC
    firstVisit = False
    v_est_evmc, v_history_evmc = MonteCarloPrediction(env, left_policy, gamma, alpha,final_alpha,decayType, maxSteps, noEpisodes, firstVisit)

    episodes = [i for i in range(noEpisodes)]
    plt.figure(figsize=(12,8))
    plt.rcParams.update({'font.size': 14})
    ax = plt.subplot(211)
    for i in range(env.observation_space.n):
        ax.plot(episodes, smooth_array(v_history_fvmc[:,i], 10), label='V('+str(i)+')', linewidth=2)
        ax.plot(episodes, [true_V_est[i] for j in range(noEpisodes)], 'k--', label='V*('+str(i)+')', linewidth=2)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    # Put a legend to the right of the current axis
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('Time Steps')
    plt.ylabel('Value function estimates')
    plt.title('First Visit Monte Carlo')
    # plt.show()

    # plt.figure(figsize=(12,8))
    # plt.rcParams.update({'font.size': 14})
    ax = plt.subplot(212)
    for i in range(env.observation_space.n):
        ax.plot(episodes, smooth_array(v_history_evmc[:,i], 10), label='V(s='+str(i)+')', linewidth=2)
        ax.plot(episodes, [true_V_est[i] for j in range(noEpisodes)], 'k--', label='V*(s='+str(i)+')', linewidth=2)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, 0.7*box.y0, box.width * 0.9, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 1))
    # Put a legend to the right of the current axis
    plt.xlabel('Time Steps')
    plt.ylabel('State-value estimates')
    plt.title('Every Visit Monte Carlo')
    plt.savefig('q2p3.jpg', dpi=300)
    plt.savefig('q2p3.svg')
    plt.show()