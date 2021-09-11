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
    
    G_hist_v3 = []

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
            if s==3:
                G_hist_v3.append(G)
            visited[s] += 1

        
        v_r[e] = v_est
    
    return v_est, v_r, G_hist_v3


if __name__ == "__main__":
    SEED = 0
    env = gym.make('random_walk-v0', seed=SEED)
    
    gamma = 0.99
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
    # FVMC
    firstVisit = True
    _, _, G_hist_v3_fvmc = MonteCarloPrediction(env, left_policy, gamma, alpha, final_alpha, decayType, maxSteps, noEpisodes, firstVisit)

    # EVMC
    firstVisit = False
    _, _, G_hist_v3_evmc = MonteCarloPrediction(env, left_policy, gamma, alpha, final_alpha, decayType, maxSteps, noEpisodes, firstVisit)


    episodes = [i for i in range(len(G_hist_v3_fvmc))]
    plt.figure(figsize=(12,8))
    plt.rcParams.update({'font.size': 14})
    ax = plt.subplot(111)
    # for i in range(1,env.observation_space.n-1):
        # smooth_v = smooth_array(v_history_fvmc[:,i], 10)
    ax.plot(episodes, G_hist_v3_fvmc, 'r*', linewidth=2)
    ax.plot(episodes, [true_V_est[3] for j in range(len(G_hist_v3_fvmc))], 'k--', label='true state-value V(3)', linewidth=2)
    
    plt.xlabel('Estimate sequence number')
    plt.ylabel('Target Value')
    plt.title('FVMC target sequence for gamma = '+str(gamma))
    plt.legend()
    plt.savefig('Q2P12_gamma_'+str(gamma)+'.svg')
    plt.savefig('Q2P12_gamma_'+str(gamma)+'.jpg', dpi=300)
    plt.show()

    episodes = [i for i in range(len(G_hist_v3_evmc))]
    plt.figure(figsize=(12,8))
    plt.rcParams.update({'font.size': 14})
    ax = plt.subplot(111)
    # for i in range(1,env.observation_space.n-1):
        # smooth_v = smooth_array(v_history_evmc[:,i], 1)
    ax.plot(episodes, G_hist_v3_evmc, 'r*', linewidth=2)
    ax.plot(episodes, [true_V_est[3] for j in range(len(G_hist_v3_evmc))], 'k--', label='true state-value V(3)', linewidth=2)
    
    plt.xlabel('Estimate sequence number')
    plt.ylabel('Target Value')
    plt.title('EVMC target sequence for gamma = '+str(gamma))
    plt.legend()
    plt.savefig('Q2P13_gamma_'+str(gamma)+'.svg')
    plt.savefig('Q2P13_gamma_'+str(gamma)+'.jpg', dpi=300)
    plt.show()