import gym
import random_walk
import numpy as np
from utils import smooth_array
import matplotlib.pyplot as plt
from ques2_part1 import generateTrajectory
from ques2_part2 import decayAlpha
from true_value_randomWalk import policyEvaluation

def MonteCarloPrediction(env, policy, gamma, alpha, final_alpha, decayType,  maxSteps_alpha, maxSteps_traject, noEpisodes, firstVisit):
    """
    Follows algorithm from lectures
    Input: env - environment
           policy - policy to follow
           gamma - discount factor
           alpha - initial step size
           final_alpha - final step size
           decayType - lin or exp
           maxSteps_alpha - steps till which to decay alpha
           maxSteps_traject - max steps within which trajectory should terminate
           noEpisodes - number of episodes
           firstVisit - whether to use FVMC or EVMC

    Output: v_est - final estimates, v_r - history of estimates, G_hist_v3 - return history of state s=3
    """
    
    # initialize estimates
    v_est = np.zeros(env.observation_space.n)
    
    # tracks history of v_est
    v_r = np.zeros((noEpisodes, env.observation_space.n))
    
    # stores return sequence
    G_hist_v3 = []

    # generate all alphas
    alphas = decayAlpha(initialValue=alpha, finalValue=final_alpha, maxSteps=maxSteps, decayType=decayType)
    
    for e in range(noEpisodes):
        if e < maxSteps:
            alpha = alphas[e]
        else:
            alpha = alphas[-1]
        
        # get a trajectory
        t = generateTrajectory(env=env, policy=policy, maxSteps=maxSteps)

        # tracks if state has been visited
        visited = np.zeros(env.observation_space.n)
        
        # follows algorithm from lectures
        for i, (s,a,s_,r,d) in enumerate(t):
            if visited[s] and firstVisit:
                continue
            
            # return calc
            G = 0
            for j in range(i, len(t)):
                G += np.power(gamma, j-i) * t[j][3]
            
            # update estimate
            v_est[s] += alpha * (G - v_est[s])
            
            # track return of state 3
            if s==3:
                G_hist_v3.append(G)

            visited[s] += 1 # mark visited

        
        v_r[e] = v_est
    
    return v_est, v_r, G_hist_v3


if __name__ == "__main__":
    # parameters
    SEED = 0
    gamma = 0.99
    alpha = 0.5
    final_alpha = 0.01
    decayType = 'exp'
    maxSteps_alpha = 250
    maxSteps_traject = 250
    noEpisodes = 500
    
    # True state value calc
    left_pi = {
        0:1, # going left with prob 1
        1:0 # going right with prob 0
    }
    env = gym.make('random_walk-v0', seed=SEED)
    policyEvaluator = policyEvaluation(left_pi,gamma=gamma, theta=1e-10, max_iterations=1000)
    true_V_est = policyEvaluator.evaluate(env)
    
    # left policy
    left_policy = np.array([0 for i in range(env.observation_space.n)])
    
    # FVMC
    firstVisit = True
    _, _, G_hist_v3_fvmc = MonteCarloPrediction(env, left_policy, gamma, alpha, final_alpha, decayType, maxSteps_alpha, maxSteps_traject, noEpisodes, firstVisit)

    # EVMC
    firstVisit = False
    _, _, G_hist_v3_evmc = MonteCarloPrediction(env, left_policy, gamma, alpha, final_alpha, decayType, maxSteps_alpha, maxSteps_traject, noEpisodes, firstVisit)


    # plot return history
    episodes = [i for i in range(len(G_hist_v3_fvmc))]
    plt.figure(figsize=(12,8))
    plt.rcParams.update({'font.size': 14})
    ax = plt.subplot(111)
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
    ax.plot(episodes, G_hist_v3_evmc, 'r*', linewidth=2)
    ax.plot(episodes, [true_V_est[3] for j in range(len(G_hist_v3_evmc))], 'k--', label='true state-value V(3)', linewidth=2)
    
    plt.xlabel('Estimate sequence number')
    plt.ylabel('Target Value')
    plt.title('EVMC target sequence for gamma = '+str(gamma))
    plt.legend()
    plt.savefig('Q2P13_gamma_'+str(gamma)+'.svg')
    plt.savefig('Q2P13_gamma_'+str(gamma)+'.jpg', dpi=300)
    plt.show()