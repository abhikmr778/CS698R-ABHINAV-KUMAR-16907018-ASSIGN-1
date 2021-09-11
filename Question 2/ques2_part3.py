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

    Output: v_est - final estimates, v_r - history of estimates
    """

    # initialize estimates
    v_est = np.zeros(env.observation_space.n)
    
    # tracks history of v_est
    v_r = np.zeros((noEpisodes, env.observation_space.n))
    
    # generate all alphas
    alphas = decayAlpha(initialValue=alpha, finalValue=final_alpha, maxSteps=maxSteps_alpha, decayType=decayType)

    for e in range(noEpisodes):
        
        # select alpha
        if e < maxSteps_alpha:
            alpha = alphas[e]
        else:
            alpha = alphas[-1]
        
        # get a trajectory
        t = generateTrajectory(env=env, policy=policy, maxSteps=maxSteps_traject)
        
        # tracks if state has been visited
        visited = np.zeros(env.observation_space.n)

        # follows algorithm from lectures
        for i, (s,a,s_,r,d) in enumerate(t):
            if visited[s] and firstVisit: # FVMC
                continue
            
            # return calc
            G = 0
            for j in range(i, len(t)):
                G += np.power(gamma, j-i) * t[j][3]
            
            # update estimate
            v_est[s] += alpha * (G - v_est[s])
            visited[s] += 1 # mark visited

        # store 
        v_r[e] = v_est
    
    return v_est, v_r

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
    
    # left policy distn
    left_pi = {
        0:1, # going left with prob 1
        1:0 # going right with prob 0
    }

    # create env
    env = gym.make('random_walk-v0', seed=SEED)
    env.reset()

    # evaluate true value
    policyEvaluator = policyEvaluation(left_pi,gamma=0.99, theta=1e-10, max_iterations=1000)
    true_V_est = policyEvaluator.evaluate(env)
    
    # left policy
    left_policy = np.array([0 for i in range(env.observation_space.n)])
    
    # FVMC
    firstVisit = True
    v_est_fvmc, v_history_fvmc = MonteCarloPrediction(env, left_policy, gamma, alpha,final_alpha,decayType, maxSteps_alpha, maxSteps_traject, noEpisodes, firstVisit)

    # EVMC
    firstVisit = False
    v_est_evmc, v_history_evmc = MonteCarloPrediction(env, left_policy, gamma, alpha,final_alpha,decayType, maxSteps_alpha, maxSteps_traject, noEpisodes, firstVisit)

    # plot results
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