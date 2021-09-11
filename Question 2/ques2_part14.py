import gym
import random_walk
import numpy as np
import matplotlib.pyplot as plt
from ques2_part1 import generateTrajectory
from ques2_part2 import decayAlpha
from true_value_randomWalk import policyEvaluation
from utils import smooth_array


def TemporalDifferencePrediction(env, policy, gamma, alpha,final_alpha,decayType, maxSteps, noEpisodes):
    """
    Follows algorithm from lectures
    Input: env - environment
           policy - policy to follow
           gamma - discount factor
           alpha - initial step size
           final_alpha - final step size
           decayType - lin or exp
           maxSteps - steps till which to decay alpha
           noEpisodes - number of episodes

    Output: v_est - final estimates, v_r - history of estimates, td_hist_v3 - target history of state s=3
    """
    
    # initialize estimates
    v_est = np.zeros(env.observation_space.n)

    # tracks history of v_est
    v_r = np.zeros((noEpisodes, env.observation_space.n))
    
    # stores target return history
    td_hist_v3 = []

    # generate alphas
    alphas = decayAlpha(initialValue=alpha, finalValue=final_alpha, maxSteps=maxSteps, decayType=decayType)
    
    for e in range(noEpisodes):
        if e < maxSteps:
            alpha = alphas[e]
        else:
            alpha = alphas[-1]
        
        # simulate a trajectory
        s = env.reset() # starting state
        done = False
        while not done: # until terminal state is reached
            a = policy[s]
            s_, r, done, _ = env.step(a)
            td_target = r
            if not done:
                td_target += gamma*v_est[s_]
            
            # track target sequence
            if s==3:
                td_hist_v3.append(td_target)
            
            td_error = td_target - v_est[s]
            v_est[s] += alpha*td_error
            s = s_
        
        v_r[e] = v_est

    return v_est, v_r, td_hist_v3


if __name__ == "__main__":

    # parameters
    SEED = 0
    alpha = 0.5
    final_alpha = 0.01
    decayType = 'exp'
    maxSteps = 250
    noEpisodes = 500
    gamma = 0.99

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
    
    # TD
    _, _, td_hist_v3 = TemporalDifferencePrediction(env, left_policy, gamma, alpha,final_alpha,decayType, maxSteps, noEpisodes)

    # plot target return history
    episodes = [i for i in range(len(td_hist_v3))]
    plt.figure(figsize=(12,8))
    plt.rcParams.update({'font.size': 14})
    ax = plt.subplot(111)
    ax.plot(episodes, td_hist_v3,'g*',linewidth=2)
    ax.plot(episodes, [true_V_est[3] for j in range(len(td_hist_v3))], 'k--', label='true state-value V(3)', linewidth=2)
    plt.title('TD target sequence for gamma = '+str(gamma))
    plt.xlabel('Estimate sequence number')
    plt.ylabel('Target Value')
    plt.legend()
    plt.savefig('Q2P14_gamma_'+str(gamma)+'.svg')
    plt.savefig('Q2P14_gamma_'+str(gamma)+'.jpg', dpi=300)
    plt.show()
