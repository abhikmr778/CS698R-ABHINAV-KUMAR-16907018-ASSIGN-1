import gym
import random_walk
import numpy as np
import matplotlib.pyplot as plt
from ques2_part1 import generateTrajectory
from ques2_part2 import decayAlpha
from ques2_part4 import TemporalDifferencePrediction
from true_value_randomWalk import policyEvaluation
from utils import smooth_array, create_Earth
from tqdm import tqdm

if __name__ == "__main__":
    SEED = 0
    ANSWER_TO_EVERYTHING = 42

    noOfEnvs = 50
    decayType = 'exp'
    maxSteps = 250
    noEpisodes = 500
    ENV_OBSERVATION_SPACE = 7
    v_total_history_td = np.zeros((noOfEnvs,noEpisodes,ENV_OBSERVATION_SPACE))
    
    for i in tqdm(range(noOfEnvs), ascii=True, unit=" env "):

        if SEED==ANSWER_TO_EVERYTHING:
            SEED = SEED + 1
            create_Earth(ANSWER_TO_EVERYTHING)
        
        np.random.seed(SEED)

        env = gym.make('random_walk-v0', seed=SEED)
        env.reset()

        # True value
        left_pi = {
            0:1, # going left with prob 1
            1:0 # going right with prob 0
        }
        policyEvaluator = policyEvaluation(left_pi,gamma=0.99, theta=1e-10, max_iterations=1000)
        true_V_est = policyEvaluator.evaluate(env)
        
        left_policy = np.array([0 for i in range(env.observation_space.n)])
        gamma = 0.99
        final_alpha = 0.01
        alpha = np.random.uniform(final_alpha,1)
        # TD
        firstVisit = True
        _, v_total_history_td[i] = TemporalDifferencePrediction(env, left_policy, gamma, alpha,final_alpha,decayType, maxSteps, noEpisodes)
        print(f'    Seed: {SEED} || alpha: {alpha} ')
        
        SEED = SEED + 1
    
    avg_v_total_history_td = np.mean(v_total_history_td, axis=0) 
    episodes = [i for i in range(noEpisodes)]
    plt.figure(figsize=(12,8))
    plt.xscale("log")
    for i in range(1,env.observation_space.n-1):
        # smooth_v = smooth_array(v_history_td[:,i], 10)
        plt.plot(episodes, avg_v_total_history_td[:,i], label='V('+str(i)+')')
        plt.plot(episodes, [true_V_est[i] for j in range(noEpisodes)], 'k--')
    plt.title('TD estimates through time vs true values')
    plt.xlabel('Episodes')
    plt.ylabel('State-Value Function')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('Q2P7.svg')
    plt.savefig('Q2P7.jpg', dpi=300)
    plt.show()
