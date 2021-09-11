import gym
from tqdm import tqdm
import custom_bandits
import numpy as np
import matplotlib.pyplot as plt
from agents import pureExploitation, pureExploration, epsilonGreedy, decayingEpsilonGreedy, softmaxExploration, UCB
from utils import smooth_array, create_Earth

if __name__ == "__main__":

    SEED = 0
    ANSWER_TO_EVERYTHING = 42

    timeSteps = 2000
    noOfEnvs = 50

    reward_exploitation = np.zeros((noOfEnvs,timeSteps))
    reward_exploration = np.zeros((noOfEnvs,timeSteps))
    reward_epsilonGreedy = np.zeros((noOfEnvs,timeSteps))
    reward_decayingEpsilonGreedy = np.zeros((noOfEnvs,timeSteps))
    reward_softmax = np.zeros((noOfEnvs,timeSteps))
    reward_ucb = np.zeros((noOfEnvs,timeSteps))

    optimalAction_exploitation = np.zeros((noOfEnvs,timeSteps))
    optimalAction_exploration = np.zeros((noOfEnvs,timeSteps))
    optimalAction_epsilonGreedy = np.zeros((noOfEnvs,timeSteps))
    optimalAction_decayingEpsilonGreedy = np.zeros((noOfEnvs,timeSteps))
    optimalAction_softmax = np.zeros((noOfEnvs,timeSteps))
    optimalAction_ucb = np.zeros((noOfEnvs,timeSteps))

    for i in tqdm(range(noOfEnvs),ascii=True, unit=" time-step "):

        if SEED==ANSWER_TO_EVERYTHING:
            SEED = SEED + 1
            create_Earth(ANSWER_TO_EVERYTHING) 

        np.random.seed(SEED)

        alpha = np.random.uniform()
        beta = np.random.uniform()

        env = gym.make('twoArm_bandits-v0', alpha=alpha, beta=beta, seed=SEED)
        env.reset()

        _, _, optimalAction_exploitation[i], _, _ = pureExploitation(env, timeSteps)
        _, _, optimalAction_exploration[i], _, _ = pureExploration(env, timeSteps)
        _, _, optimalAction_epsilonGreedy[i], _, _ = epsilonGreedy(env, timeSteps, epsilon=0.1)
        _, _, optimalAction_decayingEpsilonGreedy[i], _, _ = decayingEpsilonGreedy(env, timeSteps, decay_till=timeSteps/2, max_epsilon=1, min_epsilon=1e-6, decay_type='exp')
        _, _, optimalAction_softmax[i], _, _ = softmaxExploration(env, timeSteps, decay_till=timeSteps/4, max_tau=100, min_tau = 0.005, decay_type='exp')
        _, _, optimalAction_ucb[i], _, _ = UCB(env, timeSteps, c=0.1)

        print(f'    Seed: {SEED} || alpha: {alpha} || beta: {beta}')
        
        SEED = SEED + 1

    episodes = [i for i in range(timeSteps)]
    smooth_window = 50                          

    avg_optimalAction_exploitation = 100*smooth_array(np.mean(optimalAction_exploitation, axis=0), smooth_window)
    avg_optimalAction_exploration = 100*smooth_array(np.mean(optimalAction_exploration, axis=0), smooth_window)
    avg_optimalAction_epsilonGreedy = 100*smooth_array(np.mean(optimalAction_epsilonGreedy, axis=0), smooth_window)
    avg_optimalAction_decayingEpsilonGreedy = 100*smooth_array(np.mean(optimalAction_decayingEpsilonGreedy, axis=0), smooth_window)
    avg_optimalAction_softmax = 100*smooth_array(np.mean(optimalAction_softmax, axis=0), smooth_window)
    avg_optimalAction_ucb = 100*smooth_array(np.mean(optimalAction_ucb, axis=0), smooth_window)

    plt.figure(figsize=(12,8))
    plt.rcParams.update({'font.size': 14})
    plt.plot(episodes, avg_optimalAction_exploitation, label='Pure Exploitation',linewidth=2)
    plt.plot(episodes, avg_optimalAction_exploration, label='Pure Exploration',linewidth=2)
    plt.plot(episodes, avg_optimalAction_epsilonGreedy, label='Epsilon Greedy',linewidth=2)
    plt.plot(episodes, avg_optimalAction_decayingEpsilonGreedy, label='Deacying Epsilon',linewidth=2)
    plt.plot(episodes, avg_optimalAction_softmax, label='Softmax',linewidth=2)
    plt.plot(episodes, avg_optimalAction_ucb, label='UCB',linewidth=2)
    plt.title('Percent Optimal Action for 50 Two Arm Bandit Environments')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Optimal Action')
    plt.legend()
    plt.savefig('Q1P8.svg')
    plt.savefig('Q1P8.jpg', dpi=300)
    plt.show()