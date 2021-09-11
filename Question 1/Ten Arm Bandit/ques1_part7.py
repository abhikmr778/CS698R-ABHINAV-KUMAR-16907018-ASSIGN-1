import gym
from tqdm import tqdm
import custom_bandits
import numpy as np
import matplotlib.pyplot as plt
from agents import pureExploitation, pureExploration, epsilonGreedy, decayingEpsilonGreedy, softmaxExploration, UCB
from utils import smooth_array, create_Earth

if __name__ == "__main__":

    # parameters
    SEED = 0
    ANSWER_TO_EVERYTHING = 42

    timeSteps = 1000
    noOfEnvs = 50

    # to store regret across 50 envs
    regret_exploitation = np.zeros((noOfEnvs,timeSteps))
    regret_exploration = np.zeros((noOfEnvs,timeSteps))
    regret_epsilonGreedy = np.zeros((noOfEnvs,timeSteps))
    regret_decayingEpsilonGreedy = np.zeros((noOfEnvs,timeSteps))
    regret_softmax = np.zeros((noOfEnvs,timeSteps))
    regret_ucb = np.zeros((noOfEnvs,timeSteps))

    # for every env
    for i in tqdm(range(noOfEnvs),ascii=True, unit=" time-step "):

        # skip 42
        if SEED==ANSWER_TO_EVERYTHING:
            SEED = SEED + 1
            create_Earth(ANSWER_TO_EVERYTHING) # read hitchhiker's guide to galaxy if not yet read

        np.random.seed(SEED)

        # generate sigma
        sigma = np.random.uniform(0,2.5,1)

        # create env
        env = gym.make('tenArmGaussian_bandits-v0', sigma_square=sigma, seed=SEED)
        env.reset()

        # store regret history for every env
        _, _, _, _, regret_exploitation[i] = pureExploitation(env, timeSteps)
        _, _, _, _, regret_exploration[i] = pureExploration(env, timeSteps)
        _, _, _, _, regret_epsilonGreedy[i] = epsilonGreedy(env, timeSteps, epsilon=0.1)
        _, _, _, _, regret_decayingEpsilonGreedy[i] = decayingEpsilonGreedy(env, timeSteps, decay_till=timeSteps/2, max_epsilon=1, min_epsilon=1e-6, decay_type='exp')
        _, _, _, _, regret_softmax[i] = softmaxExploration(env, timeSteps, decay_till=timeSteps/2, max_tau=100, min_tau = 0.005, decay_type='exp')
        _, _, _, _, regret_ucb[i] = UCB(env, timeSteps, c=1)

        print(f'    Seed: {SEED} || sigma: {sigma} ')
        
        # increment seed
        SEED = SEED + 1

    # average regret across envs and plot
    episodes = [i for i in range(timeSteps)] 

    avg_regret_exploitation = np.mean(regret_exploitation, axis=0)
    avg_regret_exploration = np.mean(regret_exploration, axis=0)
    avg_regret_epsilonGreedy = np.mean(regret_epsilonGreedy, axis=0)
    avg_regret_decayingEpsilonGreedy = np.mean(regret_decayingEpsilonGreedy, axis=0)
    avg_regret_softmax = np.mean(regret_softmax, axis=0)
    avg_regret_ucb = np.mean(regret_ucb, axis=0)

    plt.figure(figsize=(12,8))
    plt.rcParams.update({'font.size': 14})
    plt.plot(episodes, avg_regret_exploitation, label='Pure Exploitation',linewidth=2)
    plt.plot(episodes, avg_regret_exploration, label='Pure Exploration',linewidth=2)
    plt.plot(episodes, avg_regret_epsilonGreedy, label='Epsilon Greedy',linewidth=2)
    plt.plot(episodes, avg_regret_decayingEpsilonGreedy, label='Deacying Epsilon',linewidth=2)
    plt.plot(episodes, avg_regret_softmax, label='Softmax',linewidth=2)
    plt.plot(episodes, avg_regret_ucb, label='UCB',linewidth=2)
    plt.title('Average Regret for 50 Ten Armed Gaussian Bandit Environments')
    plt.xlabel('Time Steps')
    plt.ylabel('Average regret')
    plt.legend()
    plt.savefig('Q1P7.svg')
    plt.savefig('Q1P7.jpg', dpi=300)
    plt.show()