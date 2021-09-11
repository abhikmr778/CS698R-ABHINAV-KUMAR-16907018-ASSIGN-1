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

    # to store rewards across environments for all 6 agents
    reward_exploitation = np.zeros((noOfEnvs,timeSteps))
    reward_exploration = np.zeros((noOfEnvs,timeSteps))
    reward_epsilonGreedy = np.zeros((noOfEnvs,timeSteps))
    reward_decayingEpsilonGreedy = np.zeros((noOfEnvs,timeSteps))
    reward_softmax = np.zeros((noOfEnvs,timeSteps))
    reward_ucb = np.zeros((noOfEnvs,timeSteps))

    # for every env
    for i in tqdm(range(noOfEnvs),ascii=True, unit=" env "):

        # skip 42
        if SEED==ANSWER_TO_EVERYTHING:
            SEED = SEED + 1
            create_Earth(ANSWER_TO_EVERYTHING) # read hitchhiker's guide to galaxy if not yet read

        np.random.seed(SEED)

        # generating sigma
        sigma = np.random.uniform(0,2.5,1)

        # create env
        env = gym.make('tenArmGaussian_bandits-v0', sigma_square=sigma, seed=SEED)
        env.reset()

        # store reward history for every env
        _, _, _, reward_exploitation[i], _ = pureExploitation(env, timeSteps)
        _, _, _, reward_exploration[i], _ = pureExploration(env, timeSteps)
        _, _, _, reward_epsilonGreedy[i], _ = epsilonGreedy(env, timeSteps, epsilon=0.1)
        _, _, _, reward_decayingEpsilonGreedy[i], _ = decayingEpsilonGreedy(env, timeSteps, decay_till=timeSteps/2, max_epsilon=1, min_epsilon=1e-6, decay_type='exp')
        _, _, _, reward_softmax[i], _ = softmaxExploration(env, timeSteps, decay_till=timeSteps/2, max_tau=100, min_tau = 0.005, decay_type='exp')
        _, _, _, reward_ucb[i], _ = UCB(env, timeSteps, c=1)

        print(f'    Seed: {SEED} || sigma: {sigma} ')
        
        # increment seed
        SEED = SEED + 1

    # average out results across environments and smooth the values and plot
    episodes = [i for i in range(timeSteps)]
    smooth_window = 50
    avg_reward_exploitation = smooth_array(np.mean(reward_exploitation, axis=0), smooth_window)
    avg_reward_exploration = smooth_array(np.mean(reward_exploration, axis=0), smooth_window)
    avg_reward_epsilonGreedy = smooth_array(np.mean(reward_epsilonGreedy, axis=0), smooth_window)
    avg_reward_decayingEpsilonGreedy = smooth_array(np.mean(reward_decayingEpsilonGreedy, axis=0), smooth_window)
    avg_reward_softmax = smooth_array(np.mean(reward_softmax, axis=0), smooth_window)
    avg_reward_ucb = smooth_array(np.mean(reward_ucb, axis=0), smooth_window)

    plt.figure(figsize=(12,8))
    plt.rcParams.update({'font.size': 14})
    plt.plot(episodes, avg_reward_exploitation, label='Pure Exploitation', linewidth=2)
    plt.plot(episodes, avg_reward_exploration, label='Pure Exploration', linewidth=2)
    plt.plot(episodes, avg_reward_epsilonGreedy, label='Epsilon Greedy', linewidth=2)
    plt.plot(episodes, avg_reward_decayingEpsilonGreedy, label='Deacying Epsilon', linewidth=2)
    plt.plot(episodes, avg_reward_softmax, label='Softmax', linewidth=2)
    plt.plot(episodes, avg_reward_ucb, label='UCB', linewidth=2)
    plt.title('Average Reward for 50 Ten Armed Gaussian Environments')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.savefig('q1p5_4.svg')
    plt.savefig('q1p45_.jpg', dpi=300)
    plt.show()