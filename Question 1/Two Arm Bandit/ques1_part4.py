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

    timeSteps = 1000
    noOfEnvs = 50

    reward_exploitation = np.zeros((noOfEnvs,timeSteps))
    reward_exploration = np.zeros((noOfEnvs,timeSteps))
    reward_epsilonGreedy = np.zeros((noOfEnvs,timeSteps))
    reward_decayingEpsilonGreedy = np.zeros((noOfEnvs,timeSteps))
    reward_softmax = np.zeros((noOfEnvs,timeSteps))
    reward_ucb = np.zeros((noOfEnvs,timeSteps))

    for i in tqdm(range(noOfEnvs),ascii=True, unit=" env "):

        if SEED==ANSWER_TO_EVERYTHING:
            SEED = SEED + 1
            create_Earth(ANSWER_TO_EVERYTHING)

        np.random.seed(SEED)

        alpha = np.random.uniform()
        beta = np.random.uniform()

        env = gym.make('twoArm_bandits-v0', alpha=alpha, beta=beta, seed=SEED)
        env.reset()

        _, _, _, reward_exploitation[i], _ = pureExploitation(env, timeSteps)
        _, _, _, reward_exploration[i], _ = pureExploration(env, timeSteps)
        _, _, _, reward_epsilonGreedy[i], _ = epsilonGreedy(env, timeSteps, epsilon=0.1)
        _, _, _, reward_decayingEpsilonGreedy[i], _ = decayingEpsilonGreedy(env, timeSteps, decay_till=timeSteps/2, max_epsilon=1, min_epsilon=1e-6, decay_type='exp')
        _, _, _, reward_softmax[i], _ = softmaxExploration(env, timeSteps, decay_till=timeSteps/2, max_tau=100, min_tau = 0.005, decay_type='exp')
        _, _, _, reward_ucb[i], _ = UCB(env, timeSteps, c=0.1)

        print(f'    Seed: {SEED} || alpha: {alpha} || beta: {beta}')
        
        SEED = SEED + 1

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
    plt.plot(episodes, avg_reward_exploitation, label='Pure Exploitation')
    plt.plot(episodes, avg_reward_exploration, label='Pure Exploration')
    plt.plot(episodes, avg_reward_epsilonGreedy, label='Epsilon Greedy')
    plt.plot(episodes, avg_reward_decayingEpsilonGreedy, label='Deacying Epsilon')
    plt.plot(episodes, avg_reward_softmax, label='Softmax')
    plt.plot(episodes, avg_reward_ucb, label='UCB')
    plt.title('Average Reward for 50 Two Arm Bandit Environments')
    plt.xlabel('Time Steps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.savefig('q1p4.svg')
    plt.savefig('q1p4.jpg', dpi=300)
    plt.show()