import gym
from tqdm import tqdm
import custom_bandits
import numpy as np
import matplotlib.pyplot as plt
from ques1_part5_3a_pureExploitation import pureExploitation
from ques1_part5_3b_pureExploration import pureExploration
from ques1_part5_3c_epsilonGreedy import epsilonGreedy
from ques1_part5_3d_DecayingEpsilonGreedy import decayingEpsilonGreedy
from ques1_part5_3e_softmax import softmaxExploration
from ques1_part5_3f_ucb import UCB

def smooth_array(data, window):
  # utility function taken from github
  alpha = 2 /(window + 1.0)
  alpha_rev = 1-alpha
  n = data.shape[0]

  pows = alpha_rev**(np.arange(n+1))

  scale_arr = 1/pows[:-1]
  offset = data[0]*pows[1:]
  pw0 = alpha*alpha_rev**(n-1)

  mult = data*pw0*scale_arr
  cumsums = mult.cumsum()
  out = offset + cumsums*scale_arr[::-1]
  return out


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

    optimalAction_exploitation = np.zeros((noOfEnvs,timeSteps))
    optimalAction_exploration = np.zeros((noOfEnvs,timeSteps))
    optimalAction_epsilonGreedy = np.zeros((noOfEnvs,timeSteps))
    optimalAction_decayingEpsilonGreedy = np.zeros((noOfEnvs,timeSteps))
    optimalAction_softmax = np.zeros((noOfEnvs,timeSteps))
    optimalAction_ucb = np.zeros((noOfEnvs,timeSteps))

    for i in tqdm(range(noOfEnvs),ascii=True, unit=" time-step "):

        if SEED==ANSWER_TO_EVERYTHING:
            SEED = SEED + 1
            print('')
        np.random.seed(SEED)

        sigma = np.random.uniform(0,2.5,1)

        env = gym.make('tenArmGaussian_bandits-v0', sigma_square=sigma, seed=SEED)
        env.reset()

        _, optimalAction_exploitation[i], _, _ = pureExploitation(env, timeSteps)

        _, optimalAction_exploration[i], _, _ = pureExploration(env, timeSteps)
        _, optimalAction_epsilonGreedy[i], _, _ = epsilonGreedy(env, timeSteps, epsilon=0.1)
        _, optimalAction_decayingEpsilonGreedy[i], _, _ = decayingEpsilonGreedy(env, timeSteps, max_epsilon=1, decay_type='exp')
        _, optimalAction_softmax[i], _, _ = softmaxExploration(env, timeSteps, tau=1e5, decay_type='lin')
        _, optimalAction_ucb[i], _, _ = UCB(env, timeSteps, c=0.4)

        print(f'    Seed: {SEED} || sigma: {sigma} ')
        
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
    plt.plot(episodes, avg_optimalAction_exploitation, label='Pure Exploitation')
    plt.plot(episodes, avg_optimalAction_exploration, label='Pure Exploration')
    plt.plot(episodes, avg_optimalAction_epsilonGreedy, label='Epsilon Greedy')
    plt.plot(episodes, avg_optimalAction_decayingEpsilonGreedy, label='Deacying Epsilon')
    plt.plot(episodes, avg_optimalAction_softmax, label='Softmax')
    plt.plot(episodes, avg_optimalAction_ucb, label='UCB')
    plt.title('Percent Optimal Action for 50 Ten Arm Gaussian Bandit Environments')
    plt.xlabel('Time Steps')
    plt.ylabel('Percent Optimal Action')
    plt.legend()
    plt.savefig('Q1P9.svg')
    plt.savefig('Q1P9.jpg', dpi=300)
    plt.show()