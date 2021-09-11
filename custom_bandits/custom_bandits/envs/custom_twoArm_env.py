###########################################################################
#                                                                         #
#           Environment Class template followed from                      #
# https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html #
#                                                                         #
###########################################################################

import gym
from gym import spaces
import numpy as np

class TwoArmBandit(gym.Env):
    
    # metadata = {'render.modes':['human']}
    

    def __init__(self, alpha, beta, seed):
        super(TwoArmBandit, self).__init__()
        N_DISCRETE_ACTIONS = 2
        LEFT = 0
        RIGHT = 1
        N_DISCRETE_STATES = 3
        self.alpha = alpha
        self.beta = beta
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Discrete(N_DISCRETE_STATES)
        
        # Data structure to store MDP of 2-arm Bernoulli Bandit
        self.P = {}
        self.P[0] = {
                        LEFT: [[self.alpha, 1, 0, True], [1-self.alpha, 2, 1, True]],
                        RIGHT: [[self.beta, 2, 1, True], [1-self.beta, 1, 0, True]]
                    }
        self.P[1] = {
                        LEFT: [[1,1,0,True]],
                        RIGHT: [[1,1,0,True]]
                    }
        self.P[2] = {
                        LEFT: [[1,2,0,True]],
                        RIGHT: [[1,2,0,True]]
                    }
        self.q_value = np.array([1-self.alpha,self.beta])
        self.agent_position = self.reset()
        self.seed(seed)

    def step(self, action):
        # get transition from MDP dynamics
        probabilities = []
        next_states = []
        rewards = []
        dones = []

        # sample transition according to the probabilities of transition function
        for dynamic in self.P[self.agent_position][action]:
            probabilities.append(dynamic[0])
        idx = [i for i in range(len(self.P[self.agent_position][action]))]
        
        j = int(np.random.choice(a=idx,size=1,p=probabilities))
        
        _, observation, reward, done = self.P[self.agent_position][action][j]
        
        # update agent's position
        self.agent_position = observation
        info = {}
        return observation, reward, done, info

    def reset(self):
        self.agent_position = 0
        return self.agent_position  # reward, done, info can't be included
    
    def seed(self, seed=0):
        np.random.seed(seed)

    def render(self, mode='human'):
        raise NotImplementedError

    def close (self):
        raise NotImplementedError
