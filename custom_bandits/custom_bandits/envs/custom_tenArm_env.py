###########################################################################
#                                                                         #
#           Environment Class template followed from                      #
# https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html #
#                                                                         #
###########################################################################

import gym
from gym import spaces
import numpy as np

class TenArmGaussianBandit(gym.Env):
    
    # metadata = {'render.modes':['human']}
    

    def __init__(self, mu = 0, sigma_square=1, seed=0):
        super(TenArmGaussianBandit, self).__init__()

        N_DISCRETE_ACTIONS = 10
        N_DISCRETE_STATES = 11
        
        self.seed(seed)

        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Discrete(N_DISCRETE_STATES)
        
        # sample from a gaussian 10 times to create the 10 arm gaussian bandit
        self.mu = mu
        self.sigma_square = sigma_square
        self.q_value = np.random.normal(self.mu, np.sqrt(self.sigma_square), self.action_space.n)

        # sample rewards at each time step from a gaussian with mean as q_value and given sigma
        self.rewards = np.random.normal(self.q_value, np.sqrt(self.sigma_square), self.action_space.n)

        self.P = self.set_MDP()
        self.agent_position = self.reset()

    def set_MDP(self):
        # sets the P data structure as told in lectures
        P = {}
        for i in range(0,self.observation_space.n):
            P[i] = {}
        for i in range(0,self.action_space.n):
            P[0][i] = [(1,i+1,self.rewards[i],True)]
        
        for i in range(1,self.observation_space.n):
            for j in range(0,self.action_space.n):
                P[i][j] = [(1,i,0,True)]
        return P
    
    def step(self, action):

        if self.agent_position != 0: # if terminal state then do nothing
            return self.agent_position, 0, True, {}

        else: # if non-terminal state then sample rewards and collect experience
            self.rewards = np.random.normal(self.q_value, np.sqrt(self.sigma_square), self.action_space.n)
            self.set_MDP()
            self.agent_position = action+1
            reward = self.rewards[action]
            done = True
            info = {}
            return self.agent_position, reward, done, info

    def reset(self):
        self.agent_position = 0
        return self.agent_position 
    
    def seed(self, seed=0):
        np.random.seed(seed)

    def render(self, mode='human'):
        raise NotImplementedError

    def close (self):
        raise NotImplementedError
