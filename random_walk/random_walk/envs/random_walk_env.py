###########################################################################
#                                                                         #
#           Environment Class template followed from                      #
# https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html #
#                                                                         #
###########################################################################

import gym
from gym import spaces
import numpy as np

class RandomWalk(gym.Env):
    def __init__(self, alpha=0.5, beta=0.5, seed=0):
        super(RandomWalk, self).__init__()
        
        LEFT = 0
        RIGHT = 1
        N_DISCRETE_ACTIONS = 2
        N_DISCRETE_STATES = 7
        
        self.seed(seed)
        self.alpha = alpha
        self.beta = beta
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Discrete(N_DISCRETE_STATES)
        self.agent_position = self.reset()

        # MDP dynamics
        P = {}
        # non terminal intermediate states
        for i in range(2,5):
            P[i] = {
                LEFT: [(self.alpha, i-1, 0, False), (1-self.alpha, i+1, 0, False)],
                RIGHT: [(self.beta, i+1, 0, False), (1-self.beta, i-1, 0, False)]
            }
        # states adjacent to terminal states
        P[1] = {
            LEFT: [(self.alpha, 0, 0, True), (1-self.alpha, 2, 0, False)],
            RIGHT: [(self.beta, 2, 0, False), (1-self.beta, 0, 0, True)]
        }
        P[5] = {
            LEFT: [(self.alpha, 4, 0, False), (1-self.alpha, 6, 1, True)],
            RIGHT: [(self.beta, 6, 1, True), (1-self.beta, 4, 0, False)]
        }
        # terminal states
        for i in [0,6]:
            P[i] = {
                LEFT: [(1, i, 0, True)],
                RIGHT: [(1, i, 0, True)]
            }
        self.P = P

    def step(self, action):
        # terminal state do nothing
        if self.agent_position == 0 or self.agent_position == 6:
            return self.agent_position, 0, True, {}
        # non terminal state, choose action according to alpha = beta = 0.5
        else:
            if action == 0:
                if np.random.uniform() < self.alpha:
                    self.agent_position -= 1 
                    reward = 0
                    done = False
                    if self.agent_position == 0:
                        done = True
                else:
                    self.agent_position += 1
                    reward = 0
                    done = False
                    if self.agent_position == 6:
                        reward = 1
                        done = True
            if action == 1:
                if np.random.uniform() < self.beta:
                    self.agent_position += 1 
                    reward = 0
                    done = False
                    if self.agent_position == 0:
                        done = True
                else:
                    self.agent_position -= 1
                    reward = 0
                    done = False
                    if self.agent_position == 6:
                        reward = 1
                        done = True
        info = {}
        return self.agent_position, reward, done, info

    def reset(self):
        self.agent_position = int(np.random.randint(1,6,1)[0])
        return self.agent_position  
    
    def seed(self, seed=0):
        np.random.seed(seed)

    def render(self, mode='human'):
        raise NotImplementedError

    def close (self):
        raise NotImplementedError