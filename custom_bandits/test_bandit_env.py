import gym
import custom_bandits
import numpy as np

SEED = 0
env = gym.make('twoArm_bandits-v0', alpha=0.8, beta=0.2, seed=SEED)

class policyEvaluation:

    def __init__(self):
        LEFT = 0
        RIGHT = 1
        # Using left policy
        self.pi = {}
        self.pi[LEFT] = 1 # prob of going left is 1
        self.pi[RIGHT] = 0 # prob of going right is 0
        
        # taking gamma as 0.99
        self.gamma = 0.99

        # threshold
        self.theta = 1e-3

    def evaluate(self, env):
        # randomly initialize old Value estimates, here initializing to 0
        Vold = np.zeros(env.observation_space.n)
        
        max_iterations = 100
        for i in range(max_iterations):
            Vnew = np.zeros(env.observation_space.n)
            for s in range(env.observation_space.n): # for all states
                for a in range(env.action_space.n): # for all actions in each state
                    temp = 0
                    
                    for p,s_,r,d in env.P[s][a]: # for all dynamics 
                        # inner summation over next state and reward
                        if not d:
                            temp += p*(r+self.gamma*Vold[s_])
                        else:
                            temp += p*r
                            # print('state:', s, 'action:', a, p, s_, 'reward:', r)

                    Vnew[s] += self.pi[a]*temp # outermost summation over policy
                    
            if np.max(np.abs(Vnew-Vold)) < self.theta:
                break 
            Vold = Vnew

        for i in range(len(Vnew)):
            print(f'Value of state {i} is {np.round(Vnew[i],2)}')


policyEvaluator = policyEvaluation()
policyEvaluator.evaluate(env)