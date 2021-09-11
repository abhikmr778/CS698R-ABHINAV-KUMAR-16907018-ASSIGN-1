import numpy as np
import gym

class policyEvaluation:

    def __init__(self, policy, gamma=0.99, theta=1e-3, max_iterations=100):
        # policy distn
        self.pi = policy
        
        # taking gamma as 0.99
        self.gamma = 0.99

        # threshold
        self.theta = 1e-3

        # max_iterations
        self.max_iterations = max_iterations

    def evaluate(self, env):
        # randomly initialize old Value estimates, here initializing to 0
        Vold = np.zeros(env.observation_space.n)
        
        for i in range(self.max_iterations):
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
            if i==0:
                print(f'    Value of initial state {i} is {np.round(Vnew[i],2)}')
            else:
                print(f'    Value of terminal state {i} is {np.round(Vnew[i],2)}')
        return Vnew
    
    def __repr__(self):
        return 'policyEvaluation(policy={}, gamma={}, theta={}, max_iterations={})'.format(self.policy, self.gamma, self.theta, self.max_iterations)