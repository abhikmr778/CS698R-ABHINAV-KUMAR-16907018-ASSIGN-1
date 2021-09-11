"""
1. In OpenAI Gym create the environment for 2-armed Bernoulli Bandit. The environment should take α
and β as input parameters and simulate 2-armed bandit accordingly. Once you have implemented the
environment, run it using different values of α and β to make sure it is executing as expected. For, example,
you can try with (α, β) = (0, 0),(1, 0),(0, 1),(1, 1),(0.5, 0.5), etc. Report about your test cases and how
they point towards the correct implementation. You can also report about your general observations.

Two Arm Bandit env id for gym.make(): twoArm_bandits-v0
"""


import gym
import custom_bandits
import numpy as np

SEED = 0

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

if __name__ == "__main__":

    LEFT = 0
    RIGHT = 1    
    alphas = [0,1,0,1,0.5]
    betas = [0,0,1,1,0.5]

    # always go left policy
    left_policy = {
        LEFT: 1,
        RIGHT: 0
    }
    # always go right policy
    right_policy = {
        LEFT: 0,
        RIGHT: 1
    }

    print('-------Testing Two Arm Bandit Environment-------')
    for alpha,beta in zip(alphas, betas):
        print()
        print(f'************ alpha={alpha} and beta={beta} ************')
        env = gym.make('twoArm_bandits-v0', alpha=alpha, beta=beta, seed=SEED)
        
        print(f'Left Policy Evaluation')
        left_policyEvaluator = policyEvaluation(left_policy)
        left_policyEvaluator.evaluate(env)
        
        print(f'Right Policy Evaluation')
        right_policyEvaluator = policyEvaluation(right_policy)
        right_policyEvaluator.evaluate(env)
        print('**********************************************')
        