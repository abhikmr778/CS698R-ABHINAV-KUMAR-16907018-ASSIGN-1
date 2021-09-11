"""
2. Similarly, in OpenAI Gym create the environment for 10-armed Gaussian Bandit. Make sure it is executing
as expected by creating certain test cases, e.g., by playing with σ. Report about your test cases and how
they point towards the correct implementation. You can also report about your general observations.

Ten Arm Gaussian Bandit env id for gym.make(): tenArmGaussian_bandits-v0
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

    sigma = 1
    print('-------Testing Ten Arm Bandit Environment-------')
    for i in range(10):
        SEED = i
        print(f'************ SEED={SEED} ************')
        env = gym.make('tenArmGaussian_bandits-v0', sigma_square=sigma, seed=SEED)
        env.reset()
        print(f'Greedy Policy Evaluation')
        print(f'Optimal Q values: {env.q_value}')
        print(f'Rewards corresponding to optimal Q values: {env.rewards}')
        a = np.argmax(env.q_value)
        print(f'Greedy Action: {a}')
        greedy_policy = np.zeros(env.action_space.n)
        greedy_policy[a] = 1
        print(f'Greedy Policy Distribution: {greedy_policy}')
        policyEvluator = policyEvaluation(greedy_policy)
        policyEvluator.evaluate(env)

        print('**********************************************')
        