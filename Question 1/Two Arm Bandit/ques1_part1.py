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
from policyevaluation import policyEvaluation

if __name__ == "__main__":

    # parameters
    SEED = 0
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
        