"""
2. Similarly, in OpenAI Gym create the environment for 10-armed Gaussian Bandit. Make sure it is executing
as expected by creating certain test cases, e.g., by playing with σ. Report about your test cases and how
they point towards the correct implementation. You can also report about your general observations.

Ten Arm Gaussian Bandit env id for gym.make(): tenArmGaussian_bandits-v0
"""


import gym
import custom_bandits
import numpy as np
from policyevaluation import policyEvaluation

if __name__ == "__main__":

    # parameters
    SEED = 0
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
        policyEvaluator = policyEvaluation(greedy_policy)
        policyEvaluator.evaluate(env)

        print('**********************************************')
        