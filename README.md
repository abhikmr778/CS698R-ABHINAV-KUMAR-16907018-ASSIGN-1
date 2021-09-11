# CS698R-Abhinav-Kumar-16907018-ASSIGN-1

# # How to install environments
- Download the zip, extract
- Move to the parent directory CS698R-Abhinav-Kumar-16907018-ASSIGN-1
- Run 'pip install custom_bandits' to install Two-Armed Bernoulli Bandit and Ten-Armed Gaussian Bandit
- Run 'pip install random_walk' to install the Random Walk Environment

# # Creating environments
- import gym
- import custom_bandit
- For Two-Armed Bernoulli Bandit use gym.make('twoArm_bandits-v0', alpha=alpha, beta=beta, seed=SEED)
- For Ten-Armed Gaussian Bandit use gym.make('tenArmGaussian_bandits-v0', sigma_square=sigma, seed=SEED)

- import random_walk
- For Random Walk use gym.make('random_walk-v0', seed=SEED)