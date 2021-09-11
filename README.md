# CS698R-ABHINAV-KUMAR-16907018-ASSIGN-1

## Directory Structure
- CS698R-Abhinav-Kumar-16907018-ASSIGN-1/
    - custum_bandits/
        - setup.py
        - custum_bandits/
            - \_\_init\_\_.py
            - envs/
                - \_\_init\_\_.py
                - custom_tenArm_env.py
                - custom_twoArm_env.py
    - random_walk/
        - setup.py
        - random_walk/
            - \_\_init\_\_.py
            - envs/
                - \_\_init\_\_.py
                - random_walk_env.py

    - Question 1/
        - Ten Arm Bandit/ .. solutions ..
        - Two Arm Bandit/ .. solutions ..
    - Question 2/ .. solutions ..

    - requirements.txt

## How to install environments
- Download the zip, extract
- Move to the parent directory CS698R-Abhinav-Kumar-16907018-ASSIGN-1/
- Run 'pip install custom_bandits' to install Two-Armed Bernoulli Bandit and Ten-Armed Gaussian Bandit
- Run 'pip install random_walk' to install the Random Walk Environment

## Creating environments
- import gym
- import custom_bandit
- import random_walk
- For Two-Armed Bernoulli Bandit use gym.make('twoArm_bandits-v0', alpha=alpha, beta=beta, seed=SEED)
- For Ten-Armed Gaussian Bandit use gym.make('tenArmGaussian_bandits-v0', sigma_square=sigma, seed=SEED)
- For Random Walk use gym.make('random_walk-v0', seed=SEED)

## About Environments
- all environments Two Arm Bernoulli Bandit, Ten Arm Gaussian Bandit and Random Walk contain underlying MDP data structure as was mentioned in lectures
- this MDP data structure is used for true value calculations


## Question 1
- contains two directories:
    - Ten Arm Bandit: contains solutions to all parts involving ten arm bandit
    - Two Arm Bandit: contains solutions to all parts involving two arm bandit
- agents.py contains all 6 agents well documented
- utils.py contains a function smooth_array to smooth out values for plotting and follow up references for the reference to 42 mentioned in the assignment 
- policyevaluation.py contains a function to evaluate a policy using MDP of the env

## Question 2
- contains solutions to Random Walk environment
- solutions contain function definitions and are documented

## Dependencies

- python 3

Install the following using pip

- numpy
- gym
- matplotlib
- scipy
- tqdm

Or run using conda

- $ conda create --name <env_name> --file requirements.txt
