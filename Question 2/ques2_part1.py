import gym
import numpy as np
import random_walk

def generateTrajectory(env, policy, maxSteps):

    experiences = []
    trajectory_end_flag = False

    state = env.reset()
    for i in range(maxSteps):
        action = policy[state]
        next_state, reward, done, _ = env.step(action)
        experiences.append((state, action, next_state, reward, done))
        state = next_state
        if done==True:
            trajectory_end_flag = True
            break
    
    if trajectory_end_flag == False:
        return []

    return experiences


if __name__ == '__main__':
    
    SEED = 0
    for i in range(10):
        print(f'Seed Value = {SEED}')
        env = gym.make('random_walk-v0', seed=SEED)

        maxSteps = 10
        random_policy = np.array([int(np.random.randint(0,2,1)[0]) for i in range(env.observation_space.n)])
        trajectory = generateTrajectory(env,random_policy, maxSteps)
        for experience in trajectory:
            print(experience, end="")
        print()
        SEED = SEED + 1