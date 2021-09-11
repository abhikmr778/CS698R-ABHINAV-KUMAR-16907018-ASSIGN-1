import gym
import numpy as np
import random_walk

def generateTrajectory(env, policy, maxSteps):
    """
    Input: env - environment
           policy - policy to tell what actions to take in a state
           maxSteps - no of steps within which trajectory should terminate
    
    Output: experiences - collection of tuples
    """
    # to store trajectory
    experiences = [] 
    trajectory_end_flag = False

    # simulate a trajectory
    state = env.reset()
    for i in range(maxSteps):
        action = policy[state]
        next_state, reward, done, _ = env.step(action)
        experiences.append((state, action, next_state, reward, done))
        state = next_state
        if done==True:
            trajectory_end_flag = True
            break
    
    # if trajectory didn't terminate, return empty list
    if trajectory_end_flag == False:
        return []

    return experiences


if __name__ == '__main__':
    
    # parameters
    SEED = 0
    maxSteps = 10

    print('-----------------Generate Trajectory---------------------')
    for i in range(10):
        print(f'Seed Value = {SEED}')
        
        # create env
        env = gym.make('random_walk-v0', seed=SEED)
        
        # generate a random policy
        random_policy = np.array([int(np.random.randint(0,2,1)[0]) for i in range(env.observation_space.n)])
        
        # generate trajectory
        trajectory = generateTrajectory(env,random_policy, maxSteps)
        
        # print trajectory
        for experience in trajectory:
            print(experience, end="")
        print()

        # increment seed
        SEED = SEED + 1