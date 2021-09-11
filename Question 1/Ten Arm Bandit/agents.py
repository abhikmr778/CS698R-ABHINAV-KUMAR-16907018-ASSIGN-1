import gym
import numpy as np
from scipy.special import softmax

##################################################################################################################
#                                                                                                                #
#                                     Pure Exploitation Agent                                                    #
#                                                                                                                #
##################################################################################################################

def pureExploitation(env, maxEpisodes):
    """
    Follows function definition from lectures

    Input: env - "openAI gym environment", maxEpisodes - "maximum no of episodes for which algorithm runs"
    
    Output: Q_est - "history of Q value estimates", a_history - "history of actions taken", optimal_action_history - "history of percentage optimal action"
            r_history - "reward history", regret_history - "cumulative history of regret"
    
    Variable Definitions:
        Q: contains Q value estimates
        N: Contains count of taking an action for all actions
        Q_est: tracks history of Q 

        a_history: tracks history of taken actions
        r_history: tracks history of rewards received
        regret_history: tracks cumulative regret
    
        optimal_action: stores optimal action
        optimal_action_history: tracks percentage optimal action history
    
    """
        Q = np.zeros(env.action_space.n)
        N = np.zeros(env.action_space.n)
        Q_est = np.zeros((maxEpisodes, env.action_space.n))
        a_history = np.zeros(maxEpisodes)
        r_history = np.zeros(maxEpisodes)
        regret_history = np.zeros(maxEpisodes)

        optimal_action = np.argmax(env.q_value)
        optimal_action_history = np.zeros(maxEpisodes)

        for i in range(maxEpisodes):
            # algorithm follows lectures
            env.reset()
            a = np.argmax(Q)
            _, R, _, _ = env.step(a)
            N[a] = N[a] + 1
            Q[a] = Q[a] + (R-Q[a])/N[a]
            Q_est[i] = Q
            a_history[i] = a
            r_history[i] = R

            # regret calculation
            if i==0:
                regret_history[i] = env.q_value[optimal_action] - env.q_value[a]
            else:
                regret_history[i] = regret_history[i-1] + (env.q_value[optimal_action] - env.q_value[a])
            
            # tracking percentage of optimal actions taken
            if a==optimal_action:
                optimal_action_history[i] = N[a]/(i+1)

        return Q_est, a_history, optimal_action_history, r_history, regret_history

##################################################################################################################


##################################################################################################################
#                                                                                                                #
#                                     Pure Exploration Agent                                                     #
#                                                                                                                #
##################################################################################################################
def pureExploration(env, maxEpisodes):
    """
    Follows function definition from lectures

    Input: env - "openAI gym environment", maxEpisodes - "maximum no of episodes for which algorithm runs"
    
    Output: Q_est - "history of Q value estimates", a_history - "history of actions taken", optimal_action_history - "history of percentage optimal action"
            r_history - "reward history", regret_history - "cumulative history of regret"
    
    Variable Definitions:
        Q: contains Q value estimates
        N: Contains count of taking an action for all actions
        Q_est: tracks history of Q 
        
        a_history: tracks history of taken actions
        r_history: tracks history of rewards received
        regret_history: tracks cumulative regret
    
        optimal_action: stores optimal action
        optimal_action_history: tracks percentage optimal action history
    
    """
        Q = np.zeros(env.action_space.n)
        N = np.zeros(env.action_space.n)
        Q_est = np.zeros((maxEpisodes, env.action_space.n))
        a_history = np.zeros(maxEpisodes)
        r_history = np.zeros(maxEpisodes)
        regret_history = np.zeros(maxEpisodes)

        optimal_action = np.argmax(env.q_value)
        optimal_action_history = np.zeros(maxEpisodes)

        for i in range(maxEpisodes):
            # follows algorithm from lectures
            env.reset()
            a = np.random.randint(0,env.action_space.n,1)[0]
            _, R, _, _ = env.step(a)
            N[a] = N[a] + 1
            Q[a] = Q[a] + (R-Q[a])/N[a]
            Q_est[i] = Q
            a_history[i] = a
            r_history[i] = R

            # regret calculation
            if i==0:
                regret_history[i] = env.q_value[optimal_action] - env.q_value[a]
            else:
                regret_history[i] = regret_history[i-1] + (env.q_value[optimal_action] - env.q_value[a])
            
            # percentage optimal action calculation
            if a==optimal_action:
                optimal_action_history[i] = N[a]/(i+1)

        return Q_est, a_history, optimal_action_history, r_history, regret_history

##################################################################################################################


##################################################################################################################
#                                                                                                                #
#                                        Epsilon Greedy Agent                                                    #
#                                                                                                                #
##################################################################################################################
def epsilonGreedy(env, maxEpisodes, epsilon):
    """
    Follows function definition from lectures

    Input: env - "openAI gym environment", maxEpisodes - "maximum no of episodes for which algorithm runs", epsilon - "probability with which to explore"
    
    Output: Q_est - "history of Q value estimates", a_history - "history of actions taken", optimal_action_history - "history of percentage optimal action"
            r_history - "reward history", regret_history - "cumulative history of regret"
    
    Variable Definitions:
        Q: contains Q value estimates
        N: Contains count of taking an action for all actions
        Q_est: tracks history of Q 
        
        a_history: tracks history of taken actions
        r_history: tracks history of rewards received
        regret_history: tracks cumulative regret
    
        optimal_action: stores optimal action
        optimal_action_history: tracks percentage optimal action history
    
    """
        Q = np.zeros(env.action_space.n)
        N = np.zeros(env.action_space.n)
        Q_est = np.zeros((maxEpisodes, env.action_space.n))
        a_history = np.zeros(maxEpisodes)
        r_history = np.zeros(maxEpisodes)
        regret_history = np.zeros(maxEpisodes)

        optimal_action = np.argmax(env.q_value)
        optimal_action_history = np.zeros(maxEpisodes)

        for i in range(maxEpisodes):
            # follows algorithm from lectures
            env.reset()

            # epsilon greedy action selection
            if np.random.uniform() < epsilon:
                a = np.random.randint(0,env.action_space.n,1)[0]
            else:
                a = np.argmax(Q)
            
            _, R, _, _ = env.step(a)
            N[a] = N[a] + 1
            Q[a] = Q[a] + (R-Q[a])/N[a]
            Q_est[i] = Q
            a_history[i] = a
            r_history[i] = R
            
            # regret calculation
            if i==0:
                regret_history[i] = env.q_value[optimal_action] - env.q_value[a]
            else:
                regret_history[i] = regret_history[i-1] + (env.q_value[optimal_action] - env.q_value[a]) 
            
            # percentage optimal action
            if a==optimal_action:
                optimal_action_history[i] = N[a]/(i+1)

        return Q_est, a_history, optimal_action_history, r_history, regret_history


##################################################################################################################


##################################################################################################################
#                                                                                                                #
#                                        Decaying Epsilon Greedy Agent                                           #
#                                                                                                                #
##################################################################################################################
def decayingEpsilonGreedy(env, maxEpisodes, decay_till, max_epsilon=1, min_epsilon=1e-6, decay_type='exp'):
    """
    Follows function definition from lectures

    Input: env - "openAI gym environment", maxEpisodes - "maximum no of episodes for which algorithm runs", decay_till - "episode till which to decay epsilon"
           max_epsilon - "starting value of epsilon", min_epsilon - "final value of epsilon", decay_type - "linear or exponential decay"
           
    Output: Q_est - "history of Q value estimates", a_history - "history of actions taken", optimal_action_history - "history of percentage optimal action"
            r_history - "reward history", regret_history - "cumulative history of regret"
    
    Variable Definitions:
        Q: contains Q value estimates
        N: Contains count of taking an action for all actions
        Q_est: tracks history of Q 
        
        epsilon: probability with which to explore

        a_history: tracks history of taken actions
        r_history: tracks history of rewards received
        regret_history: tracks cumulative regret
    
        optimal_action: stores optimal action
        optimal_action_history: tracks percentage optimal action history
    
    """
        Q = np.zeros(env.action_space.n)
        N = np.zeros(env.action_space.n)
        Q_est = np.zeros((maxEpisodes, env.action_space.n))
        a_history = np.zeros(maxEpisodes)
        r_history = np.zeros(maxEpisodes)
        epsilon_history = np.zeros(maxEpisodes)
        regret_history = np.zeros(maxEpisodes)

        optimal_action = np.argmax(env.q_value)
        optimal_action_history = np.zeros(maxEpisodes)

        # epsilon setup
        epsilon = max_epsilon
        exp_decay_rate = np.power(min_epsilon/max_epsilon,(1/(decay_till))) # decay rate for exponential
        lin_decay_rate = (max_epsilon - min_epsilon)/decay_till # decay rate for linear

        for i in range(maxEpisodes):
            # follows algorithm from lectures
            env.reset()
            if np.random.uniform() < epsilon:
                a = np.random.randint(0,env.action_space.n,1)[0]
            else:
                a = np.argmax(Q)
            _, R, _, _ = env.step(a)
            N[a] = N[a] + 1
            Q[a] = Q[a] + (R-Q[a])/N[a]
            Q_est[i] = Q
            a_history[i] = a
            r_history[i] = R

            # epsilon decay
            if epsilon > min_epsilon:
                if decay_type == 'exp':
                    epsilon = epsilon * exp_decay_rate
                elif decay_type == 'lin':
                    epsilon = max(epsilon - lin_decay_rate, min_epsilon)
                epsilon_history[i] = epsilon

            # regret calculation
            if i==0:
                regret_history[i] = env.q_value[optimal_action] - env.q_value[a]
            else:
                regret_history[i] = regret_history[i-1] + (env.q_value[optimal_action] - env.q_value[a])
            
            # percentage optimal action
            if a==optimal_action:
                optimal_action_history[i] = N[a]/(i+1)

        return Q_est, a_history, optimal_action_history, r_history, regret_history

##################################################################################################################


##################################################################################################################
#                                                                                                                #
#                                               Softmax Agent                                                    #
#                                                                                                                #
##################################################################################################################
def softmaxExploration(env, maxEpisodes, decay_till, max_tau=100, min_tau = 0.005, decay_type='lin'):
    """
    Follows function definition from lectures

    Input: env - "openAI gym environment", maxEpisodes - "maximum no of episodes for which algorithm runs", decay_till - "episode till which to decay epsilon"
           max_tau - "starting value of tau", min_tau - "final value of tau", decay_type - "linear or exponential decay"
           
    Output: Q_est - "history of Q value estimates", a_history - "history of actions taken", optimal_action_history - "history of percentage optimal action"
            r_history - "reward history", regret_history - "cumulative history of regret"
    
    Variable Definitions:
        Q: contains Q value estimates
        N: Contains count of taking an action for all actions
        Q_est: tracks history of Q 
        
        tau: temperature parameter of softmax

        a_history: tracks history of taken actions
        r_history: tracks history of rewards received
        regret_history: tracks cumulative regret
    
        optimal_action: stores optimal action
        optimal_action_history: tracks percentage optimal action history
    
    """
        Q = np.zeros(env.action_space.n)
        N = np.zeros(env.action_space.n)
        Q_est = np.zeros((maxEpisodes, env.action_space.n))
        a_history = np.zeros(maxEpisodes)
        r_history = np.zeros(maxEpisodes)
        tau_history = np.zeros(maxEpisodes)
        regret_history = np.zeros(maxEpisodes)

        optimal_action = np.argmax(env.q_value)
        optimal_action_history = np.zeros(maxEpisodes)

        # temperature setup
        exp_decay_rate = np.power(min_tau/max_tau,(1/(decay_till))) # exponential decay rate
        lin_decay_rate = (max_tau - min_tau)/decay_till # linear decay rate
        tau = max_tau

        for i in range(maxEpisodes):
            # follows algorithm from lectures
            env.reset()
            probs = softmax(Q/tau)
            a = np.random.choice(a = env.action_space.n, size = 1, p = probs)[0] # dereference from a list
            _, R, _, _ = env.step(a)
            N[a] = N[a] + 1
            Q[a] = Q[a] + (R-Q[a])/N[a]
            Q_est[i] = Q
            a_history[i] = a
            r_history[i] = R

            # decay temperature tau
            if tau > min_tau:
                if decay_type == 'exp':
                    tau = tau * exp_decay_rate
                elif decay_type == 'lin':
                    tau = max(tau - lin_decay_rate, min_tau)
                tau_history[i] = tau
            
            # regret calculation
            if i==0:
                regret_history[i] = env.q_value[optimal_action] - env.q_value[a]
            else:
                regret_history[i] = regret_history[i-1] + (env.q_value[optimal_action] - env.q_value[a])
            
            # perecentage optimal action
            if a==optimal_action:
                optimal_action_history[i] = N[a]/(i+1)

        return Q_est, a_history, optimal_action_history, r_history, regret_history

##################################################################################################################


##################################################################################################################
#                                                                                                                #
#                                        Upper Confidence Bound Agent                                            #
#                                                                                                                #
##################################################################################################################
def UCB(env, maxEpisodes, c):
    """
    Follows function definition from lectures

    Input: env - "openAI gym environment", maxEpisodes - "maximum no of episodes for which algorithm runs", c - "hyper-parameter for U function"
           
    Output: Q_est - "history of Q value estimates", a_history - "history of actions taken", optimal_action_history - "history of percentage optimal action"
            r_history - "reward history", regret_history - "cumulative history of regret"
    
    Variable Definitions:
        Q: contains Q value estimates
        N: Contains count of taking an action for all actions
        Q_est: tracks history of Q 

        a_history: tracks history of taken actions
        r_history: tracks history of rewards received
        regret_history: tracks cumulative regret
    
        optimal_action: stores optimal action
        optimal_action_history: tracks percentage optimal action history
    
    """
        Q = np.zeros(env.action_space.n)
        N = np.zeros(env.action_space.n)
        Q_est = np.zeros((maxEpisodes, env.action_space.n))
        a_history = np.zeros(maxEpisodes)
        r_history = np.zeros(maxEpisodes)
        regret_history = np.zeros(maxEpisodes)

        optimal_action = np.argmax(env.q_value)
        optimal_action_history = np.zeros(maxEpisodes)

        for i in range(maxEpisodes):
            # follows definition from lectures
            env.reset()
            if i < env.action_space.n:
                a = i
            else:
                U = c * np.sqrt(np.log(i)/N)
                a = np.argmax(Q+U)
            _, R, _, _ = env.step(a)
            N[a] = N[a] + 1
            Q[a] = Q[a] + (R-Q[a])/N[a]
            Q_est[i] = Q
            a_history[i] = a
            r_history[i] = R
            
            # regret calculations
            if i==0:
                regret_history[i] = env.q_value[optimal_action] - env.q_value[a]
            else:
                regret_history[i] = regret_history[i-1] + (env.q_value[optimal_action] - env.q_value[a])

            # percentage optimal calculation
            if a==optimal_action:
                optimal_action_history[i] = N[a]/(i+1)

        return Q_est, a_history, optimal_action_history, r_history, regret_history