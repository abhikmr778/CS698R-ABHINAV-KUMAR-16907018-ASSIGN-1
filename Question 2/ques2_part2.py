import numpy as np
import matplotlib.pyplot as plt


def decayAlpha(initialValue, finalValue, maxSteps, decayType):
    """
    Input: initialValue - starting value of step size alpha
           finalvalue - final value of step size alpha
           maxSteps - steps till which to decay alpha
           decayType - lin or exp decay

    Output: step_sizes: list of decayed alphas over maxSteps
    """
    exp_decay_rate = np.power(finalValue/initialValue, (1/maxSteps)) # exponential decay rate
    lin_decay_rate = (initialValue - finalValue)/maxSteps # linear decay rate
    
    step_sizes = []
    
    # initialize
    alpha = initialValue
    
    # decay and store alpha
    for i in range(maxSteps):
        step_sizes.append(alpha)
        if decayType == 'exp':
            alpha = alpha * exp_decay_rate
        elif decayType == 'lin':
            alpha = max(alpha - lin_decay_rate, finalValue)
    
    return step_sizes

if __name__ == "__main__":

    # linear decay
    lin_alpha = decayAlpha(initialValue=1,finalValue=0.01,maxSteps=500,decayType='lin')

    # exp decay
    exp_alpha = decayAlpha(initialValue=1,finalValue=0.01,maxSteps=500,decayType='exp')
    
    plt.figure(figsize=(12,8))
    plt.rcParams.update({'font.size': 14})
    plt.plot(lin_alpha, 'r', label='linear', linewidth=2)
    plt.plot(exp_alpha, 'g', label='exponential', linewidth=2)
    plt.title('Linear and Exponential Decay of Alpha')
    plt.xlabel('Time steps')
    plt.ylabel('Alpha')
    plt.legend()
    plt.savefig('q2p2.svg')
    plt.savefig('q2p2.jpg', dpi=300)
    plt.show()