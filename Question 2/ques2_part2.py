import numpy as np
import matplotlib.pyplot as plt


def decayAlpha(initialValue, finalValue, maxSteps, decayType):
    exp_decay_rate = np.power(finalValue/initialValue, (1/maxSteps))
    lin_decay_rate = (initialValue - finalValue)/maxSteps
    
    step_sizes = []
    alpha = initialValue
    
    for i in range(maxSteps):
        step_sizes.append(alpha)
        if decayType == 'exp':
            alpha = alpha * exp_decay_rate
        elif decayType == 'lin':
            alpha = max(alpha - lin_decay_rate, finalValue)
    
    return step_sizes

if __name__ == "__main__":

    # linear decay
    lin_alpha = decayAlpha(1,0.01,500,'lin')

    # exp decay
    exp_alpha = decayAlpha(1,0.01,500,'exp')
    
    
    plt.plot(lin_alpha, 'r', label='linear')
    plt.plot(exp_alpha, 'g', label='exponential')
    plt.legend()
    plt.show()