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