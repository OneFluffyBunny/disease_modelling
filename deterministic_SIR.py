import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def SIR_DES(N, beta, gamma, I = 1):
    """Plot the disease outbreak using an SIR deterministic model"""
    def f(y, t, params):
        S, I = y
        beta, gamma = params
        equations = [-beta * S * I, beta * S * I - gamma * I]
        return equations
    params = [beta, gamma]
    y_0 = [1 - I / N, I / N]
    t_stop = 100
    t_increment = 0.1
    t = np.arange(0, t_stop, t_increment)
    solution = odeint(f, y_0, t, args=(params,))
    removed = np.array([1 - sol[0] - sol[1] for sol in solution])
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Population', fontsize=15)
    plt.plot(t, N * solution[:, 0], color='green', label='susceptible')
    plt.plot(t, N * solution[:, 1], color='red', label='infected')
    plt.plot(t, N * removed, color='black', label='removed')
    plt.legend()
    plt.show()
SIR_DES(N=1000, beta=0.4, gamma=0.2)
