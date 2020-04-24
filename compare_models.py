from homogeneous_SIR import *
from two_populations_SIR import *
from deterministic_SIR import *

def compare_stochastic_DE(N, beta, gamma, I=10):
    #plt.figure(1)
    SIR_DES(N=N, beta=beta, gamma=gamma, I=I)
    #plt.figure(2)
    data = SIR_trial(N=N, beta=beta, gamma=gamma, I=I)
    times, susceptible, infected, recovered = [np.array([element[i] for element in data]) for i in range(4)]
    plt.plot(times, susceptible)
    plt.plot(times, infected)
    plt.show()


def compare_MO_simple_heterogeneous(N_1, N_2, I_1, I_2, beta_11, beta_12, beta_21, beta_22, gamma_1, gamma_2, R_1=0, R_2=0):
    N = N_1 + N_2
    I = I_1 + I_2
    gamma = N / (N_1 * gamma_1 ** (-1) + N_2 * gamma_2 ** (-1))
    beta = (N_1 * (beta_11 + beta_21) + N_2 * (beta_22 + beta_12)) / (N_1 + N_2)
    if gamma>beta:
        MO_simple = 0
    else:
        MO_simple = 1 - (gamma / beta) ** I
    print(beta)
    rate_1 = beta_11 + beta_21 + gamma_1
    rate_2 = beta_22 + beta_12 + gamma_2
    a = beta_11 / rate_1
    b = beta_21 / rate_1
    c = beta_22 / rate_2
    d = beta_12 / rate_2
    sol = efficient_solver(a,b,c,d)
    MO_heterogeneous = 1 - sol[0] ** I_1 * sol[1] ** I_2
    return MO_simple, MO_heterogeneous

#print(compare_MO_simple_heterogeneous(N_1=500, N_2=500, I_1=0, I_2=1, beta_11=0.19, beta_12=0.07, beta_21=0.035, beta_22=0.1, gamma_1=0.2, gamma_2=0.2))