import numpy as np
from scipy.optimize import fsolve
from multi_purpose_functions import *

def two_populations_SIR(N_1, N_2, I_1, I_2, beta_11, beta_12, beta_21, beta_22, gamma_1, gamma_2, R_1=0, R_2=0, time_limit=10000):
    """Creates a realisation of a stochastic SIR model on a 2 class heterogeneous population."""
    beta_11 = beta_11 / N_1
    beta_12 = beta_12 / N_1
    beta_21 = beta_21 / N_2
    beta_22 = beta_22 / N_2 #these are just normalising constants for the infection parameters to save computational time
    t = 0
    S_1 = N_1 - I_1 - R_1
    S_2 = N_2 - I_2 - R_2
    SIR_data = [(t, S_1, I_1, R_1, S_2, I_2, R_2)]
    while I_1 + I_2 and t < time_limit:
        infection_11 = beta_11 * S_1 * I_1
        infection_12 = beta_12 * S_1 * I_2
        infection_1 = infection_11 + infection_12
        infection_21 = beta_21 * S_2 * I_1
        infection_22 = beta_22 * S_2 * I_2
        infection_2 = infection_21 + infection_22
        recovery_1 = I_1 * gamma_1
        recovery_2 = I_2 * gamma_2
        rates = np.array([infection_1, recovery_1, infection_2, recovery_2])
        competing_rate = sum(rates)
        rates = rates / competing_rate

        dt = -np.log(np.random.random_sample()) / competing_rate
        t += dt

        r = np.random.random_sample()
        if r < rates[0]:
            S_1 -= 1
            I_1 += 1
        elif r < rates[0] + rates[1]:
            I_1 -= 1
            R_1 += 1
        elif r < rates[0] + rates[1] + rates[2]:
            S_2 -= 1
            I_2 += 1
        else:
            I_2 -= 1
            R_2 += 1
        SIR_data.append((t, S_1, I_1, R_1, S_2, I_2, R_2))
    return SIR_data


def pickle_heterogeneity_data(file_name, runs, N_1, N_2, I_1, I_2, beta_11, beta_12, beta_21, beta_22,
                            gamma_1, gamma_2, R_1=0, R_2=0, time_limit=10000, directory='heterogeneous_SIR'):
    """Creates a pickle file with multiple realisations of a SIR model on a 2 class heterogeneous population."""
    initial_data = {'N_1':N_1, 'N_2':N_2, 'I_1':I_1, 'I_2':I_2, 'beta_11':beta_11, 'beta_12':beta_12, 'beta_21':beta_21,
                    'beta_22':beta_22, 'gamma_1':gamma_1, 'gamma_2':gamma_2, 'R_1':R_1, 'R_2':R_2,
                    'time_limit':time_limit, 'runs':runs}
    all_runs = []
    for i in range(runs):
        all_runs.append(two_populations_SIR(N_1=N_1, N_2=N_2, I_1=I_1, I_2=I_2, beta_11=beta_11,
                                              beta_12=beta_12, beta_21=beta_21, beta_22=beta_22, gamma_1=gamma_1,
                                              gamma_2=gamma_2, R_1=R_1, R_2=R_2, time_limit=time_limit))
    with open(directory + '\\' + file_name +'.txt', "wb") as file:
        pickle.dump(initial_data, file) #pickles the initial parameters of the runs
        pickle.dump(all_runs, file) #stores a list of all the runs


def empirical_major_outbreak_probability_from_pickle(file_name, directory='heterogeneous_SIR'):
    """computes the probability of a major outbreak using the realisations of a SIR model on a 2 class
    heterogeneous population that has been saved in a pickle file"""
    initial_data, all_runs = unpickle_data(file_name, directory)
    major_outbreaks = 0
    for run in all_runs:
        major_outbreaks += (run[-1][1] + run[-1][4] < (initial_data['N_1'] + initial_data['N_2']) * 0.95)
        # be careful how you actually save the data especially if you are going to change the format
        #you might change the major outbreak condition in the future
    return major_outbreaks / initial_data['runs']


def efficient_solver(a,b,c,d):
    """Computes the outbreak probabilities in a very efficient manner"""
    if (1-2*a-b>0 and 1-2*c-d>0 and (1-2*a-b)*(1-2*c-d)>b*d):
        return [1,1]
    x_min = ((1 - np.sqrt(1 - 4 * a * (1 - a - b))) / (2 * a))
    y_min = ((1 - np.sqrt(1 - 4 * c * (1 - c - d))) / (2 * c))
    def f(variables):
        x, y = variables
        return [x * (-a / b) + 1 / b + 1 / x * (a + b - 1) / b - y, y * (-c / d) + 1 / d + 1 / y * (c + d - 1) / d - x]
    return fsolve(f, [x_min, y_min])


def homo_from_hetero(beta_11, beta_12, beta_21, beta_22, gamma_1, gamma_2):
    """Returns the naive infection parameters one would get if assuming a homogeneous model"""
    gamma = 2 * gamma_1 * gamma_2 / (gamma_1 + gamma_2)
    beta = ((beta_11 + beta_21) * gamma_2 + (beta_12 + beta_22) * gamma_1) / (gamma_1 + gamma_2)
    return (beta, gamma)


def homo_vs_hetero_probability(beta_11, beta_12, beta_21, beta_22, gamma_1, gamma_2):
    """Computes the major outbreak probability when using the homogeneous/heterogeneous stochastic SIR model"""
    beta, gamma = homo_from_hetero(beta_11, beta_12, beta_21, beta_22, gamma_1, gamma_2)
    homo_probability = 1 - gamma/beta

    rate_1 = beta_11 + beta_21 + gamma_1
    rate_2 = beta_12 + beta_22 + gamma_2
    hetero_solutions = efficient_solver(a = beta_11 / rate_1, b = beta_21 / rate_1, c = beta_22 / rate_2, d = beta_12 / rate_2)
    return homo_probability, 1 - hetero_solutions[0]


def vary_beta_11_beta_21(beta_11, beta_12, beta_21, beta_22, gamma_1, gamma_2):
    """Varies the values of beta_11 and beta_21 while maintaining the same naive values for the infection parameters"""
    accuracy = 100
    constant = beta_11 + beta_21
    beta_11_values = [i / accuracy for i in range(1, int(np.floor(constant * accuracy)))]
    homo_probability = homo_vs_hetero_probability(beta_11, beta_12, beta_21, beta_22, gamma_1, gamma_2)[0]

    hetero_list_one = []
    for value in beta_11_values:
        print(homo_vs_hetero_probability(value, beta_12, constant-value, beta_22, gamma_1, gamma_2)[1])
        hetero_list_one.append(homo_vs_hetero_probability(value, beta_12, constant-value, beta_22, gamma_1, gamma_2)[1])
    plt.plot(beta_11_values, hetero_list_one, color='red', label='actual probability')
    plt.plot(beta_11_values, [homo_probability for i in range(1, int(np.floor(constant * accuracy)))], linestyle = '--', label='naive probability')
    plt.xlabel('Assortative infection parameter', fontsize=15)
    plt.ylabel('Major outbreak probability', fontsize=15)
    plt.legend()
    plt.show()


def compute_dominant_eigenvalue(beta_11, beta_12, beta_21, beta_22, gamma_1, gamma_2):
    """Compute the dominant eigenvalue of the next generation matrix"""
    A = gamma_1 * gamma_2
    B = beta_11 * gamma_2 + beta_22 * gamma_1
    C = beta_11 * beta_22 - beta_12 * beta_21
    dominant_eigenvalue = (B + np.sqrt(B * B - 4 * A * C)) / (2 * A)
    return dominant_eigenvalue


def dominant_eigenvalue_vary_beta_11_beta_21(beta_11, beta_12, beta_21, beta_22, gamma_1, gamma_2):
    """Varies beta_11 and beta_21 while maintaining the dominanat eigenvalue of the next generation matrix unchanged"""
    dominant_eigenvalue = compute_dominant_eigenvalue(beta_11, beta_12, beta_21, beta_22, gamma_1, gamma_2)

    varied_beta_11_21 = []
    copy_beta_11 = 0.01
    hetero_list = []
    while True:
        copy_beta_21 = (gamma_1 * gamma_2 * dominant_eigenvalue ** 2 - dominant_eigenvalue * (copy_beta_11 * gamma_2 + beta_22 * gamma_1) + copy_beta_11 * beta_22) / beta_12
        if copy_beta_21 < 0:
            break
        varied_beta_11_21.append((copy_beta_11, copy_beta_21))
        copy_beta_11 += 0.01
    for values in varied_beta_11_21:
        homo, hetero = homo_vs_hetero_probability(values[0], beta_12, values[1], beta_22, gamma_1, gamma_2)
        hetero_list.append(hetero)

    plt.plot([element[0] for element in varied_beta_11_21], hetero_list, color='red', label='actual probability')
    plt.plot([element[0] for element in varied_beta_11_21], [1 - 1/dominant_eigenvalue] * len(varied_beta_11_21), linestyle='--',
             label='naive probability')
    plt.xlabel('Assortative infection parameter', fontsize=15)
    plt.ylabel('Major outbreak probability', fontsize=15)
    plt.legend()
    plt.show()


def predicted_major_outbreak_calculator_from_pickle(file_name, directory='heterogeneous_SIR'):
    """Computes the probability of a major outbreak starting from 1 individual from the pickle file"""
    initial_data = unpickle_data(file_name, directory, only_initial=True)
    rate_1 = (initial_data['beta_11'] + initial_data['beta_21'] + initial_data['gamma_1'])
    rate_2 = (initial_data['beta_22'] + initial_data['beta_12'] + initial_data['gamma_2'])
    a = initial_data['beta_11'] / rate_1
    b = initial_data['beta_21'] / rate_1
    c = initial_data['beta_22'] / rate_2
    d = initial_data['beta_12'] / rate_2
    return efficient_solver(a,b,c,d)


def compare_predicted_and_empirical(file_name, directory='heterogeneous_SIR'):
    """Returns the empirical and predicted major outbreak probabilities for I_1=1"""
    empirical = empirical_major_outbreak_probability_from_pickle(file_name, directory)
    solutions = predicted_major_outbreak_calculator_from_pickle(file_name, directory)
    I_1, I_2 = unpickle_data(file_name, directory, only_initial=True)['I_1'], unpickle_data(file_name, directory, only_initial=True)['I_2']
    predicted = 1 - solutions[0] ** I_1 * solutions[1] ** I_2
    return empirical, predicted


def draw_infection_from_pickle_hetero(file_name, directory='heterogeneous_SIR', run=0, separate_plots=False,
                               save_figure=False, figure_format='eps', figure_name=' ',
                                      figure_title=None, figure_title_2=None):
    """Draws the evolution of the infection over time for a single run in a pickle file"""
    initial_data, all_runs = unpickle_data(file_name, directory)
    times, S_1, I_1, R_1, S_2, I_2, R_2 = [np.array([element[i] for element in all_runs[run]]) for i in range(7)]
    if separate_plots:
        draw_SIR(times, S_1, I_1, R_1, 1,figure_title=figure_title)
        draw_SIR(times, S_2, I_2, R_2, 2, figure_title=figure_title_2)
    else:
        draw_SIR(times, S_1+S_2, I_1+I_2, R_1+R_2, figure_title=figure_title)
    if save_figure:
        path = 'figures\\' + figure_name + '.' + figure_format
        plt.savefig(path, format=figure_format)
    plt.show()


def final_distibution_from_hetero_pickle(file_name, bins=100, directory='heterogeneous_SIR', save_figure=False,
                                       figure_format='eps', figure_name=' '):
    """Plots the distribution of the total number of infected by the end of the outbreak. Divide into 100 compartments
    We mainly used this function to analyse the distribution of the final infected distribution"""
    initial_data, all_runs = unpickle_data(file_name, directory)
    N = initial_data['N_1'] + initial_data['N_2']
    runs = initial_data['runs']
    step = N // bins
    distribution = [0] * bins
    for run in range(runs):
        distribution[(all_runs[run][-1][-1] + all_runs[run][-1][-4]) // step] += 1
    draw_distribution(distribution, bins=bins)
    if save_figure:
        path = 'figures\\' + figure_name + '.' + figure_format
        plt.savefig(path, format=figure_format)
    plt.show()


def vaccination_optimisation(N_1, N_2, beta_11, beta_12, beta_21, beta_22, gamma_1, gamma_2):
    """Draws the function v(d1) over the prescribed interval"""
    sign_term = beta_11 * gamma_2 + beta_12 * beta_21 - beta_11 * beta_22
    if sign_term > 0:
        lower_bound = max(0, gamma_1 * (gamma_2 - beta_22) / (beta_11 * gamma_2 + beta_12 * beta_21 - beta_11 * beta_22))
        upper_bound = min(1, gamma_1 / beta_11)
    else:
        lower_bound = 0
        upper_bound = min(1, gamma_1 / beta_11, gamma_1 * (gamma_2 - beta_22) / (beta_11 * gamma_2 + beta_12 * beta_21 - beta_11 * beta_22))
    print(lower_bound, upper_bound)
    d_range = [i/10000 for i in range(int(lower_bound * 10000), int(upper_bound * 10000))]
    v_range = [int(N_1 * d_1 + N_2 * (gamma_1 * gamma_2 - d_1 * gamma_2 * beta_11) / (beta_22 * gamma_1 + d_1 * (beta_12 * beta_21 - beta_11 * beta_22))) for d_1 in d_range]
    print(v_range)
    plt.plot(d_range, v_range, color='green')
    plt.xlabel('d1', fontsize=20)
    plt.ylabel('v(d1)', fontsize=20)
    plt.show()
    
"""Example uses for the functions above:"""
#vaccination_optimisation(N_1=1000, N_2=1000, beta_11=0.7, beta_12=0.1, beta_21=0.2, beta_22=0.3, gamma_1=0.25, gamma_2=0.2)
#pickle_heterogeneity_data(file_name='1000_runs', runs=1000, N_1=400, N_2=1600, I_1=1, I_2=0, beta_11=0.7, beta_12=0.1, beta_21=0.2, beta_22=0.3, gamma_1=0.25, gamma_2=0.2)
#print(compare_predicted_and_empirical('children_adults'))
#draw_infection_from_pickle_hetero('depletion',separate_plots=True, run=2, save_figure=False, figure_name='just_children', figure_format='png', figure_title='Children', figure_title_2='Adults')
#print(efficient_solver(a,b,c,d))
#print(compare_predicted_and_empirical('naive_vs_hetero_2'))
#final_distibution_from_hetero_pickle('10000_runs_trimodal')
#final_distibution_from_hetero_pickle('1000_runs')
#unpickle_data('10000_runs', directory='heterogeneous_SIR', only_initial=True, print_initial=True)
#print(compare_predicted_and_empirical('10000_6'))
#vary_beta_11_beta_21(beta_11=0.6, beta_12=0.1, beta_21=0.2, beta_22=0.4, gamma_1=0.3, gamma_2=0.2)
#dominant_eigenvalue_vary_beta_11_beta_21(beta_11=0.6, beta_12=0.1, beta_21=0.2, beta_22=0.1, gamma_1=0.3, gamma_2=0.15)
