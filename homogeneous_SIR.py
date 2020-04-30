import numpy as np
from multi_purpose_functions import *

def homogeneous_SIR(N, beta, gamma, time_limit=1000, I=1, R=0):
    """Create a list with the number of people in each category at each event for a basic SIR model"""
    beta = beta / N #we make the division here so we don't have to divide the total infection pressure by N at every iteration
    t = 0 #the start time
    S = N - I - R #the initial number of susceptible individuals
    SIR_data = [(t, S, I, R)] #SIR_data will hold the evolution of the process
    while I and t < time_limit: #while there are still infected individuals and we're in reasonable time limits
        infection_rate = beta * S * I #notice how beta already absorbed the constant 1/N
        recovery_rate = gamma * I
        competing_rate = infection_rate + recovery_rate #total competing rate for the next event exponential
        infection_event = infection_rate / competing_rate #the chance of an infection event
        dt = -np.log(np.random.random_sample()) / competing_rate #the time until the next event
        t += dt #increase the time by the increment

        if np.random.random_sample() < infection_event: #then the event was an infection
            S -= 1
            I += 1
        else: #then the event was a recovery
            I -= 1
            R += 1
        SIR_data.append((t, S, I, R))
    return SIR_data


def pickle_homogeneous_SIR_data(file_name, runs, N, beta, gamma, I=1, R=0, time_limit=10000, directory='simple_SIR'):
    """Creates a pickle file with multiple realisations of a simple SIR model."""
    initial_data = {'N':N, 'I':I, 'R':R, 'time_limit':time_limit, 'runs':runs, 'beta':beta, 'gamma':gamma}
    all_runs = []
    for i in range(runs):
        all_runs.append(homogeneous_SIR(N=N, beta=beta, gamma=gamma, time_limit=time_limit, I=I, R=R))
    with open(directory + '\\' + file_name + '.txt', "wb") as file:
        pickle.dump(initial_data, file) #pickles the initial parameters of the runs
        pickle.dump(all_runs, file) #stores a list of all the runs


def draw_infection_from_homogeneous_pickle(file_name, directory='simple_SIR', run=0, save_figure=False, figure_format='eps',
                                    figure_name=' '):
    """Draws the evolution of the infection over time for a single run in a pickle file"""
    initial_data, all_runs = unpickle_data(file_name, directory)
    times, susceptible, infected, recovered = [np.array([element[i] for element in all_runs[run]]) for i in range(4)]
    draw_SIR(times, susceptible, infected, recovered)
    if save_figure:
        path = 'figures\\' + figure_name + '.' + figure_format
        plt.savefig(path, format=figure_format)
    plt.show()


def final_distibution_from_homogeneous_pickle(file_name, bins=100, directory='simple_SIR', save_figure=False,
                                       figure_format='eps', figure_name=' '):
    """Plots the distribution of the total number of infected by the end of the outbreak. Divide into 100 compartments
    We mainly used this function to prove that the final distribution of the infected is bimodal"""
    initial_data, all_runs = unpickle_data(file_name, directory)
    N = initial_data['N']
    runs = initial_data['runs']
    step = N // bins
    distribution = [0] * bins
    for run in range(runs):
        distribution[all_runs[run][-1][-1] // step] += 1
    draw_distribution(distribution, bins=bins)
    if save_figure:
        path = 'figures\\' + figure_name + '.' + figure_format
        plt.savefig(path, format=figure_format)
    plt.show()


def empirical_major_outbreak_calculator(N_list, beta, gamma, runs=10000, I=1, R=0):
    """Compute the empirical major outbreak probabilities for populations in the N_list
    The rest of the function is identical with homogeneous_SIR except that we stop the process if 5%
    of the population became infected and thus a major outbreak has been achieved. Also we do not care
    about the timescale anymore."""
    major_outbreak_probabilities = []
    copy_I, copy_R = I, R
    for N in N_list:
        adjusted_beta = beta / N
        total_major_outbreaks = 0

        for run in range(runs):
            I, R = copy_I, copy_R
            S = N - I - R  # the initial number of susceptible individuals
            major_outbreak = 0
            while I and S > 0.95 * N:  # while there are still infected individuals and we're in reasonable time limits
                infection_event = adjusted_beta * S / (adjusted_beta * S + gamma)
                if np.random.random_sample() < infection_event:  # then the event was an infection
                    S -= 1
                    I += 1
                else:  # then the event was a recovery
                    I -= 1
                    R += 1
            if S <= 0.95 * N:
                total_major_outbreaks += 1
        major_outbreak_probabilities.append((N, total_major_outbreaks / runs))
        print(major_outbreak_probabilities)
    return major_outbreak_probabilities

"""Example uses of the above functions:"""
#pickle_homogeneous_SIR_data(file_name='10000_runs_3', runs=10000, N=1000, beta=0.3, gamma=0.2, I=1, R=0, time_limit=1000)
#final_distibution_from_homogeneous_pickle('10000_runs_3', save_figure=True, figure_name='affected_distribution_R3', figure_format='eps')
#draw_infection_from_homogeneous_pickle('10000_runs', run=4, save_figure=True, figure_format='eps', figure_name='homo_outbreak')
