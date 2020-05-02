import numpy as np
from multi_purpose_functions import *

def f(infected, size, percentage_time):
    """Returns the infection pressure exerted by a hyperedge on a particular individual. We do not work with beta here
    as we are just going to multiply the final infection pressures by it thus avoiding pointless calculations"""
    return infected * percentage_time / (size - 1)


def assign_hyperedges(N, size, first_hyperedge=1):
    """Assigns to each node a hyperedge (each hyperedge is denoted by an unique integer).
    "size" represents the number of nodes in this hyperedge type
     The function returns a list having the associated hyperedge for each node
    [1,2,1,1,2] means that nodes 0, 2 and 3 are part of hyperedge 1 and nodes 1 and 4 are part of hyperedge 2"""
    assignment = []
    i = -1 # in case N < size, we still want to assign a hyperedge to all nodes
    for i in range(N // size):
        assignment += [first_hyperedge + i] * size # add the hyperedge for a number of "size" times to the list as that hyperedge will contain that many nodes
    assignment += [first_hyperedge + i + 1] * (N % size) # in case N is not a multiple of size, we'll have a hyperedge of smaller size
    np.random.shuffle(assignment) # shuffle the list to obtain the hyperedge assignment
    return assignment


def hyper_infection(N, beta, gamma, hyperedge_sizes=(0,3,30), times_in_hyperedges=(0.2, 0.3, 0.5), time_limit=1000, infected_set = None):
    """ "hyperedge_sizes" represents the sizes of the hyperdge types. 0 refers to the entire node set (should thus be read as N)
    "times_in_hyperedges" will correspond to the time spent in each hyperedge type"""
    if infected_set is None:
        infected_set = {0} # the set of nodes that are initially infected
    if hyperedge_sizes[0]==0: # a silly hack to encode the random mixing hyperedge with 0 instead of N. I'm not proud
        hyperedge_sizes = tuple([N] + list(hyperedge_sizes)[1:])
    hyperedge_assignments = [[] for i in range(N)] # a list that will hold the incident hyperedges for each node
    susceptible_dictionary = {i:0 for i in range(N) if not i in infected_set} # the dictionary of susceptible nodes with the associated rates of infection (we initiate them at 0)

    current_hyperedge = 1 # will keep track of the number of the hyperedge when randomly assigning them to nodes
    for size in hyperedge_sizes: # we assign the set of incident hyperedges for each node
            assignment = assign_hyperedges(N, size, first_hyperedge=current_hyperedge)
            for i in range(N):
                hyperedge_assignments[i].append(assignment[i])
            current_hyperedge += N // size + (N % size != 0) # increase the current number by the number of hyperedges we added
    print(hyperedge_assignments)

    hyperedge_infections = {i:0 for i in range(1,current_hyperedge)} # keep track of the number of infected in each hyperedge
    for i in infected_set:
        for j in hyperedge_assignments[i]:
            hyperedge_infections[j] += 1  # we compute the number of infected in each hyperedge

    t = 0 # the start time of the process. it will change whenever a new event occurs
    SIR_data = [(t, len(susceptible_dictionary), len(infected_set), N - len(susceptible_dictionary) - len(infected_set))]
    # SIR_data will hold the evolution of the process by the recording the number of individuals in each class every time a new event occurs
    while len(infected_set) and t < time_limit: # continue the iteration whilst there are still infected nodes
        # There are 2 main events that can occur: a new infection or a recovery
        # Thus we need to know the rate of each after every new event occurs
        for susceptible_individual in susceptible_dictionary.keys():
            infectiousness = 0 # a local parameter for the total infection rate for each node
            for i in range(len(hyperedge_sizes)):
                corresponding_hyperedge = hyperedge_assignments[susceptible_individual][i]
                infected_in_corresponding_hyperedge = hyperedge_infections[corresponding_hyperedge]
                infectiousness += f(infected_in_corresponding_hyperedge, hyperedge_sizes[i], times_in_hyperedges[i])
            susceptible_dictionary[susceptible_individual] = infectiousness

        total_infection_rate = sum(susceptible_dictionary.values()) # the total rate of infection for the susceptibles without yet multiplying by beta
        total_recovery_rate = gamma * len(infected_set) # the total rate of recovery
        competing_rate = total_infection_rate * beta + total_recovery_rate # the rate at which a new event occurs
        infection_event = total_infection_rate * beta / competing_rate # chance that the next event is an infection
        #print(infection_event)
        #print((beta * len(infected_set) * len(susceptible_dictionary) * times_in_hyperedges[0]) / competing_rate / N)
        # Now we start our event based iteration
        dt = -np.log(np.random.random_sample()) / competing_rate # time until the next event occurs
        t += dt # increment the time

        if np.random.random_sample() < infection_event:  # then somebody got infected
            a = [] # this will hold all the individuals that could be infected
            p = [] # this will hold the respective probabilities of the individuals that could be infected
            for pair in susceptible_dictionary.items():
                a.append(pair[0])
                p.append(pair[1] / total_infection_rate)
            infected_individual = np.random.choice(a=a,p=p) # determine the infected individual from the distribution
            del susceptible_dictionary[infected_individual] # remove the infected individual from the susceptible class
            infected_set.add(infected_individual) # add the newly infected individual to the infected class
            #print(infected_set)
            for hyperedge in hyperedge_assignments[infected_individual]:
                hyperedge_infections[hyperedge] += 1 # increase the number of infected in all hyperedges containing the newly infected node
        else: # then somebody recovered
            # Note that the rate of recovery coincides for all infected individuals as a result of competing exponentials
            recovered_individual = np.random.choice(list(infected_set))
            infected_set.remove(recovered_individual) # Remove the newly recovered individual from the infected set
            for hyperedge in hyperedge_assignments[recovered_individual]:
                hyperedge_infections[hyperedge] -= 1 # decrease the number of infected in all hyperdges containing the recovered node
        SIR_data.append((t, len(susceptible_dictionary), len(infected_set), N - len(susceptible_dictionary) - len(infected_set)))
    return(SIR_data)


def pickle_hyper_SIR_data(file_name, runs, N, beta, gamma, hyperedge_sizes=(0,3,30),
                          times_in_hyperedges=(0.2, 0.3, 0.5), time_limit=1000, infected_set = None, directory='hyper_SIR'):
    """Creates a pickle file with multiple realisations of a hyper SIR model."""
    initial_data = {'N':N, 'infected_set':infected_set, 'times_in_hyperedges':times_in_hyperedges,
                    'hyperedge_sizes':hyperedge_sizes, 'time_limit':time_limit, 'runs':runs, 'beta':beta, 'gamma':gamma}
    all_runs = []
    for i in range(runs):
        print(i)
        all_runs.append(hyper_infection(N=N, beta=beta, gamma=gamma, time_limit=time_limit,
                                        hyperedge_sizes=hyperedge_sizes, times_in_hyperedges=times_in_hyperedges,
                                        infected_set=infected_set))
    with open(directory + '\\' + file_name + '.txt', "wb") as file:
        pickle.dump(initial_data, file) #pickles the initial parameters of the runs
        pickle.dump(all_runs, file) #stores a list of all the runs


def MO_calculator_hyper_pickle(file_name, directory='hyper_SIR'):
    """computes the probability of a major outbreak using the realisations of a hypergraph SIR model
    that have been saved in a pickle file"""
    initial_data, all_runs = unpickle_data(file_name, directory)
    major_outbreaks = 0
    for run in all_runs:
        major_outbreaks += (run[-1][1] < initial_data['N'] * 0.95)
        # be careful how you actually save the data especially if you are going to change the format
        #you might change the major outbreak condition in the future
    return major_outbreaks / initial_data['runs']


def MO_probability_calculator(beta, gamma, times_in_hyperedges):
    """Computes the roots of the quartic produced by a hypergraph containing only the random mixing hpyeredge
    and couples"""
    alpha = beta * (1 - times_in_hyperedges[1])
    coef = []
    coef.append(alpha ** 3)
    coef.append(-2 * alpha ** 2 * (alpha + gamma) - alpha ** 2 * (beta + gamma))
    coef.append(alpha * (alpha + gamma) ** 2 + gamma * alpha ** 2 + 2 * alpha * (alpha + gamma) * (beta + gamma))
    coef.append(-2 * alpha * (alpha + gamma) * gamma - (beta + gamma) * (alpha + gamma) ** 2)
    coef.append(gamma ** 2 * beta * times_in_hyperedges[1] + gamma * (alpha + gamma) ** 2)
    return np.roots(coef)


def draw_MO_graph(beta, gamma):
    """Draw the major outbreak probability as a function of random mixing hyperege weight in the married model"""
    percentage_array = []
    MO_array = []
    for i in range(101):
        times_in_hyperedges = (i/100, (100 - i)/100)
        solutions = MO_probability_calculator(beta=beta, gamma=gamma, times_in_hyperedges=times_in_hyperedges)
        solution = [1 - sol.real for sol in solutions if abs(sol.imag) < 0.001 and 0<sol.real<0.99]
        if solution == []:
            solution = [0]
        percentage_array.append(i/100)
        MO_array.append(solution)
    plt.plot(percentage_array, MO_array, color='red')
    plt.xlabel('Weight of random mixing hyperedge')
    plt.ylabel('Probability of a major outbreak')
    plt.show()


def draw_infection_from_hyper_pickle(file_name, directory='hyper_SIR', run=0, save_figure=False, figure_format='eps',
                                    figure_name=' '):
    """Draws the evolution of the infection over time for a single run in a pickle file"""
    initial_data, all_runs = unpickle_data(file_name, directory)
    times, susceptible, infected, recovered = [np.array([element[i] for element in all_runs[run]]) for i in range(4)]
    draw_SIR(times, susceptible, infected, recovered)
    if save_figure:
        path = 'figures\\' + figure_name + '.' + figure_format
        plt.savefig(path, format=figure_format)
    plt.show()


def number_of_total_infected_from_hyper_pickle(file_name, directory='hyper_SIR'):
    """Computes the total number of infected in each run for a pickled set of runs"""
    initial_data, all_runs = unpickle_data(file_name, directory)
    total_infected = []
    for run in all_runs:
        total_infected.append(initial_data['N'] - run[-1][1])
    return total_infected

def average_and_variane_of_total_infected(total_infected):
    """Computes the average and stdev for a list of numbers representing the total number of infected individuals"""
    total_infected_in_MO = [infected for infected in total_infected if infected > 50]
    total_infected = np.array(total_infected)
    print('The average number of infected individuals is ' + str(np.mean(total_infected)))
    print('The average number of infected for major outbreak is ' + str(np.mean(total_infected_in_MO)))
    print('The standard deviation for the number of infected individuals is ' + str(np.std(total_infected)))
    print('The standard deviation for the number of infected individuals in a MO is ' + str(np.std(total_infected_in_MO)))

    
"""Example uses:"""
#pickle_hyper_SIR_data('test_10runs', runs=10, N=1000, beta=0.6, gamma=0.2, hyperedge_sizes=(0, 3, 30), times_in_hyperedges=(0.2, 0.3, 0.5))
#pickle_hyper_SIR_data('test_10runs_2', runs=10, N=1000, beta=0.6, gamma=0.2, hyperedge_sizes=(0, 3, 30), times_in_hyperedges=(1, 0, 0))
#print(MO_calculator_hyper_pickle('test_1000_no_office'))
#draw_infection_from_hyper_pickle('test_10runs')
#unpickle_data('10000_full_runs', directory='hyper_SIR',)
#print(MO_calculator_hyper_pickle('10000_full_runs'))
#average_and_variane_of_total_infected(number_of_total_infected_from_hyper_pickle('10000_full_runs'))
