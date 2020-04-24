import pickle
import os
import matplotlib.pyplot as plt

def unpickle_data(file_name, directory, only_initial=False, print_initial=False):
    """unpickles a file containing the initial data and all the runs of a SIR model on a 2 class heterogeneous population."""
    file_path = directory + '\\' + file_name + '.txt'
    if only_initial:
        with open(file_path, 'rb') as file:
            initial_data = pickle.load(file)
            if print_initial:
                print(initial_data)
            return initial_data
    else:
        with open(file_path, 'rb') as file:
            return pickle.load(file), pickle.load(file)


def draw_SIR(times, susceptible, infected, recovered, figure=0, figure_title=None):
    """Draws a nice figure with the outbreak evolution"""
    if figure:
        plt.figure(figure)
    plt.plot(times, susceptible, color='green', label='susceptible')
    plt.plot(times, infected, color='red', label='infected')
    plt.plot(times, recovered, color='black', label='removed')
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Population', fontsize=15)
    if figure_title:
        plt.title(figure_title)
    plt.legend()


def draw_distribution(distribution, bins=100):
    plt.bar(range(1, bins+1), distribution, color='red')
    if bins==100:
        plt.xlabel('Affected population (%)', fontsize=15)
    plt.ylabel('Runs', fontsize=15)
    #plt.ylim(top=5200)


def create_directory(directory):
    try:
        os.mkdir(directory)
    except:
        pass

#create_directory('hyper_SIR')