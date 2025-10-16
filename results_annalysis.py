import numpy as np
import matplotlib.pyplot as plt

# This file is for reading results saved by the computing.py file and to make various visualization of it

# HOW TO IMPLEMET NEW PLOTS: Build a function of (states, energies) that ends in some sort of plt.show(), then run this function from the if __name__ == "__main__"

# Open the results of a simulation saved in a file
def open_results(path):
    # unpack the parameters used for the simulation
    instance_size, step, n_itterations, n_cond_init = path[18:].split('_')[:-1]

    # load and unpack the datas
    data = np.load(path)
    J, H = data['J'], data['H']
    states, energies = data['states'], data['energies']

    return states, energies

# energies evolution
def plot_energies_evolution(energies):
    n_cond_init = len(energies)
    n_iterration = len(energies[0])
    abcisses = np.arange(n_iterration)

    for i in range(n_cond_init):
        plt.plot(abcisses, energies[i])

    plt.xlabel("Iteration number")
    plt.ylabel("Energy level")
    plt.title("Energy level evolution for each of the simulations")

    plt.show()

def complete_plot(states, energies, sim_index):

    positions = states[sim_index, :, :, 0]
    energies = energies[sim_index]
    num_particles, n_iterration = positions.shape
    abcisses = np.arange(n_iterration)

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot


    for i in range(num_particles):
        plt.plot(abcisses, (positions[i, :]))

    plt.xlabel("$t$")
    plt.ylabel("Oscillators positions $x_i$")
    plt.title("Evolution of the positions of the oscillators")

    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
    plt.plot(abcisses, energies)
    plt.xlabel("$t$")
    plt.ylabel("Energy level")
    plt.title("Evolution of the system's energy")

    plt.tight_layout()  # Adjust the layout
    plt.show()

def plot_speeds_evolution(states):
    speeds = states[:, :, :, 2]
    n_cond_init, _,  n_iterration = speeds.shape 
    abcisses = np.arange(n_iterration)

    for i in range(n_cond_init):
        plt.plot(abcisses, speeds[i])

    plt.xlabel("Iteration number")
    plt.ylabel("Particle speed")
    plt.title("Energy level evolution for each of the simulations")

    plt.show()

def plot_min_energy_evolution(energies):
    n_iterration = len(energies[0])
    abcisses = np.arange(n_iterration)

    min_energies = energies.min(axis=0)

    plt.plot(abcisses, min_energies)

    plt.xlabel("Iteration number")
    plt.ylabel("Minum of energy reached at itteration $t$")
    plt.title("Evolution of the minimum of energy reached across all simulations.")

    plt.show()

# Histogram of the minimum energies reached by each simulation
def plot_energies_hist(energies):
    minimums = energies.min(axis=1)
    plt.hist(minimums)
    plt.xlabel("Minimum of energy reached by each simulation")
    plt.ylabel("Number of simulations")
    plt.title("Energies reached by the simulations")

    plt.show()

# Histogram of the normalized difference between the global minimum nad minimum energies reached by each simulation
def plot_diff_norm_energies_hist(energies):
    minimums = energies.min(axis=1)
    m = min(minimums)
    y_values = np.array(abs((m-minimums)/m))
    plt.hist(y_values)
    plt.xlabel("normalized difference of energy with the global minimu")
    plt.ylabel("Number of simulations")
    plt.title("the normalized difference between the global minimum and minimum energies reached by each simulation")

    plt.show()

# Returns the solution energy, the corresponding spin configuration and the index of the simulation that reached that energy
def extract_full_solution(states, energies):
    mins_indexes = np.argmin(energies, axis=1)
    solution_energy = np.min(energies)
    simulaiton_solution_index = 0

    for i in range(len(mins_indexes)):
        iteration_index = mins_indexes[i]
        if energies[i, iteration_index] == solution_energy:
            simulaiton_solution_index = i

    min_coord = simulaiton_solution_index, mins_indexes[simulaiton_solution_index]

    spin_configuration = np.where(states>0, 1, -1)
    spin_configuration = spin_configuration[min_coord[0], :, min_coord[1], 0]
    spin_configuration = spin_configuration.flatten()

    return solution_energy, spin_configuration, simulaiton_solution_index


if __name__=='__main__':
    plot_energies_hist(1000)