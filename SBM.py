import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
from joblib import Parallel, delayed
import random

# This file creats the IsingModel class

class SBM:
    def __init__(self, J, H, step_size, num_iterations, num_simulations, stopping_criterion=0, save_states_history=True, save_energies_history=True, custom_pumping_rate=None):
        """
        Args:
            ------Numerical resultion parameters-------
            step_size (float): Step size for the Euler scheme.
            num_iterations (int): Number of iterations to run the simulation for at maximum.
            num_simulations (int): Number of simulations to run with this SBM.
            stopping_criterion (float): % of the particles that have to reach the stopping criterion for the simulation to stop.
            save_history (bool): Whether to save the history of the simulation or not.

            ------------Physical parameters--------------
            J (numpy.ndarray): Array of shape (num_particles, num_particles) representing the interactions between the particles.
            H (numpy.ndarray): Array of shape (num_particles) representing the external field applied on the particles.
            custom_pumping_rate (function): Two photo pumping rate.
            
        """
        self.num_particles = J.shape[0]
        self.num_iterations = num_iterations
        self.stopping_criterion = stopping_criterion
        self.J =J
        self.H=H
        self.save_states_history = save_states_history
        self.save_energies_history = save_energies_history
        self.pumping_rate_func = custom_pumping_rate if custom_pumping_rate is not None else self.default_pumping_rate
        self.step_size = step_size if callable(step_size) else (lambda self, t: step_size)
        self.num_simulations=num_simulations
        self.initialize_model()


    def default_pumping_rate(self, t, _):
        """
        Default pumping rate function in case no argument is given.
        """
        return 0
    
    def pumping_rate(self, t):
        """
        Caller for the pumping rate function. Weather it is the detault one or the one specified in the class builder arguments.
        """
        return self.pumping_rate_func(self, t)
    
    def initialize_model(self):
        """
        Initializes the model for the simulation.

        Args:
            None

        Returns:
            None

        Additional notes: It initializes following
            - The full states array (if asked to)
            - The model numerical parameters such as ksi and other
            - The initial conditions for the simulations
        """

        #-------- Arrays to store states, current state and energies -------
        if self.save_states_history:
            self.states = np.zeros(shape=(self.num_simulations, self.num_particles, self.num_iterations, 2))
        if self.save_energies_history:
            self.energies = np.zeros(shape=(self.num_simulations, self.num_iterations))

        self.current_state = np.zeros(shape=(self.num_simulations, self.num_particles, 2))
        self.energies = np.zeros(shape=(self.num_simulations, self.num_iterations))

        #-------- Model Parameters --------
        # self.ksi = 0.5/np.sqrt( np.sum(np.square(self.J)) / (self.num_particles-1) )
        self.ksi = 1/np.linalg.eigvals(self.J).max()

        #------ Initial conditions -------
        self.current_state[:, :, 0] = np.random.normal(0, 0.0001, size=(self.num_simulations, self.num_particles))


    def simplectic_update(self, positions, speeds, t):
        """
        Update of the speeds/positions based on the current state and the described system dynamics. Solves the Euler scheme.

        Args:
            positions (numpy.ndarray): Array of shape (n_simulations, num_particles) representing the positions of the particles for each simulation.
            speeds (numpy.ndarray): Array of shape (n_simulations, num_particles) representing the speeds of the particles for each simulation.

        Returns:
            positions (numpy.ndarray): Array of shape (n_simulations, num_particles) representing the updated positions of the particles for each simulation.
            speeds (numpy.ndarray): Array of shape (n_simulations, num_particles) representing the updated speeds of the particles for each simulation.
        """

        #-------- Gradient of the potential energy --------
        forces = -np.dot(self.J, positions.T).T * self.ksi
        
        #-------- For CIM amplitude dynamics --------
        forces += (-1 + self.pumping_rate(t) - np.square(positions)) * positions

        #-------- For b(alistic)SB simulation  dynamics --------
        # forces += (-1 + self.pumping_rate(t)) * positions

        #-------- For d(iscrete)SB simulatio  dynamics --------
        # forces = -np.dot(self.J, np.sign(positions).T).T * self.ksi + (-1 + self.pumping_rate(t)) * positions # for d(iscrete)SB simulation
        
        # Update speeds and positions
        speeds = speeds + self.step_size(self, t) * forces
        positions = positions + self.step_size(self, t) * speeds

        # Implement the walls at +1 and -1
        positions = np.clip(positions, -1, 1)
        speeds = np.where((positions == 1) | (positions == -1), 0, speeds)
        
        return positions, speeds
    
    def compute_energies(self, positions):
        """
        Computes the different energies of the different simulations based on the positions of the particles.

        Args:
            positions (numpy.ndarray): Array of shape (n_simulations, num_particles) representing the positions of the particles for each simulation.

        Returns:
            current_energies (numpy.adarray): Array of shape (n_simulations) representing the energies of the different simulations.

        Additional notes: This computes the energies of the binarized states.
        """

        signed_positions = np.where(positions > 0, 1, -1)
        # current_energies = np.sum(signed_positions @ self.J * signed_positions, axis=1) + self.H @ signed_positions.T
        current_energies = np.sum(positions @ self.J * positions, axis=1) + self.H @ positions.T
        
        return current_energies

    def sign(self, current_state):
        current_state = np.copy(current_state)

        positions = current_state[:, :, 0]

        return np.sign(positions)


    def TAC(self, current_state):
            current_state = np.copy(current_state)

            positions = current_state[:, :, 0]

            k = 0.5
            
            spins = np.zeros_like(positions)

            epsilons = k*np.linalg.norm(positions)/np.sqrt(800)        

            # STEP 1: Trap the traped nodes and put the swing nodes to zer
            spins = np.where(np.abs(positions) < epsilons, 0, np.sign(positions))

            # STEP 2: Randomly select the the oder in which we are going to set the spins associated to the swing nodes
            zero_indexes = np.where(spins==0)

            # STEP 3:
            for sim_index in range(self.num_simulations):
                    sim_spins = spins[sim_index]
                    zero_indexes = np.where(sim_spins==0)[0]
                    np.random.shuffle(zero_indexes)

                    for index in zero_indexes:
                            forces = -np.dot(self.J, sim_spins.T).T
                            sim_spins[index] = np.sign(forces[index])
                    
                    prev_count = zero_indexes.shape[0]
                    zero_indexes = np.where(sim_spins==0)[0]
                    if zero_indexes.shape[0] == prev_count:
                            # print("TAC failed to converge")
                            break
                    
                    spins[sim_index] = sim_spins
            
            return spins


    def early_sign_stopping(self, current_state):
        current_state = np.copy(current_state)

        mask1 = np.all(current_state == [-1, 0], axis=2)
        mask2 = np.all(current_state == [1, 0], axis=2)

        mask = mask1 | mask2

        bifurcated_percentage = np.sum(mask) / (self.num_particles*self.num_simulations)

        if bifurcated_percentage >= 1-self.stopping_criterion:
            return True
        
        return False
    
    def early_TAC_stopping(self, current_state):
        current_state = np.copy(current_state)

        mask1 = np.all(current_state == [-1, 0], axis=2)
        mask2 = np.all(current_state == [1, 0], axis=2)

        mask = mask1 | mask2

        bifurcated_percentage = np.sum(mask) / (self.num_particles*self.num_simulations)

        if bifurcated_percentage >= 1-self.stopping_criterion:
            return True
        
        return False

    def simulate(self):
        """
        Runs the simulation for the given parameters.
        
        Returns:
            states (numpy.ndarray): Array of shape (n_simulations, num_particles, n_iterations, 2) representing the states of the different simulations at every iteration time.
            energies (numpy.ndarray): Array of shape (n_simulations, n_iterations) representing the energies of the different simulations at every iteration time.
            current_state (numpy.ndarray): Array of shape (n_simulations, num_particles, 2) representing the states of the different simulations at the last iteration time.
            current_energies (numpy.ndarray): Array of shape (n_simulations) representing the energies of the different simulations at the last iteration time.
        """

        stopped_sign_early, stopped_TAC_early = False, False

        early_stopped_states_sign, early_stopped_states_TAC = None, None
        sign_energies, TAC_energies = None, None

        start_time = time.time()
        sign_stop_time, TAC_stop_time = 0, 0
        for t in range(self.num_iterations):
            # fetch the previous positions and speeds
            positions, speeds = self.current_state[:, :, 0], self.current_state[:, :, 1]

            # Update the positions and speeds
            self.current_state[:, :, 0], self.current_state[:, :, 1] = self.simplectic_update(positions, speeds, t)

            # Compute the new energies
            current_energies = self.compute_energies(positions)

            # Save what needs to be saved
            if self.save_states_history:
                self.states[:, :, t, :] = self.current_state
            if self.save_energies_history:
                self.energies[:, t] = current_energies

            # early stopping
            if self.early_sign_stopping(self.current_state) and not stopped_sign_early: #and not stroped_early:
                stopped_sign_early = True
                early_stopped_states_sign = self.sign(self.current_state)
                sign_energies = self.compute_energies(early_stopped_states_sign)

                sign_stop_time = time.time()

            if self.early_TAC_stopping(self.current_state) and not stopped_TAC_early:
                stopped_TAC_early = True
                early_stopped_states_TAC = self.TAC(self.current_state)
                TAC_energies = self.compute_energies(early_stopped_states_TAC)

                TAC_stop_time = time.time()
                

        # Return what needs to be returned
        if not self.save_energies_history:
            self.energies = None
        if not self.save_states_history:
            self.states = None

        final_stop_time = time.time()
        
        return self.states, self.energies, (self.current_state, early_stopped_states_sign, early_stopped_states_TAC), (current_energies, sign_energies, TAC_energies), (sign_stop_time-start_time, TAC_stop_time-sign_stop_time, final_stop_time-TAC_stop_time)