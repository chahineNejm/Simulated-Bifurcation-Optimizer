from SBM import SBM
import numpy as np
import datetime
import concurrent.futures

class SimulationManager:
    def __init__(self, step_size, num_iterations, num_simulations, J, H, pumping_rate=None, stopping_criterion=0, save_states_history=True, save_energies_history=True, n_threads=1, savetofile=True):
        self.step_size = step_size
        self.num_iterations = num_iterations
        self.num_simulations = num_simulations
        self.J = J
        self.H = H
        self.pumping_rate = pumping_rate
        self.savetofile = savetofile
        self.stopping_criterion = stopping_criterion
        self.save_states_history = save_states_history
        self.save_energies_history = save_energies_history
        self.n_threads = n_threads

    def create_sbm_instance(self):
        """
        Creates an instance of the SBM class with the right number of simulations its gonna run, that we are going to run multiple times in parallel

        Returns:
            machine (SBM): The Simulated Bifurcation Machine instance it generated
        """
        # Create the ising model with the right number of simulations its gonna run, that we are going to run multiple times in parallel
        n_sims_per_thread = int(self.num_simulations / self.n_threads)

        machine = SBM(
            J=self.J,
            H=self.H,
            step_size=self.step_size,
            num_iterations=self.num_iterations,
            num_simulations=n_sims_per_thread,
            stopping_criterion=self.stopping_criterion,
            save_states_history=self.save_states_history,
            save_energies_history=self.save_energies_history,
            custom_pumping_rate=self.pumping_rate
            )
        
        return machine
        
    def run_simulation(self):
            """
            Runs the simulation in small batches in parrallel

            Args:
                n_threads (int): Number of parralel simulations to run
                n_sims_per_thread (int): Number of simulations to run in each thread
            """
            # Run the simulations in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.create_sbm_instance().simulate) for _ in range(self.n_threads)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]

            # Unpack the results
            loc_states_arrays, loc_energies_arrays, loc_last_states_array, loc_last_TAC_states_array, loc_last_sign_states_array, loc_last_energies_array, loc_last_TAC_energies_array, loc_last_sign_energies_array, loc_final_times_array, loc_sign_times_array, loc_TAC_times_array = [], [], [], [], [], [], [], [], [], [], []

            for i in range(self.n_threads):
                loc_states, loc_energies, loc_last_state, loc_last_energies, loc_times = results[i]

                # special unpacking for early stopping
                loc_last_state, loc_last_TAC_state, loc_last_sign_state = loc_last_state
                loc_last_energies, loc_last_TAC_energies, loc_last_sign_energies = loc_last_energies
                loc_final_time, loc_sign_time, loc_TAC_time = loc_times

                loc_states_arrays.append(loc_states)
                loc_energies_arrays.append(loc_energies)
                loc_last_states_array.append(loc_last_state)
                loc_last_TAC_states_array.append(loc_last_TAC_state)
                loc_last_sign_states_array.append(loc_last_sign_state)
                loc_last_energies_array.append(loc_last_energies)
                loc_last_TAC_energies_array.append(loc_last_TAC_energies)
                loc_last_sign_energies_array.append(loc_last_sign_energies)
                loc_final_times_array.append(loc_final_time)
                loc_sign_times_array.append(loc_sign_time)
                loc_TAC_times_array.append(loc_TAC_time)
        
            agg_states, agg_energies = None, None
            if self.save_states_history:
                agg_states = np.concatenate(loc_states_arrays, axis=0)
            if self.save_energies_history:
                agg_energies = np.concatenate(loc_energies_arrays, axis=0)

            agg_last_states = np.concatenate(loc_last_states_array, axis=0)
            agg_last_TAC_states = None#np.concatenate(loc_last_TAC_states_array, axis=0)
            agg_last_sign_states = None#np.concatenate(loc_last_sign_states_array, axis=0)

            agg_last_energies = np.concatenate(loc_last_energies_array, axis=0)
            agg_last_TAC_energies = None#np.concatenate(loc_last_TAC_energies_array, axis=0)
            agg_last_sign_energies = None#np.concatenate(loc_last_sign_energies_array, axis=0)

            # agg_final_times = np.concatenate(loc_final_times_array, axis=0)
            # agg_sign_times = np.concatenate(loc_sign_times_array, axis=0)
            # agg_TAC_times = np.concatenate(loc_TAC_times_array, axis=0)

            return agg_states, agg_energies, agg_last_states, agg_last_TAC_states, agg_last_sign_states, agg_last_energies, agg_last_TAC_energies, agg_last_sign_energies, loc_final_times_array, loc_sign_times_array, loc_TAC_times_array