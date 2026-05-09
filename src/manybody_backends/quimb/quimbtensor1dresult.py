class MPSResult:
    '''
    Stores the result of MPS calculations

    '''

    def __init__ (self, string_uuid, tlist, psi_t):

        self.str_uuid = string_uuid
        self.tlist = tlist
        self.psi_t = psi_t

        self.parameters = {}

    def calc_observable (self):
        #
        # Calculates the observables 
        # using modern quimb.tensor MPS gate methods

        pass


    
################################################################################
class MPSTrajectoryResult:
    '''
    Stores the result of MPS calculation 
    for a single trajectory
    '''

    def __init__ (self, string_uuid, tlist, psi_t, \
        tjumps, whichjumps, random_numbers):

        self.str_uuid = string_uuid

        self.tlist = tlist
        self.psi_t = psi_t

        self.tjumps = tjumps
        self.whichjumps = whichjumps

        self.random_numbers = random_numbers

        self.parameters = {}

import numpy as np
import quimb.tensor as qtn

################################################################################
class QuimbTEBD1DSolver:
    '''
    Performs a TEBD simulation
    '''

    def __init__ (self,
        initial_mps,
        n_sites,
        hamiltonian_model,
        t_initial,
        t_final,
        n_steps,
        trotter_opts,
        split_opts=None):

        self.initial_mps = initial_mps
        self.n_sites = n_sites
        self.hamiltonian_model = hamiltonian_model

        self.t_initial = t_initial
        self.t_final = t_final
        self.n_steps = n_steps

        self.split_opts = split_opts
        self.trotter_opts = trotter_opts

    def run (self):

        self.t_list =  np.linspace(self.t_initial, self.t_final, self.n_steps)

        #self.h_local = self.hamiltonian_model.construct_hamiltonian_quimb(self.n_sites)
        self.h_local = self.hamiltonian_model.build_local_ham(self.n_sites)

        self.tebd = qtn.TEBD(
            self.initial_mps,
            self.h_local,
            split_opts=self.split_opts,
        )

        self.states = np.empty_like(self.t_list, dtype=object)

        for t_index in range(len(self.t_list)):
            mps_current = next(
                self.tebd.at_times([self.t_list[t_index]], **self.trotter_opts)
            )

            self.states[t_index] = mps_current
