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
        # using quimb.tensor.tensor_1d.gate

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
import quimb
import quimb.tensor

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

        self.tebd = quimb.tensor.TEBD(self.initial_mps, self.h_local)
        self.tebd.split_opts = self.split_opts

        self.states = np.empty_like(self.t_list, dtype=object)

        for t_index in range(len(self.t_list)):
            mps_current = self.tebd.at_times(self.t_list[t_index], **self.trotter_opts)

            self.states[t_index] = mps_current

################################################################################
class UniformTwoBodyInteraction:
    """
    Represents a hamiltonian with two body interactions and local terms
    """

    def __init__ (self, \
        operators_interaction,
        operators_local,
        energy_interaction,
        energy_local,
        interact_graph=None):
        
        self.operators_interaction = operators_interaction
        self.operators_local = operators_local
        self.energy_interaction = energy_interaction
        self.energy_local = energy_local

        self.interact_graph = interact_graph

    def construct_hamiltonian_qutip(self, n_spins):
        '''
        Constructs a Hamiltonian in a qutip representation
        '''

        h_local_terms = []
        h_interact_terms = []

        #for ll in range(len(local_ops)):
            #builder.add_term(local_energies[ll], local_ops[ll])

        assert len(self.operators_interaction) == len(self.energy_interaction)

        for ll in range(len(self.operators_interaction)):
                builder.add_term(interact_energies[ll], interact_ops[ll], interact_ops[ll])

        for energy, operator in zip(self.energy_local, self.operators_local):
            h_local_terms += \
                [energy * qutip.qip.operations.expand_operator(operator, 
                    n_spins, targets=(n,)) \
                for n in range(n_spins)]

        for l1, l2 in self.interact_graph.get_edges():
            weight = self.interact_graph.get_edge(l1, l2)
            
            for energy, operator_pair in\
                zip(self.energy_interaction, self.operators_interaction):
                
                h_interact_terms += \
                    [energy * weight * qutip.qip.operations.expand_operator(\
                        qutip.tensor(operator_pair[0], operator_pair[1]), \
                        n_spins, targets=(l1, l2))]
                
        if False:

            for energy, operator_pair in\
                zip(self.energy_interaction, self.operators_interaction):
                h_interact_terms += \
                    [energy * qutip.qip.operations.expand_operator(\
                            qutip.tensor(operator_pair[0], operator_pair[1]), \
                            n_spins, targets=(n, n+1)) \
                    for n in range(n_spins-1)]

        self.h_local_terms = h_local_terms
        self.h_interact_terms = h_interact_terms

        hamiltonians = h_local_terms + h_interact_terms
        return hamiltonians
################################################################################