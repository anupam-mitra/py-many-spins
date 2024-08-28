import qutip
import qutip.qip
import qutip.qip.operations
import itertools
import numpy as np
from numpy import sqrt

import pickle

# __all__ = [
#     InteractionGraph,
#     Interaction1DNearest,
#     Interaction1DNextNearest
# ]

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
        try:
            self.dim_local = self.operators_local[0].dims[0]
        except Exception as e:
            self.dim_local = [2 for _ in range(interact_graph.n_sites)]

    def construct_hamiltonian_qutip(self, n_spins):
        '''
        Constructs a Hamiltonian in a qutip representation
        '''

        h_local_terms = []
        h_interact_terms = []
        dims = [self.dim_local[0]] * n_spins

        for energy, operator in zip(self.energy_local, self.operators_local):
            h_local_terms += \
                [energy * qutip.qip.operations.expand_operator(operator,
                    n_spins, targets=(n,), dims=dims) \
                for n in range(n_spins)]

        for l1, l2 in self.interact_graph.get_edges():
            weight = self.interact_graph.get_edge(l1, l2)

            for energy, operator_pair in\
                zip(self.energy_interaction, self.operators_interaction):

                h_interact_terms += \
                    [energy * weight * qutip.qip.operations.expand_operator(\
                        qutip.tensor(operator_pair[0], operator_pair[1]), \
                        n_spins, targets=(l1, l2), dims=dims)]

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

################################################################################
class UniformOneBodyDecoherence:
    """
    Represents a model of Lindbladian decoherence with
    """

    def __init__ (self, \
        operators_jump,
        rate_jump):

        self.operators_jump = operators_jump
        self.rate_jump = rate_jump

    def construct_jumps_qutip(self, n_spins):
        '''
        Constructs a jump operators in a qutip representation
        '''
        jmp_local_terms = []

        for rate, operator in zip(self.rate_jump, self.operators_jump):
            jmp_local_terms += \
                [sqrt(rate) * qutip.qip.operations.expand_operator(operator,
                    n_spins, targets=(n,)) \
                for n in range(n_spins)]

        self.jmp_local_terms = jmp_local_terms
        return jmp_local_terms

    construct_local_jumpops = construct_jumps_qutip

################################################################################
class QuantumDynamicsStateEvolution:
    '''
    Represents a simultion of quantum dynamics
    '''

    def __init__ (self, \
        t_initial, \
        t_final, \
        n_steps):

        self.t_initial = t_initial
        self.t_final = t_final
        self.n_steps = n_steps

    def get_times_list(self):

        if hasattr(self, times_list):
            return self.times_list
        else:
            return None

    def get_states_list(self):

        if hasattr(self, times_list):
            return self.times_list
        else:
            return None

class QutipSESolve (QuantumDynamicsStateEvolution):

    def __init__ (self, \
        t_initial, \
        t_final,
        n_steps, \
        hamiltonian_model, \
        n_dof, \
        initial_state):

        QuantumDynamicsStateEvolution.__init__(self, t_initial, t_final, n_steps)
        self.n_dof = n_dof
        self.hamiltonian_model = hamiltonian_model
        self.initial_state = initial_state

        self.is_se = True
        self.is_me = False
        self.is_mc = False

    def run (self):

        self.t_list =  np.linspace(self.t_initial, self.t_final, self.n_steps)
        self.hamiltonians = self.hamiltonian_model.construct_hamiltonian_qutip(self.n_dof)

        self.resultse = qutip.sesolve(self.hamiltonians, self.initial_state, \
            self.t_list)

        self.times_list = self.resultse.times
        #self.states = np.asarray(self.resultse.states, dtype=object)
        self.states = self.resultse.states

class QutipMCSolve (QuantumDynamicsStateEvolution):

    def __init__ (self, \
        t_initial, \
        t_final,
        n_steps, \
        hamiltonian_model, \
        decoherence_model, \
        n_dof, \
        initial_state, \
        n_trajectories):

        QuantumDynamicsStateEvolution.__init__(self, t_initial, t_final, n_steps)
        self.n_dof = n_dof
        self.hamiltonian_model = hamiltonian_model
        self.decoherence_model = decoherence_model
        self.n_trajectories = n_trajectories
        self.initial_state = initial_state

        self.is_se = False
        self.is_me = False
        self.is_mc = True

    def run (self):
        self.t_list =  np.linspace(self.t_initial, self.t_final, self.n_steps)
        self.hamiltonians = self.hamiltonian_model.construct_hamiltonian_qutip(self.n_dof)
        self.jumpops = self.decoherence_model.construct_local_jumpops(self.n_dof)

        self.resultmc = qutip.mcsolve(\
            self.hamiltonians, \
            self.initial_state, \
            self.t_list, \
            c_ops=self.jumpops, \
            ntraj=self.n_trajectories)

        self.times_list = self.resultmc.times
        self.states = np.asarray(self.resultmc.states, dtype=object)


class QutipMESolve (QuantumDynamicsStateEvolution):

    def __init__ (self, \
        t_initial, \
        t_final,
        n_steps, \
        hamiltonian_model, \
        decoherence_model, \
        n_dof, \
        initial_state):

        QuantumDynamicsStateEvolution.__init__(self, t_initial, t_final, n_steps)
        self.n_dof = n_dof
        self.hamiltonian_model = hamiltonian_model
        self.decoherence_model = decoherence_model
        self.initial_state = initial_state

        self.is_se = False
        self.is_me = True
        self.is_mc = False

    def run(self):
        self.t_list =  np.linspace(self.t_initial, self.t_final, self.n_steps)
        self.hamiltonians = self.hamiltonian_model.construct_hamiltonian_qutip(self.n_dof)
        self.jumpops = self.decoherence_model.construct_local_jumpops(self.n_dof)

        self.resultme = qutip.mesolve(\
            self.hamiltonians, \
            self.initial_state, \
            self.t_list, \
            c_ops=self.jumpops)

        self.times_list = self.resultme.times
        #self.states = np.asarray(self.resultme.states, dtype=object)
        self.states = self.resultme.states

################################################################################
class ManyBodyState:
    '''
    Represents a many body state
    '''
    def __init__ (self,\
        n_dof, \
        n_dim_local):

        self.n_dof = n_dof
        self.n_dim_local = n_dim_local

        self.qutip_ket = None
        self.qutip_dm = None

        self.quimb_mps = None

class ManySpinProductState:
    '''
    Represents a product state of many spin half
    degrees of freedom
    '''

    def __init__(self, \
        j_spin, \
        n_spins, \
        ang_polar, \
        ang_azimuthal):

        self.j_spin = j_spin
        self.n_spins = n_spins
        self.ang_polar = ang_polar
        self.ang_azimuthal = ang_azimuthal


    def _calc_qutip_ket(self):
        if not (hasattr(self.ang_polar, '__iter__') \
            or hasattr(self.ang_polar, '__iter__')):

            self.qutip_ket = qutip.spin_coherent(self.j_spin, \
                self.ang_polar, self.ang_azimuthal)

        else:
            raise NotImplementedError(\
                'Only identically prepared state is implemented')

    def get_qutip_ket (self):
        if self.qutip_ket == None:
            self._calc_qutip_ket(self)

        return self.qutip_ket

    def _calc_qutip_dm(self):
        if self.qutip_ket == None:
            self._calc_qutip_ket(self)

        self.qutip_dm = qutip.ket2dm(self.qutip_ket)

    def get_qutip_dm (self):
        if self.qutip_dm == None:
            self._calc_qutip_dm(self)

        return self.qutip_dm
