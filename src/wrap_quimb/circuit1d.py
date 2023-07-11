import quimb as qu
import quimb.tensor as qtn

import numpy as np
import pickle
import uuid
import time
import scipy
import scipy.linalg

import itertools


from numpy import pi, cos, sin, sqrt, exp


class AlternateOneQubitTwoQubit1D:
    '''
    Represents a circuit with alternate layers
    of one qubit gates and two qubit gates
    
    Parameters
    ----------
    
    onequbitgate: one qubit gate
    
    twoqubitgate: two qubit gate

    '''
    
    def __init__ (self, \
                  onequbitgate, \
                  twoqubitgate, \
                  n_spins, max_bond):

        self.onequbitgate = onequbitgate
        self.twoqubitgate = twoqubitgate
        
        self.n_spins = n_spins
        self.max_bond = max_bond
        self.states = []
        
    def evolve_circuit(self, psi_in, depth):
        '''
        Evolves an input state `psi_in` 
        using a `depth` deep circuit
        '''
        self.states += [psi_in]
        
        psi = psi_in
        for d in range(depth):

            for l in range(self.n_spins):
                psi = qtn.gate_TN_1D(psi, G=self.onequbitgate, where=(l,), contract=True, \
                    inplace=False, max_bond=self.max_bond)

            self.states += [psi]

            for l in range(self.n_spins-1):
                psi = qtn.gate_TN_1D(psi, G=self.twoqubitgate, where=(l, l+1), contract='swap+split', \
                    inplace=False, max_bond=self.max_bond)

            self.states += [psi]


class TrotterSuzuki1D:
    '''

    '''
    def __init__ (self, \
        n_qubits, \
        interact_gen, \
        local_gen, \
        interact_energy, \
        local_energy,
        t_initial,
        t_final,
        t_step=None,
        n_steps=None): \

        self.n_spins = n_qubits
        self.interact_gen = interact_gen
        self.local_gen = local_gen
        self.interact_energy = interact_energy
        self.local_energy = local_energy
        self.t_initial = t_initial
        self.t_final = t_final

        self.t_step = t_step
        self.n_steps = n_steps

    def calc_tstep(self):
        '''
        Calculates the time step size
        '''
        if (self.n_steps == None) and (self.t_step == None):
            self.n_steps = 256
            self.t_step = (self.t_final - self.t_initial) / self.n_steps
        elif self.n_steps != None:
            self.t_step = (self.t_final - self.t_initial) / self.n_steps
        elif self.t_step != None:
            self.n_steps = np.round((self.t_final - self.t_initial) / self.t_step)

    def gen_gates(self):
        '''
        Generates the gates
        '''

        self.local_gate = scipy.linalg.expm(
            -1j * self.t_step * self.local_energy * self.local_gen)
    
        self.interact_gate = scipy.linalg.expm(
            -1j * self.t_step * self.interact_energy * self.interact_gen)

    def tevolve_step(self, psi_in):

        psi = psi_in.copy(deep=True)

        for l in range(self.n_qubits):
            psi = qtn.gate_TN_1D(tn=psi, G=self.local_gate, where=l, inplace=True, \
                contract=True)

        for l1, l2 in self.interact_graph:
            psi = qtn.gate_TN_1D(tn=psi, G=self.interact_gate, where=(l1, l2), inplace=True, \
                contract='swap+split')

class TransverseFieldIsing1DKicked (AlternateOneQubitTwoQubit1D):
    '''
    Represents a circuit with alternate layersof one qubit gates and 
    two qubit gates for the transverse field Ising model in 1D

    Parameters
    ----------

    ax_rotate: Axis of rotation

    ax_interact: Axis of interaction

    ang_rotate: Angle of rotation

    ang_interact: Angle of interaction

    '''

    def __init__ (self, ax_rotate, ax_interact, ang_rotate, ang_interact, n_spins):
        self.ax_interact = ax_interact
        self.ax_rotate = ax_rotate
        self.ang_interact = ang_interact
        self.ang_rotate = ang_rotate

        if isinstance(self.ax_interact, str):
            if self.ax_interact.lower() == 'x' \
                or self.ax_interact.lower() == 'y' \
                or self.ax_interact.lower() == 'z':
                self.gen_interact = \
                    qu.kronpow(qu.pauli(self.ax_interact), 2)
        else:
            self.gen_interact = None

        self.gate_interact = \
            scipy.linalg.expm(-1j * 1/4 * self.gen_interact * self.ang_interact)


        if isinstance(self.ax_rotate, str):
            if self.ax_rotate.lower() == 'x' \
            or self.ax_rotate.lower() == 'y' \
            or self.ax_rotate.lower() == 'z':
                self.gen_rotate = \
                    qu.pauli(self.ax_rotate)
    
        else:
                self.gen_rotate = None

        self.gate_rotate = \
            scipy.linalg.expm(-1j * 1/2 * self.gen_rotate * self.ang_rotate)

        AlternateOneQubitTwoQubit1D.__init__(self, self.gate_rotate, self.gate_interact, self.n_spins)

class KickedTFIM1DCircuit:
    '''
    Represents a kicked TFIM in 1D as a 
    quantum circuit
    
    Parameters
    ----------
    
    ax_rotate: axis of rotation
    
    ax_interact: axis of interaction
    
    ang_rotate: angle of rotation
    
    ang_interact: angle of interaction
    '''
    
    def __init__ (self, \
                  ang_rotate, \
                  ang_interact, \
                  ax_rotate, \
                      
                  ax_interact, \
                  n_spins, \
                  max_bond):
        self.ax_rotate = ax_rotate
        self.ax_interact = ax_interact
        
        self.ang_rotate = ang_rotate
        self.ang_interact = ang_interact
        
        self.n_spins = n_spins
        
        if max_bond != None:
            self.max_bond = max_bond
        
        self.states = []

        self._generate_gates()
        
    def _generate_gates (self):
        
        if isinstance(self.ax_rotate, str):
            if self.ax_rotate.lower() == 'x' \
            or self.ax_rotate.lower() == 'y' \
            or self.ax_rotate.lower() == 'z':
                
                self.gen_rotate = \
                    qu.pauli(self.ax_rotate)
                self.gate_rotate = \
                    scipy.linalg.expm(-1j * 1/2 * self.gen_rotate * self.ang_rotate)

        else: # Assuming it is the operator
            self.gen_rotate = self.ax_rotate
            self.gate_rotate = \
                    scipy.linalg.expm(-1j * 1/2 * self.gen_rotate * self.ang_rotate)
                
        if isinstance(self.ax_interact, str):
            if self.ax_interact.lower() == 'x' \
            or self.ax_interact.lower() == 'y' \
            or self.ax_interact.lower() == 'z':
                self.gen_interact = \
                    qu.kronpow(qu.pauli(self.ax_interact), 2)
                self.gate_interact = \
                    scipy.linalg.expm(-1j * 1/4 * self.gen_interact * self.ang_interact)

        else:
                self.gen_interact = None

#     def _apply_local_gates(self, psi_in):
        
#         for l in range(self.n_spins):
#             psi_out = qtn.gate_TN_1D(psi_in, self.gate_rotate, (l,), \
#                                      contract=True, inplace=False,\
#                                     )
        
#         self.states += [psi_out]

#     def _apply_interact_gates(self, psi_in):
        
#         for l in range(self.n_spins-1):
#             psi_out = qtn.gate_TN_1D(psi_in, self.gate_interact, (l, l+1), \
#                                      contract='swap+split', inplace=False, \
#                                     )
        
#         self.states += [psi_out]

        
    def evolve_circuit(self, psi_in, depth):
        '''
        Evolves an input state `psi_in` 
        using a `depth` deep circuit
        '''
        self.states += [psi_in]
        
        psi = psi_in
        for d in range(depth):
            for l in range(self.n_spins):
                psi = qtn.gate_TN_1D(psi, self.gate_rotate, (l,), contract=True, inplace=False, \
                                     max_bond=self.max_bond)

            self.states += [psi]

            for l in range(self.n_spins-1):
                psi = qtn.gate_TN_1D(psi, self.gate_interact, (l, l+1), contract='swap+split', inplace=False, \
                                    max_bond=self.max_bond)

            self.states += [psi]



if __name__ == '__main__':
    n_spins = 16
    zeros_all = '0' * n_spins
    psi_in = qtn.MPS_computational_state(zeros_all)
    
    depth = n_spins*8
    bonds = range(1, n_spins)

    vartheta1 = pi/2
    vartheta2 = pi

    circuit = TransverseFieldIsing1DKicked('x', 'z', vartheta1, vartheta2, n_spins)

    circuit.evolve_circuit(psi_in, depth)