import numpy as np
import qutip

from manybody_calculator import QutipSingleSimulationCalculator

import convquimbqutip

class EntanglementEntropyCalculator(QutipSingleSimulationCalculator):
    '''
    Represents a calculation of entanglement entropy
    '''
    def __init__ (self, dynamics, n_spins, keep=None, base=2):
        QutipSingleSimulationCalculator.__init__(self, dynamics)
        self.n_spins = n_spins
        
        if keep != None:
            self.keep = keep
        else:
            self.keep = list(range(self.n_spins//2))
            
        self.base = base
        
    def calc_entangle_entropy(self):
        assert self.dynamics.is_se
        
        self.entropies = np.empty((self.n_steps,), dtype=float)
        
        for s in range(self.n_steps):
            psi = self.dynamics.states[s]
            self.entropies[s] = \
                qutip.entropy.entropy_vn(qutip.ptrace(psi, self.keep), self.base)
        
    
def most_probable_state (rho, r=1):
    """
    Returns the most `r`th probable pure state from the
    ensemble of pure state represented by a density matrix
    `rho`
    
    `r=1` gives the most probable state.
    
    `r=2` gives the second most probable state
    """
    return rho.eigenstates()[1][-r]

def conv_truncate_qutip_ket(qutip_ket, cutoff=None, max_bond=None):
    '''
    Truncates a qutip ket by converting it to a quimb matrix 
    product state and converts it back to a qutip ket
    '''
    
    quimb_mps = convquimbqutip.convert_qutip_ket_to_quimb_mps(qutip_ket, \
                                max_bond=max_bond, cutoff=cutoff)
    qutip_ket_trunc = convquimbqutip.convert_quimb_mp_to_qutip_qobj(quimb_mps)
            
    norm = qutip_ket_trunc.norm()
    qutip_ket_trunc = qutip_ket_trunc / norm
    
    return qutip_ket_trunc

class TruncationCalculator(QutipSingleSimulationCalculator):
    '''
    Represents a calculation of a qutip representation
    of a truncated matrix product state
    '''
    def __init__ (self, dynamics, n_spins, bonddim):
        QutipSingleSimulationCalculator.__init__(self, dynamics)
        self.n_spins = n_spins
        self.bonddim = bonddim
        
        self.is_se = dynamics.is_se
        self.is_mc = dynamics.is_mc
        self.is_me = dynamics.is_me
        
    def _calc_trunc_states_se (self):
        self.states = []
        
        for s in range(self.n_steps):
            psi = self.dynamics.states[s]
            psi_trunc = conv_truncate_qutip_ket(psi, max_bond=self.bonddim)
            self.states += [psi_trunc]
            
    def _calc_trunc_states_mc (self):
        assert hasattr(self.dynamics, 'n_trajectories')
        self.n_trajectories = self.dynamics.n_trajectories
        self.states = np.empty((self.n_trajectories, self.n_steps), dtype=object)
        
        for r in range(self.n_trajectories):
            for s in range(self.n_steps):
                psi = self.dynamics.states[r, s]
                psi_trunc = conv_truncate_qutip_ket(psi, max_bond=self.bonddim)
                self.states[r, s] = psi_trunc
                
    def _calc_trunc_states_me(self):
        raise NotImplementedError('ME not supported yet')
                
    def calc_trunc_states (self):
        '''
        Calculates the truncated states for for all states in the timeseries 
        of states in the dynamics object
        '''
        
        if self.dynamics.is_se:
            print('_calc_trunc_se')
            self._calc_trunc_states_se()
        
        elif self.dynamics.is_me:
            print('_calc_trunc_me')
            self._calc_trunc_states_me()
            
        elif self.dynamics.is_mc:
            print('_calc_trunc_mc')
            self._calc_trunc_states_mc()
    
    
class TruncationCalculatorOld:
    '''
    Truncated a qutip ket using a matrix product state
    representation.
    
    '''
    
    def __init__ (self, dynamics, n_spins):

        self.dynamics = dynamics
        self.n_spins = n_spins
        
    def calc_trunc_mps_mc(self, bonddim):
        
        if isinstance(self.dynamics.states, list):
            n_steps = len(self.dynamics.states)
            states_flat = np.empty((n_steps,), dtype=object)
            for s in range(n_steps):
                states_flat[s] = self.dynamics.states[s]
        else:
            states_flat = self.dynamics.states.flatten()

        trunc_states_flat = np.zeros_like(states_flat, dtype=object)

        for ix in range(len(states_flat)):
            qutip_ket_curr = states_flat[ix]
            quimb_mps_curr = \
                convquimbqutip.convert_qutip_ket_to_quimb_mps(qutip_ket_curr, \
                                max_bond=bonddim)

            trunc_states_flat[ix] = \
                convquimbqutip.convert_quimb_mp_to_qutip_qobj(quimb_mps_curr)
            
            norm = np.sqrt((trunc_states_flat[ix].dag() * trunc_states_flat[ix])[0, 0])
            
            trunc_states_flat[ix] = trunc_states_flat[ix] / norm
        
        if isinstance(self.dynamics.states, list):
            self.trunc_states = trunc_states_flat
        else:
            self.trunc_states = np.reshape(trunc_states_flat, self.dynamics.states.shape)
        
        self.states = self.trunc_states