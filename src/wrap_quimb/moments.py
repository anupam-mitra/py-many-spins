import quimb as qu
import quimb.tensor as qtn

import numpy as np
import pickle
import uuid
import time
import scipy
import scipy.linalg

import itertools


def parse_term (term):
    '''
    Parses a term
    '''

    operator_names = term[0]
    sites = terms[1]

    n_operators = len(operator_names)
    n_sites = len(sites)

    answer

    if n_operators == 1 and operators[0].lower() == 'rho':
            answer = 'densitymatrix'
            raise NotImplementedError(answer)

   operators = [qu.pauli(opname) for opname in operators_names]

   return operators, sites


    

    def calc_manyspin_moment (state, ops, ls, n_spins=None):
    
        ls_unique = list(set(ls))
        duplicate_flag = (len(ls_unique) != len(ls))
        
        if not duplicate_flag:
            moment = state.H @ qtn.gate_TN_1D(state, qu.kron(*ops), np.asarray(ls))
        else:
            moment = float("nan")
        return moment

class SpinMomentsCalculation:
    '''
    Calculates spin moments for a time series of matrix product 
    states

    Parameters
    ----------

    dynamics:
    An object representing the dynamics, which has a history of
    the state, in matrix product state representation, as it
    traverses the circuit

    n_spins:
    Number of spins in the dynamics

    term:
    Iterable of tuples to calculate from the matrix product state
    time series

    For example 
    - `(('Sz', 0), ('Sz', 1))` represents the one-spin operator
       $\sigma^{z}_0$
    - `(('Sz', 0), ('Sz', 1))` represents the two-spin operator
       $\sigma^{z}_0 \sigma^{z}_1$
    - `(('Sz', 0), ('Sz', 4))` represents the two-spin operator
       $\sigma^{z}_0 \sigma^{z}_4$
    - `(('Sz', 0), ('Sz', 2), ('Sz', 6), ('Sz', 10))` represents 
       the four-spin operator
       $\sigma^{z}_0 \sigma^{z}_2 \sigma^{z}_6 \sigma^{z}_{10}$
    - `('rho', (0, 2, 7, 8))` represents the four-spin marginal 
        density matrix with the sites $0, 2, 7, 8$ retained and
        others traced out.
    
    '''

    def __init__ (self, dynamics, n_spins, term):

        self.dynamics = dynamics
        self.n_spins = n_spins
        self.term = term


class Circuit1SpinMoments1D:
    '''
    Calculates 1-spin moments for a circuit
    
    Parameters
    ----------
    
    circuit:
    An object representing a 1d quantum circuit, which has 
    history of the state as it traverses the circuit
    
    depth:
    Depth of the circuit, specifying the number of layers of
    two-qubit gates
    
    n_spins:
    Number of spins in the circuit
    
    '''
    
    def __init__ (self, circuit, depth, n_spins):
        self.circuit = circuit
        self.depth = depth
        self.n_spins = n_spins
        
    def calc_moments (self):
        '''
        Calculates the 1 spin moments at all sites for all
        matrix product states occuring during traversal of the
        circuit.
        '''
        self.sites = range(n_spins)
        self.depths = range(0, 2*depth+1)

        self.x_exp = np.zeros((2*depth + 1, n_spins))
        self.y_exp = np.zeros((2*depth + 1, n_spins))
        self.z_exp = np.zeros((2*depth + 1, n_spins))

        for d in self.depths:
            psi = self.circuit.states[d]

            for l in self.sites:
                self.x_exp[d, l] = np.real(psi.H @ qtn.gate_TN_1D(psi, qu.pauli('X'), (l,)))
                self.y_exp[d, l] = np.real(psi.H @ qtn.gate_TN_1D(psi, qu.pauli('Y'), (l,)))
                self.z_exp[d, l] = np.real(psi.H @ qtn.gate_TN_1D(psi, qu.pauli('Z'), (l,)))

    def get_data(self):
        '''
        Returns the data calculated by the 1-spin moment
        calculations. This is useful for plotting
        '''
        return self.sites, self.depths, self.x_exp, self.y_exp, self.z_exp

class Circuit2SpinMoments1D:
    '''
    Calculates 2-spin moments for a circuit
    
    Parameters
    ----------
    
    circuit:
    An object representing a 1d quantum circuit, which has 
    history of the state as it traverses the circuit
    
    depth:
    Depth of the circuit, specifying the number of layers of
    two-qubit gates
    
    n_spins:
    Number of spins in the circuit
    
    '''
    
    def __init__ (self, circuit, depth, n_spins):
        self.circuit = circuit
        self.depth = depth
        self.n_spins = n_spins

    def calc_moments (self):
        '''
        Calculates the 2 spin moments between the 
        middle site and other sites at different distances
        '''
        self.distances = range(1, n_spins//2+1)
        self.depths = range(0, 2*depth+1)

        self.xx_exp = np.zeros((2*depth + 1, n_spins//2))
        self.xy_exp = np.zeros((2*depth + 1, n_spins//2))
        self.xz_exp = np.zeros((2*depth + 1, n_spins//2))

        self.yx_exp = np.zeros((2*depth + 1, n_spins//2))
        self.yy_exp = np.zeros((2*depth + 1, n_spins//2))
        self.yz_exp = np.zeros((2*depth + 1, n_spins//2))

        self.zx_exp = np.zeros((2*depth + 1, n_spins//2))
        self.zy_exp = np.zeros((2*depth + 1, n_spins//2))
        self.zz_exp = np.zeros((2*depth + 1, n_spins//2))

        m = n_spins//2-1 # Middle
        for d in self.depths:
            psi = self.circuit.states[d]

            for l in self.distances:

                self.xx_exp[d, l-1] = np.real(psi.H @ \
                                qtn.gate_TN_1D(psi, qu.kron(qu.pauli('X'), qu.pauli('X')), \
                                    (m, m+l)))
                self.xy_exp[d, l-1] = np.real(psi.H @ \
                                qtn.gate_TN_1D(psi, qu.kron(qu.pauli('X'), qu.pauli('Y')), \
                                    (m, m+l)))
                self.xz_exp[d, l-1] = np.real(psi.H @ \
                                qtn.gate_TN_1D(psi, qu.kron(qu.pauli('X'), qu.pauli('Z')), \
                                    (m, m+l)))
                self.yx_exp[d, l-1] = np.real(psi.H @ \
                                qtn.gate_TN_1D(psi, qu.kron(qu.pauli('Y'), qu.pauli('X')), \
                                    (m, m+l)))
                self.yy_exp[d, l-1] = np.real(psi.H @ \
                                qtn.gate_TN_1D(psi, qu.kron(qu.pauli('Y'), qu.pauli('Y')), \
                                    (m, m+l)))
                self.yz_exp[d, l-1] = np.real(psi.H @ \
                                qtn.gate_TN_1D(psi, qu.kron(qu.pauli('Y'), qu.pauli('Z')), \
                                    (m, m+l)))
                self.zx_exp[d, l-1] = np.real(psi.H @ \
                                qtn.gate_TN_1D(psi, qu.kron(qu.pauli('Y'), qu.pauli('X')), \
                                    (m, m+l)))
                self.zy_exp[d, l-1] = np.real(psi.H @ \
                                qtn.gate_TN_1D(psi, qu.kron(qu.pauli('Z'), qu.pauli('Y')), \
                                    (m, m+l)))
                self.zz_exp[d, l-1] = np.real(psi.H @ \
                                qtn.gate_TN_1D(psi, qu.kron(qu.pauli('Z'), qu.pauli('Z')), \
                                    (m, m+l)))

        self.xx_exp = np.round(self.xx_exp, 10)
        self.xy_exp = np.round(self.xy_exp, 10)
        self.xz_exp = np.round(self.xz_exp, 10)
        self.yx_exp = np.round(self.yx_exp, 10)
        self.yy_exp = np.round(self.yy_exp, 10)
        self.yz_exp = np.round(self.yz_exp, 10)
        self.zx_exp = np.round(self.zx_exp, 10)
        self.zy_exp = np.round(self.zy_exp, 10)
        self.zz_exp = np.round(self.zz_exp, 10)

    def get_data(self):
        '''
        Returns the data calculated by the 1-spin moment
        calculations. This is useful for plotting
        '''
        return self.distances, self.depths, \
            self.xx_exp, self.xy_exp, self.xz_exp, \
            self.yx_exp, self.yy_exp, self.yz_exp, \
            self.zx_exp, self.zy_exp, self.zz_exp

