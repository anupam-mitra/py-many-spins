import qutip

import numpy as np
from numpy import sqrt

from deprecated import deprecated

#from . import compositesystem

from .manybodystateevolve import *


################################################################################
@deprecated(reason="Use different class")
class TransverseFieldIsingModelNearestNeighbor1D:
    '''
    Represents a transverse field Ising model
    with interactions

    - Only spin 1/2 is implemented
    - Only qutip operators are implemented
    '''

    def __init__ (self,\
        interaction_energy, \
        transversefield_energy, \
        interaction_axis='z', \
        transversefield_axis='x', \
        boundarycondition='open'

        ):
        self.interaction_axis = interaction_axis
        self.tranversefield_axis = transversefield_axis
        self.interaction_energy = interaction_energy
        self.transversefield_energy = transversefield_energy
        self.boundarycondition = boundarycondition

        if self.interaction_axis == 'x':
            self.interact_op = qutip.sigmax()

    def construct_hamiltonian_qutip (self, n_spins):
        '''
        Constructs the Hamiltonian as an array of qutip
        objects for `n_spins`
        '''

        hamiltonians = []
        # Construct the transverse field term
        if self.transversefield_energy != 0:
            if self.tranversefield_axis.lower() == 'x':
                htransversefield = sum([\
                    self.transversefield_energy * \
                    qutip.qip.operations.expand_operator(\
                    qutip.sigmax(), n_spins, targets=l)
                    for l in range(n_spins)])
                hamiltonians += [htransversefield]
                
            elif self.tranversefield_axis.lower() == 'z':
                htransversefield = sum([\
                    self.transversefield_energy * \
                    qutip.qip.operations.expand_operator(\
                    qutip.sigmaz(), n_spins, targets=l)
                    for l in range(n_spins)])
                hamiltonians += [htransversefield]

            elif self.tranversefield_axis.lower() == 'y':
                htransversefield = sum([\
                    self.transversefield_energy * \
                    qutip.qip.operations.expand_operator(\
                    qutip.sigmay(), n_spins, targets=l)
                    for l in range(n_spins)])
                hamiltonians += [htransversefield]


        if self.boundarycondition.lower() == 'periodic':
            pbcFlag = True
        else:
            pbcFlag = False

        # Construct the interaction term
        if self.interaction_energy != 0 :
            if self.interaction_axis.lower() == 'z':
                hinteractions = sum([\
                    self.interaction_energy * \
                    qutip.qip.operations.expand_operator(\
                    qutip.tensor(qutip.sigmaz(), qutip.sigmaz()),
                    n_spins, targets=(l, l+1))
                    for l in range(n_spins-1)])

                if pbcFlag:
                    hinteractions += \
                        self.interaction_energy * \
                        qutip.qip.operations.expand_operator(\
                        qutip.tensor(qutip.sigmaz(), qutip.sigmaz()),
                        n_spins, targets=(n_spins-1, 0))
                hamiltonians += [hinteractions]

            elif self.interaction_axis.lower() == 'x':
                hinteractions = sum([\
                    self.interaction_energy * \
                    qutip.qip.operations.expand_operator(\
                    qutip.tensor(qutip.sigmax(), qutip.sigmax()),
                    n_spins, targets=(l, l+1))
                    for l in range(n_spins-1)])

                if pbcFlag:
                    hinteractions += \
                        self.interaction_energy * \
                        qutip.qip.operations.expand_operator(\
                        qutip.tensor(qutip.sigmax(), qutip.sigmax()),
                        n_spins, targets=(n_spins-1, 0))
                hamiltonians += [hinteractions]

            elif self.interaction_axis.lower() == 'y':
                hinteractions = sum([\
                    self.interaction_energy * \
                    qutip.qip.operations.expand_operator(\
                    qutip.tensor(qutip.sigmay(), qutip.sigmay()),
                    n_spins, targets=(l, l+1))
                    for l in range(n_spins-1)])

                if pbcFlag:
                    hinteractions += \
                        self.interaction_energy * \
                        qutip.qip.operations.expand_operator(\
                        qutip.tensor(qutip.sigmay(), qutip.sigmay()),
                        n_spins, targets=(n_spins-1, 0))
                hamiltonians += [hinteractions]

        return hamiltonians
################################################################################

################################################################################
@deprecated(reason="Use different class")
class SpinModelDecoherence:
    '''
    Represents an spin model with Markovian decoherence
    given by jump operators

    - Only spin 1/2 is implemented
    - Only qutip operators are implemented
    - Only $\sqrt{\Gamma_+}\sigma_+$, $\sqrt{\Gamma_-}\sigma_-$ and
      $\sqrt{\Gamma_z}\sigma_z$ are implemented
    '''

    def __init__ (self, \
            gammaplus, \
            gammaminus, \
            gammaz, \
        ):
        self.gammaplus = gammaplus
        self.gammaminus = gammaminus
        self.gammaz = gammaz

    def construct_local_jumpops (self, n_spins):
        '''
        Constructs local jump operators, acting on each spin independently
        as an array of qutip objects for `n_spins`
        '''

        jumpops = []

        if self.gammaminus != 0:
            for n in range(n_spins):
                jumpops += \
                    [sqrt(self.gammaminus) * \
                        compositesystem.calcLocalOp(n_spins, qutip.sigmam(), n)]
        if self.gammaplus != 0:
            for n in range(n_spins):
                jumpops += \
                    [sqrt(self.gammaplus) * \
                        compositesystem.calcLocalOp(n_spins, qutip.sigmap(), n)]
        if self.gammaz != 0:
            for n in range(n_spins):
                jumpops += \
                    [sqrt(self.gammaz) * \
                        compositesystem.calcLocalOp(n_spins, qutip.sigmaz(), n)]

        return jumpops

    def construct_collective_jumpops (self, n_spins):
            '''
            Constructs collective jump operators, acting on the collective spin
            as an array of qutip objects for `n_spins`
            '''

            jumpops = []

            if self.gammaminus != 0:
                jumpops += \
                    [sqrt(self.gammaminus) * \
                        compositesystem.calcCollectiveOp(n_spins, qutip.sigmam())]
            if self.gammaplus != 0:
                jumpops += \
                    [sqrt(self.gammaplus) * \
                        compositesystem.calcCollectiveOp(n_spins, qutip.sigmap())]
            if self.gammaz != 0:
                jumpops += \
                    [sqrt(self.gammaz) * \
                        compositesystem.calcCollectiveOp(n_spins, qutip.sigmaz())]

            return jumpops
################################################################################


################################################################################
class SpinDynamicsSimulation:
    '''
    Represents a simulation of quantum dynamics of a spin system with `n_spins`

    - Assume spin 1/2
    - Assume initial state is a pure, product state
    '''

    def __init__ (self,\
            t_initial, \
            t_final, \
            n_spins, \
            theta_initial, \
            phi_initial, \
            n_steps, \
            hamiltonianModel, \
            decoherenceModel, \
        ):
        self.t_initial = t_initial
        self.t_final = t_final
        self.n_spins = n_spins
        self.theta_initial = theta_initial
        self.phi_initial = phi_initial
        self.n_steps = n_steps
        self.hamiltonianModel = hamiltonianModel
        self.decoherenceModel = decoherenceModel
        
        self.construct_initial_state_qutip()

    def construct_initial_state_qutip (self):
        '''
        Constructs the initial state
        '''

        ket_initial_onebody = \
            qutip.spin_coherent(j=1/2, theta=self.theta_initial, phi=self.phi_initial, type='ket')

        self.ket_initial_onebody = ket_initial_onebody

        ket_initial_manybody = \
            qutip.tensor([ket_initial_onebody] * self.n_spins)

        self.ket_initial_manybody = ket_initial_manybody


    def solve_montecarlo (self, n_trajectories):
        '''
        Solves the Master equation using the Monte Carlo wave function
        method with `n_trajectories`
        '''

        self.n_trajectories = n_trajectories
        self.t_list =  np.linspace(self.t_initial, self.t_final, self.n_steps)

        self.hamiltonians = self.hamiltonianModel.construct_hamiltonian_qutip(self.n_spins)
        self.jumpops = self.decoherenceModel.construct_local_jumpops(self.n_spins)

        if len(self.jumpops) == 0:
            self.solve_schroedingerequation()
            self.resultmc = self.resultse
        else:
            self.resultmc = qutip.mcsolve(self.hamiltonians, self.ket_initial_manybody, \
                self.t_list, c_ops=self.jumpops, ntraj=n_trajectories)


    def solve_masterequation (self):
        '''
        Solves the Master equation directly
        '''

        self.rho_initial_manybody = qutip.ket2dm(self.ket_initial_manybody)
        self.t_list =  np.linspace(self.t_initial, self.t_final, self.n_steps)

        self.hamiltonians = self.hamiltonianModel.construct_hamiltonian_qutip(self.n_spins)
        self.jumpops = self.decoherenceModel.construct_local_jumpops(self.n_spins)

        self.resultme = qutip.mesolve(self.hamiltonians, self.rho_initial_manybody, self.t_list, \
            c_ops=self.jumpops)


    def solve_schroedingerequation (self):
        '''
        Solves the time dependent Schroedinger equation, assuming no decoherence
        '''

        self.t_list =  np.linspace(self.t_initial, self.t_final, self.n_steps)

        self.hamiltonians = self.hamiltonianModel.construct_hamiltonian_qutip(self.n_spins)

        self.resultse = qutip.sesolve(self.hamiltonians, self.ket_initial_manybody, \
            self.t_list)


    @staticmethod
    def run_simulation ():
        '''
        Runs the simulation and saves the result
        '''

        # Prepare a uuid to refer to this simulation

        # Check for whether a simulation exists in the database

        # If there is no decoherence, solve the time dependent Schroedinger
        # equation

        # Save the results to persistent memory

        pass

################################################################################
