
import numpy as np
import quimb 
import quimb.tensor

import time
import scipy
import scipy.linalg

import pandas
import pickle
import uuid
import os

from numpy import sin, cos, exp, sqrt, pi

from . import sqlwrap

def uuidGen():
    '''
    Wrapping the generation of uuids
    using the module `uuid` for identifying
    parameter sets and simuations
    '''
    uuidStr = str(uuid.uuid4())
    return uuidStr

def spinHalfState(angPolar, angAzimuth):
    '''

    '''
    return [cos(angPolar/2),\
           sin(angPolar/2) * exp(1j * angAzimuth)]    

'''
Relating the simulation to the hamiltonian, decoherence and approximation

simulations
(uuidSimulation, uuidHamiltonian, uuidDecoherence, uuidSolution,
uuidApproximation, nSpins)

states
(uuidState, uuidSimulation, indexTime, indexTrajectory, marginalLocations)

hamiltonian
(uuidHamiltonian, jxx, jyy, jzz, bx, by, bz)

decoherence
(uuidDecoherence, gammax, gammay, gammaz, gammap, gammam)

solutionMethod
(uuidSolution, tStep, description)

approximations
(uuidApproximation, bonddim, threshold, method)

initialCondition
(uuidInitialCondition, category, uuidInCategory)

spincoherentStates
(uuidSpinCoherent, angPolar, angAzimuth)
'''

class SpinDynamicsCalculation:
    """
    Represents an instance of spin dynamics simulation

    """

    quimbOps = {
        'X': quimb.pauli('X'),
        'Y': quimb.pauli('Y'),
        'Z': quimb.pauli('Z'),
        'XX': quimb.pauli('X') & quimb.pauli('X'),
        'YY': quimb.pauli('Y') & quimb.pauli('Y'),
        'ZZ': quimb.pauli('Z') & quimb.pauli('Z'),
    }

    def __init__ (self,
        hamiltonianParams,
        decoherenceParams,
        approximationParams,
        solutionParams,
        initialConditionParams,
        nSpins,
        ):
        self.hamiltonianParams = hamiltonianParams
        self.decoherenceParams = decoherenceParams
        self.approximationParams = approximationParams
        self.solutionParams = solutionParams
        self.initialConditionParams = initialConditionParams
        self.nSpins = nSpins


    def check_solved(self):
        """Queries the database to see if the dynamics has
        been calculated. If it has been calculated, the field
        `self.tIndexSoFar` is updated and the state
        is updated.
        """

        # Query Hamiltonians
        dfHamiltonians = sqlwrap.query_Hamiltonians(self.hamiltonianParams
        )
        if dfHamiltonians.size > 0:
            self.uuidHamitonian = dfHamiltonians.uuidHamiltonian[0]
        else:
            uuidHamitonianCurrent = None

        # Query Approximations
        dfApproximations = sqlwrap.query_Approximations(self.approximationParams)
        if dfApproximations.size > 0:
            self.uuidApproximation = dfApproximations.uuidApproximation[0]
        else:
            uuidApproximationCurrent = None

        # Query SolutionMethods
        dfSolutionMethods = sqlwrap.query_SolutionMethods(self.solutionParams)
        if dfApproximations.size > 0:
            self.uuidSolution = dfSolutionMethods.uuidSolution[0]
        else:
            uuidSolution = None

        # Query Decoherence
        dfDecoherence = sqlwrap.query_Decoherence(self.decoherenceParams)
        if dfDecoherence.size > 0:
            self.uuidDecoherence = dfDecoherence.uuidDecoherence[0]
        else:
            self.uuidDecoherence = None

        # Query InitialCondition
        dfInitialCondition = sqlwrap.query_InitialConditions(self.initialConditionParams)


        # Query States
        pass

    def to_sql (self):
        """Saves the data to SQL database
        """
        self.uuidHamiltonian = uuidGen()
        hamiltonianRow = {
            'uuidHamiltonian': self.uuidHamiltonian,
            **self.hamiltonianParams,
        }

        self.uuidApproxmation = uuidGen()
        approximationRow = {
            'uuidApproximation': self.uuidApproxmation,
            **self.approximationParams,
        }

        self.uuidSolution = uuidGen()
        solutionMethodsRow = {
            'uuidSolution' : self.uuidSolution,
            **self.solutionParams,
            #'tStep': tStep,
            #'description': 'quimbTEBD',
            #'order': 4,
        }

        self.uuidInitialCondition = uuidGen()
        initialConditionRow = {
            'uuidInitialCondition': self.uuidInitialCondition,
            **self.initialConditionParams
            # 'category': 'SpinCoherent',
            # 'uuidInCategory': None
        }

        self.uuidSimulation = uuidGen()
        simulationRow = {
            'uuidSimulation': self.uuidSimulation,
            'uuidHamiltonian': self.uuidHamiltonian,
            'uuidDecoherence': None,
            'uuidSolution': self.uuidSolution,
            'uuidApproximation': self.uuidApproxmation,
            'nSpins': self.nSpins
        }

    def setup_quimb_tebd (self):

        # Set the parameters for Trotter Suzuki formulas
        # and the parameters for splitting tensors
        # using singular value decomposition
        quimbTEBDParams = {
            'tol': None,
            'dt': None,
            'split_opts': {
                'max_bond': self.approximationParams.get('bonddimMax'),
                'cutoff': self.approximationParams.get('svThreshold'),
            },
        }

        # Construct the hamiltonian as an object of type
        # quimb.tensor.tensor_1d.Local1DHam
        h2_terms = []

        jxx = self.hamiltonianParams.get('jxx')
        if  jxx != None:
            h2_terms += [jxx * self.quimbOps.get('XX')]

        jyy = self.hamiltonianParams.get('jyy')
        if jyy != None:
            h2_terms += [jyy * self.quimbOps.get('YY')]
                
        jzz = self.hamiltonianParams.get('jzz')
        if jzz != None:
            h2_terms += [jzz * self.quimbOps.get('ZZ')]

        h1_terms = []

        bx = self.hamiltonianParams.get('bx')
        if  bx != None:
            h1_terms += [bx * self.quimbOps.get('X')]
        
        by = self.hamiltonianParams.get('by')
        if  by != None:
            h1_terms += [by * self.quimbOps.get('Y')]

        bz = self.hamiltonianParams.get('bz')
        if  bz != None:
            h1_terms += [bz * self.quimbOps.get('Z')]

        # Prepare the 2-site Hamiltonian
        h2 = sum(h2_terms)

        # Prepare the 2-site Hamiltonian
        h1 = sum(h1_terms)

        quimbHamiltonianParams = {
            'L': self.nSpins,
            'H2': h2,
            'H1': h1,
        }

        localHam1d = quimb.tensor.tensor_1d.LocalHam1D(\
            **quimbHamiltonianParams)

        # Construct the initial state
        wallTimeStart = time.time_ns()

        if self.initialConditionParams.get('type') == 'SpinCoherent':
            angAzimuth = self.initialConditionParams.get('angAzimuth', 0)
            angPolar = self.initialConditionParams.get('angPolar', 0)
  
            oneSpinState = spinHalfState(angPolar, angAzimuth)
            quimbInitialState = quimb.tensor.MPS_product_state(\
                [oneSpinState] * self.nSpins)

        elif self.initialConditionParams.get('type') == 'BitString':
            bitstring = self.initialConditionParams.get('bitstring')
            quimbInitialState = quimb.tensor.MPS_computational_state(binary=bitstring)

        elif self.initialConditionParams.get('type') == 'RandomProduct':
            quimbInitialState = quimb.tensor.MPS_rand_state(L=self.nSpins, 
                bond_dim=1, phys_dim=2,)

        else:
            quimbInitialState = quimb.tensor.MPS_computational_state(binary='0' * self.nSpins)

        wallTimeEnd = time.time_ns()

        # Prepare a list to store rows of states
        self.statesRows = [{
            'uuidState': self.uuidInitialState,
            'uuidSimulation': self.uuidSimulation,
            'timeIndex': self.tIndexSoFar,
            'state': self.tebd.pt,
            'errEstimate': self.tebd.err,
            'wallTimeStart': wallTimeStart,
            'wallTimeEnd': wallTimeEnd,
            'wallTimeDuration': wallTimeEnd - wallTimeStart
        }]

        # Prepare a TEBD object
        self.tebd = quimb.tensor.tensor_1d.TEBD(quimbInitialState, 
                    localHam1d,
                    **quimbTEBDParams)


    def step(self):
        """Take a step
        """

        # Update the state
        wallTimeStart = time.time_ns()

        self.tebd.update_to((self.tIndexSoFar + 1) * self.solutionParams.get('tStep'), \
            order=self.solutionParams.get('order'))

        wallTimeEnd = time.time_ns()

        uuidState = uuidGen()
        
        statesRow = {
            'uuidState': uuidState,
            'uuidSimulation': self.uuidSimulation,
            'timeIndex': self.tIndexSoFar,
            'state': self.tebd.pt,
            'errEstimate': self.tebd.err,
            'wallTimeStart': wallTimeStart,
            'wallTimeEnd': wallTimeEnd,
            'wallTimeDuration': wallTimeEnd - wallTimeStart
        }

        self.statesRows += [statesRow]

        # Update time so far
        self.tIndexSoFar += 1

    def save_and_index(self):
        """
        """
        pass

    def evolve_to(self, t):
        """

        """
        pass
