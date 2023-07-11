import itertools
import pickle
import qutip

import compositesystem

import numpy as np

################################################################################
@DeprecationWarning
def truncateDensityMatrix (rho, threshold):
    '''
    Truncates a density matrix `rho` based on a 
    threshold set on its eigenvalues
    '''
    if threshold == 0:
        return rho

    # Get the eigenvalues and eigenvectors of the density matrix
    rhoEValues, rhoEVectors = rho.eigenstates()

    # Select the eigenvectors and eigenvalues based on the decimattion threshold
    indices_removed = np.where(rhoEValues < threshold)[0]

    # Iterate over the eigenvalues and eigenvectors to remove
    rhoTrunc = rho
    for index in indices_removed:
        evalue = rhoEValues[index]
        ket = rhoEVectors[index]
        bra = ket.dag()

        rhoTrunc = rhoTrunc - (evalue * ket * bra)

    # Renormalize the density matrix
    rhoTrunc = rhoTrunc / rhoTrunc.tr()
    
    return rhoTrunc
################################################################################

class ReducedDensityMatrixCalculation:
    '''
    Represents calculations of reduced density matrices of a many body state

    What is a good way to index these marginals?

    rho:
        Many body density matrix

    nDof:
        Number of degrees of freedom in the many body density matrix

    locations:
        Locations at which to calculate the reduced density matrices
    '''
    def __init__ (self,
        rho,
        nDof,
        locations=None,):

        self.rho = rho
        self.nDof = nDof

        if locations != None:
            self.locations = locations
        else:
            self.locations = {n: list(itertools.combinations(range(nDof), n)) \
                for n in range(1, nDof)}

        self.rhoMarginals = {}
        self.entropyMarginals = {}

    def calcMarginals (self):
        '''
        Calculates the marginals
        '''

        for nDofMoment in self.locations.keys():
            if self.rhoMarginals.get(nDofMoment) == None:
                self.rhoMarginals[nDofMoment] = {}

            for locationTuple in self.locations[nDofMoment]:
                if self.rhoMarginals.get(locationTuple) == None:
                    rhoMarginalCurrent = self.rho.ptrace(locationTuple)
                    self.rhoMarginals[nDofMoment][locationTuple] = \
                        rhoMarginalCurrent

    def calcMarginalEntropies (self):
        '''
        Calculates the entropies of the marginals
        '''
        for nDofMoment in self.locations.keys():
            if self.entropyMarginals.get(nDofMoment) == None:
                self.entropyMarginals[nDofMoment] = {}

            for locationTuple in self.locations[nDofMoment]:
                if self.entropyMarginals.get(locationTuple) == None:
                    rhoMarginalCurrent = \
                        self.rhoMarginals[nDofMoment][locationTuple] 
                    entropyMarginalCurrent = qutip.entropy_vn(\
                        rhoMarginalCurrent, base=2)
                    self.entropyMarginals[nDofMoment][locationTuple] = \
                        entropyMarginalCurrent

################################################################################
class DensityMatrixTruncations:
    '''
    Represents a trunctions of marginal density matrices

    marginal:
        Object which contains the calculations for marginals

    thresholds:
        Iteratable containing the list of thresholds

    '''
    def __init__ (self, rho, thresholds):
        self.rho = rho
        self.thresholds = thresholds
        self.rhoTruncatedList = []
        self.fidelities = np.zeros_like(thresholds)


    def calcTruncatedMatrices (self):
        for th in self.thresholds:
            rhoTruncated = truncateDensityMatrix(\
                self.rho, th)
            self.rhoTruncatedList.append(rhoTruncated)

    def calcFidelities (self):
        for i in range(len(self.thresholds)):
            self.fidelities[i] = qutip.fidelity(self.rho, self.rhoTruncatedList[i])
        
################################################################################
class SpinHalfMarginalCalculations:
    '''
    Represents a calculation of marginal spin 1/2 density matrices with `n_spins`

    '''

    def __init__ (self, \
                  nSpins,
                  states,
                  uuidSimulation, \
                  uuidApproximationList=None, \
                  thresholdEigenvaluesList=None):
        self.nSpins = nSpins
        self.uuidSimulation = uuidSimulation

        if isinstance(states, np.ndarray):
            (self.nTraj, self.nSteps) = states.shape
            self.states = states
        elif isinstance(states, list):
            self.nTraj = 1
            self.nSteps = len(states)
            self.states = np.asarray([states], dtype=object)

        if uuidApproximationList != None:
            self.uuidApproximationList = uuidApproximationList
            self.thresholdEigenvaluesList = thresholdEigenvaluesList
        else:
            self.uuidApproximationList = ['NoApprox']
            self.thresholdEigenvaluesList = [0]

        self.columns1SpinMarginals = (\
            'uuidSimulation',
            'uuidApproximation',
            'trajectoryIndex',
            'timeIndex',
            'location',	
            'rho',
            'entropyMarginal',
            'xExpect', 
            'yExpect', 
            'zExpect')

        self.columns2SpinMarginals = (\
            'uuidSimulation',
            'uuidApproximation',
            'trajectoryIndex',
            'timeIndex',
            'location1',
            'location2',
            'rho',
            'entropyMarginal',
            'xxExpect', 
            'yyExpect', 
            'zzExpect')

    def calc1SpinMarginalsQutip (self, locations=None):
        '''
        Calculates 1 spin marginals

        '''

        xOp = qutip.sigmax()
        yOp = qutip.sigmay()
        zOp = qutip.sigmaz()

        self.rows1SpinMarginals = []

        if locations == None:
            locations = range(self.nSpins)

        for approxIndex, tr, st, in \
            itertools.product(range(len(self.uuidApproximationList)), \
                range(self.nTraj), range(self.nSteps)):

            state = self.states[tr, st]

            for l in locations:
                # Marginal density matrix
                rhoMarginal = state.ptrace([l,])

                # Entropy of the marginal density matrix
                entropyMarginal = qutip.entropy_vn(rhoMarginal, base=2)

                # Serialized marginal density matrix
                #rhoMarginalBytes = pickle.dumps(rhoMarginal)

                # 1-spin expected values
                xExpect = qutip.expect(xOp, rhoMarginal)
                yExpect = qutip.expect(yOp, rhoMarginal)
                zExpect = qutip.expect(zOp, rhoMarginal)

                self.rows1SpinMarginals += [(\
                    self.uuidSimulation, \
                    self.uuidApproximationList[approxIndex], \
                    tr, st, tr, \
                    None, #rhoMarginalBytes, \
                    entropyMarginal, \
                    xExpect, yExpect, zExpect, \
                    )]

    def calc2SpinMarginalsQutip (self, locations=None):
        '''
        Calculates 2 spin marginals

        '''
        xxOp = qutip.tensor(qutip.sigmax(), qutip.sigmax())
        yyOp = qutip.tensor(qutip.sigmay(), qutip.sigmay())
        zzOp = qutip.tensor(qutip.sigmaz(), qutip.sigmaz())

        if locations == None:
            locations = itertools.combinations(range(self.nSpins), 2)

        self.rows2SpinMarginals = []

        for approxIndex, tr, st, in \
            itertools.product(range(len(self.uuidApproximationList)), \
                range(self.nTraj), range(self.nSteps)):

            state = self.states[tr, st]

            for (l1, l2) in locations:
                 
                # Marginal density matrix
                rhoMarginal = state.ptrace([l1, l2])

                # Entropy of the marginal density matrix
                entropyMarginal = qutip.entropy_vn(rhoMarginal, base=2)

                # Serialized marginal density matrix
                #rhoMarginalBytes = pickle.dumps(rhoMarginal)

                # 1-spin expected values
                xxExpect = qutip.expect(xxOp, rhoMarginal)
                yyExpect = qutip.expect(yyOp, rhoMarginal)
                zzExpect = qutip.expect(zzOp, rhoMarginal)
                
                self.rows2SpinMarginals += [(\
                    self.uuidSimulation, \
                    self.uuidApproximationList[approxIndex], \
                    tr, st, l1, l2, \
                    None, #rhoMarginalBytes, \
                    entropyMarginal, \
                    xxExpect, yyExpect, zzExpect, \
                    )]


################################################################################

