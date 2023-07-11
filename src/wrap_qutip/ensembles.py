import numpy as np
import qutip

######################################################################################
class StateEnsemble:
    """Represents an ensemble of quantum states"""

    def __init__(self, states, weights=None):

        if weights == None:
            weights = [1] * len(states)
        else:
            assert len(weights) == len(states)

        self.weights = weights
        self.states = states

        self.num_states = len(self.states)
        self.weight_sum = sum(weights)

        self.probabilities = \
            [w/self.weight_sum for w in self.weights]

        self.flag_dirty = False

    def add_state(self, weight, state):
        """Adds state `state` with weight `weight`
        to the ensemble"""

        old_weight_sum = self.weight_sum
        new_weight_sum = old_weight_sum + weight

        old_probabilties = self.probabilities
        new_probabilities = \
            [p * old_weight_sum / new_weight_sum \
             for p in old_probabilties]

        new_probabilities.append(weight / new_weight_sum)

        self.weights.append(weight)
        self.states.append(state)
        self.weight_sum = new_weight_sum
        self.num_states += 1
        self.probabilities = new_probabilities

        self.flag_dirty = True


######################################################################################
class StateMomentCalculation:
    """Represents a calculation of moments of an ensemble of pure states"""

    def __init__(self, ensembles):

        self.ensembles = ensembles

    # Calculate a moment for each ensemble and take weighed average to calculate
    # the combined moment.

######################################################################################
class QutipQobjEnsemble(StateEnsemble):
    """Represents an ensemble of pure stateseach represented as `qutip.qobj.Qobj`"""

    def __init__(self, states, weights=None):
        StateEnsemble.__init__(self, states, weights)

        self.sample_values = {}
        self.moments = {}

    def calc_projectors(self):
        """Calculates the first moment"""
        self.projectors = []
        for s in self.states:
            if s.isket:
                proj = qutip.ket2dm(s)
            elif s.isoper and s.isherm:
                proj = s
            elif s.isbra:
                proj = qutip.ket2dm(s.dag())

            self.projectors.append(proj)

    def calc_moment(self, k):
        """Calculates `k`-th moment"""

        if not hasattr(self, "projectors") or self.flag_dirty:
            self.calc_projectors()

        if self.sample_values.get(k) == None or self.flag_dirty:

            if k == 1:
                self.sample_values[1] = self.projectors

            else:
                self.sample_values[k] = \
                    [qutip.tensor([proj for _ in range(k)]) \
                        for proj in self.projectors]

        if self.moments.get(k) == None  or self.flag_dirty:
            self.moments[k] = sum([prob * s for prob, s in \
                    zip(self.probabilities, self.sample_values[k])])

        return self.moments[k]

    def calc_covariance(self, sem=False):
        """Calculates the covariance, which is the difference
        between the second moment and two copies of the first
        moment"""

        rho2 = self.calc_moment(2)
        rho1 = self.calc_moment(1)

        rho_cov = rho2 - qutip.tensor(rho1, rho1)

        rho_cov_sq = rho_cov * rho_cov

        if not sem:
            return rho_cov_sq
        else:
            rho_sem_sq = rho_cov_sq \
            / (self.num_states * self.num_states - self.num_states)

            return rho_sem_sq

######################################################################################
class EmpiricalHaarEnsemble:
    """Represents an empirical Haar ensemble, obtained by sampling pure states
    from the Haar measure."""

    def __init__(self, size, dims, vector_dim=None):
        self.size = size
        self.dims = dims

        self.vector_dim = np.prod(dims[0])

        if vector_dim != None:
            assert vector_dim == self.vector_dim

        self.samples = []

    def gen_samples(self):

        for ix in range(len(self.samples), self.size):
            ket = qutip.rand_ket(self.vector_dim, dims=self.dims)
            self.samples.append(ket)
