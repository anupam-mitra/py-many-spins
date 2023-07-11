import itertools
import numpy as np

################################################################################
class InteractionGraph:
    '''
    Represents an interaction graph for a model with two body interactions
    '''

    def __init__ (self, n_sites):
        self.n_sites = n_sites
################################################################################

################################################################################
class Interaction1DNearest (InteractionGraph):
    """Represents an interaction graph for a model with two body interaction
    """

    def __init__(self, n_sites, bc='periodic'):
        self.n_sites = n_sites
        self.bc = bc

    def get_edges (self):
        '''
        Returns edges
        '''

        if not hasattr(self, "edges"):
            self.edges = [(l, l+1) for l in range(self.n_sites-1)]
            if self.bc == 'periodic':
                self.edges += [(self.n_sites-1, 0)]

        return self.edges

    def get_edge (self, l1, l2):

        if (l1, l2) in self.edges:
            return 1
        else:
            return 0
################################################################################

################################################################################
class Interaction1DNextNearest (InteractionGraph):
    """Represents an interaction graph for a model with two body interaction
    """

    def __init__(self, n_sites, bc='periodic'):
        self.n_sites = n_sites
        self.bc = bc

    def get_edges (self):
        '''
        Returns edges
        '''

        if not hasattr(self, "edges"):
            self.edges = [(l, l+1) for l in range(self.n_sites-1)]
            self.edges += [(l, l+2) for l in range(self.n_sites-2)]
            if self.bc == 'periodic':
                self.edges += [(self.n_sites-1, 0)]
                self.edges += [(self.n_sites-1, 1)]
                self.edges += [(self.n_sites-2, 0)]

        return self.edges

    def get_edge (self, l1, l2):

        if (l1, l2) in self.edges:
            return 1
        else:
            return 0
################################################################################

################################################################################
class Interaction1DPowerLaw (InteractionGraph):
    """Represents an interaction graph for a model with two body interaction
    with power law
    """

    def __init__(self, n_sites, alpha, bc='periodic'):
        self.n_sites = n_sites
        self.alpha = alpha
        self.bc = bc

    def get_edges (self):
        '''
        Returns edges
        '''

        if not hasattr(self, "edges"):
            self.edges = list(itertools.combinations(range(self.n_sites), 2))
        return self.edges

    def _calc_weights (self):
        self.weights = {}
        self.weights_norm_factor = 0

        for (l1, l2) in self.edges:
            w = np.abs(float(l1 - l2))**(-self.alpha)
            self.weights[(l1, l2)] = w
            self.weights_norm_factor += w

        self.weights_norm_factor = np.sum([self.weights[e] for e in self.edges]) / (self.n_sites - 1)

        #for (l1, l2) in self.edges:
        #    self.weights[(l1, l2)] /= self.weights_norm_factor


    def get_edge (self, l1, l2, normalize=True):
        if not hasattr(self, "weights"):
            self._calc_weights()

        if (l1, l2) in self.edges:
            if normalize:
                return self.weights[(l1, l2)] / self.weights_norm_factor
            else:
                return self.weights[(l1, l2)] 
################################################################################

################################################################################
