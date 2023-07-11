import qutip
import numpy as np
import uuid

import manybodystateevolve
import spinsystems

class QutipSingleSimulationCalculator:
    '''
    Represents an object that calculates properties from
    a time series of states 
    '''
    
    def __init__ (self, dynamics):
        self.dynamics = dynamics
        self.t_list = dynamics.t_list
        self.n_steps = dynamics.n_steps
        
class QutipTwoSimulationCalculator:
    '''
    Represents an object that calculates properties
    from two time series of states. Typically this
    involves a comparison between these two time
    series.
    '''
    
    def __init__ (self, dynamics1, dynamics2):
        self.dynamics1 = dynamics1
        self.dynamics2 = dynamics2
        
        assert len(dynamics1.t_list) == len(dynamics2.t_list)
        self.n_steps = dynamics1.n_steps

        assert sum(abs(dynamics1.t_list - dynamics2.t_list)) < 1e-6
        self.t_list = dynamics1.t_list

class HilbertSchmidtDistanceCalculator (QutipTwoSimulationCalculator):
    '''
    Represents a calculation of Hilbert Schmidt distance
    '''
    
    def __init__ (self, dynamics1, dynamics2):
        QutipTwoSimulationCalculator.__init__(self, dynamics1, dynamics2)
        
        
    def calc_hs_distance(self):
        assert (self.dynamics1.is_se or self.dynamics1.is_me) and \
               (self.dynamics2.is_se or self.dynamics2.is_me)
        
        self.hsdistances = np.empty_like(self.t_list)
        
        for s in range(self.n_steps):
            state1 = self.dynamics1.states[s]
            state2 = self.dynamics2.states[s]
            
            self.hsdistances[s] = qutip.metrics.hilbert_dist(state1, state2)
             
class PurityCalculator (QutipSingleSimulationCalculator):
    '''
    Represents a calculation of purity
    '''
    
    def __init__ (self, dynamics):
        QutipSingleSimulationCalculator.__init__(self, dynamics)
        
    def calc_purity(self):
        assert (self.dynamics.is_se or self.dynamics.is_me)
        
        self.purities = np.empty_like(self.t_list)

        for s in range(self.n_steps):
            state = self.dynamics.states[s]

            self.purities[s] = state.purity()
