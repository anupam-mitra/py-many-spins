import numpy as np
import qutip

def qutip_calc_manyspin_moment (state, ops, ls, n_spins):

    ls_unique = list(set(ls))
    duplicate_flag = (len(ls_unique) != len(ls))
    
    if duplicate_flag:
        raise NotImplementedError(\
            'Not implemented: more than one operator on a site %s' % ls)
    if len(ls) == n_spins:
        rho_marginal = state
    else:
        rho_marginal = qutip.ptrace(state, ls)
    op_expanded = qutip.tensor(*ops)
    moment = qutip.expect(op_expanded, rho_marginal) \
        / rho_marginal.tr()
    
    return moment

class QutipSpinMomentsCalculation:
    '''
    Calculates spin moments for a time series of matrix product 
    states

    Parameters
    ----------

    dynamics:
    An object representing the dynamics, which has a history of
    the state, in state representation, as it
    traverses the circuit

    n_spins:
    Number of spins in the dynamics

    term:
    Iterable of tuples to calculate from the state
    time series

    '''

    def __init__ (self, dynamics, n_spins, term):

        self.dynamics = dynamics
        self.n_spins = n_spins
        self.term = term
        self.n_steps = dynamics.n_steps
        
    def _calc_moments_se (self):
        self.moments = np.empty((self.n_steps,))
        
        for s in range(self.n_steps):
            psi = self.dynamics.states[s]
            moment = qutip_calc_manyspin_moment(psi, self.term[0], self.term[1], self.n_spins)
            self.moments[s] = np.real(moment)
    
    def _calc_moments_me (self):
        self.moments = np.empty((self.n_steps,))
        
        for s in range(self.n_steps):
            rho = self.dynamics.states[s]
            moment = qutip_calc_manyspin_moment(rho, self.term[0], self.term[1], self.n_spins)
            self.moments[s] = np.real(moment)
            
    def _calc_moments_mc (self):
        assert hasattr(self.dynamics, 'n_trajectories')
        self.n_trajectories = self.dynamics.n_trajectories
        self.moments = np.empty((self.n_trajectories, self.n_steps))
        
        for r in range(self.n_trajectories):
            for s in range(self.n_steps):
                psi = self.dynamics.states[r, s]
                moment = qutip_calc_manyspin_moment(psi, self.term[0], self.term[1], self.n_spins)
                self.moments[r, s] = np.real(moment)

        self.moments_mc_mean = np.mean(self.moments, axis=0)
        self.moments_mc_std = np.std(self.moments, axis=0)
        self.moments_mc_sterr = self.moments_mc_std / np.sqrt(self.n_trajectories)
        

    def calc_moments (self):
        '''
        Calculates the moments for all states in the timeseries 
        of states in the dynamics object
        '''

        if self.dynamics.is_se:
            print('_calc_moments_se')
            self._calc_moments_se()
        
        elif self.dynamics.is_me:
            print('_calc_moments_me')
            self._calc_moments_me()
            
        elif self.dynamics.is_mc:
            print('_calc_moments_mc')
            self._calc_moments_mc()

        
    def calc_moments_old (self):
        '''
        Calculates the moments for all states in the timeseries 
        of states in the dynamics object
        '''
        print(type(self.dynamics.states))
        
        if isinstance (self.dynamics.states, list):
            states_flat = self.dynamics.states
        elif len(np.shape(self.dynamics.states)) == 1:
             states_flat = self.dynamics.states
        else:
            states_flat = self.dynamics.states.flatten()
        moments_flat = np.zeros_like(states_flat, dtype=float)
        print(moments_flat.shape)
        print(len(states_flat))
        for ix in range(len(states_flat)):
            state = states_flat[ix]
            moments_flat[ix] = np.real(\
                qutip_calc_manyspin_moment(state, self.term[0], self.term[1], self.n_spins))
        
        if isinstance (self.dynamics.states, list):
            self.moments = moments_flat
        elif len(np.shape(self.dynamics.states)) == 1:
            self.moments = moments_flat
        else:
            self.moments = np.reshape(moments_flat, self.dynamics.states.shape)