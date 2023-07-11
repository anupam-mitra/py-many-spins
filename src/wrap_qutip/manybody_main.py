import qutip
import numpy as np
import uuid

import manybodystateevolve
import spinsystems

import logging

class QutipManySpinSimulation:
    '''
    Represents an instance of a many spin
    dynamics simulation
    
    '''

    def __init__(self, n_spins, interact_graph, \
                 jxx, jyy, jzz, bx, by, bz, \
                 gammax, gammay, gammaz, gammaplus, gammaminus, \
                 angle_polar, angle_azimuthal, \
                 t_initial, t_final, n_steps, \
                 SOLVE_SE=True, SOLVE_ME=True, SOLVE_MC=True,
                 n_traj=None
                ):
        
        # Identifier
        self.uuid_str = str(uuid.uuid4())
        
        
        # System parameters
        self.n_spins = n_spins
        
        # Hamiltonian model parameters
        self.interact_graph = interact_graph
        
        self.jzz = jzz
        self.jyy = jyy
        self.jxx = jxx
        
        self.bx = bx
        self.by = by
        self.bz = bz
        
        # Decoherence model parameters
        self.gammax = gammax
        self.gammay = gammay
        self.gammaz = gammaz
        
        self.gammaplus = gammaplus
        self.gammaminus = gammaminus
        
        self.FLAG_CLOSED_SYSTEM = \
            ((self.gammax == None) or (np.abs(gammax) < 1e-6)) and \
            ((self.gammay == None) or (np.abs(gammay) < 1e-6)) and \
            ((self.gammaz == None) or (np.abs(gammaz) < 1e-6)) and \
            ((self.gammaminus == None) or (np.abs(gammaplus) < 1e-6)) and \
            ((self.gammaplus == None) or (np.abs(gammaminus) < 1e-6)) 
        
        # Solution methods
        self.FLAG_SOLVE_SE = SOLVE_SE and self.FLAG_CLOSED_SYSTEM
        self.FLAG_SOLVE_ME = SOLVE_ME and (not self.FLAG_CLOSED_SYSTEM)
        self.FLAG_SOLVE_MC = SOLVE_MC and (not self.FLAG_CLOSED_SYSTEM)
        
        # Solution parameters
        self.n_steps = n_steps
        
        if self.FLAG_SOLVE_MC:
            if n_traj != None:
                self.n_traj = n_traj
            else:
                self.n_traj = 223
        
        # Initial condition parameters
        self.angle_polar = angle_polar
        self.angle_azimuthal = angle_azimuthal
        
        # Duration parameters
        self.t_initial = t_initial
        self.t_final = t_final
        
    def __str__(self):
        '''
        Returns a string representation
        
        WARNING: Only TFIM with Jzz != 0 and Bx !=0 are supported at present
        '''
        
        if hasattr(self, 'description'):
            return self.description

        description = self.uuid_str + '\n'

        if abs(self.angle_polar) < 1e-6:
            description += 'init-z; '
        elif abs(self.angle_polar - pi/2) < 1e-6:
            if abs(self.angle_azimuthal) < 1e-6:
                description += 'init-x; ' 
            elif  abs(self.angle_azimuthal -3*pi/2) < 1e-6:
                     description += 'init-x; '

        if isinstance(self.interact_graph, manybodystateevolve.Interaction1DNearest):
            description += 'nn; '
        elif isinstance(self.interact_graph, manybodystateevolve.Interaction1DNextNearest):
            description += 'nnn; '
        elif isinstance(self.interact_graph, manybodystateevolve.Interaction1DPowerLaw):
            description += r'power-law, $\alpha = %g$; ' % (self.interact_graph.alpha)

        if self.interact_graph.bc == 'open':
            description +=  '1d obc; '
        elif self.interact_graph.bc == 'periodic':
            description +=  '1d pbc; '

        description += r'$B_x/J_{zz}$ = %g; ' % (self.bx/self.jzz)
        description += r'$\Gamma/J_{zz}$ = %g' % (self.gammaz)
                   
        return description

    def _init_initial_state (self):
        '''
        Initializes the initial state
        '''
        self.initial_ket_onebody = qutip.spin_coherent(\
                                    j=1/2, theta=self.angle_polar, phi=self.angle_azimuthal, type='ket')
        self.initial_ket = qutip.tensor([self.initial_ket_onebody] * self.n_spins)
        
        
    def _init_dynamics_generators (self):
        '''
        Initialize the operators that generate the dynamic
        '''
        
        self.interact_energies = []
        self.interact_ops = []
        self.local_energies = []
        self.local_ops = []
        
        if self.jxx != None:
            self.interact_energies += [self.jxx]
            self.interact_ops += [(qutip.sigmax(), qutip.sigmax())]
        
        if self.jyy != None:
            self.interact_energies += [self.jyy]
            self.interact_ops += [(qutip.sigmay(), qutip.sigmay())]

        if self.jzz != None:
            self.interact_energies += [self.jzz]
            self.interact_ops += [(qutip.sigmaz(), qutip.sigmaz())]
            
        if self.bx != None:
            self.local_energies += [self.bx]
            self.local_ops += [qutip.sigmax()]
            
        if self.by != None:
            self.local_energies += [self.bx]
            self.local_ops += [qutip.sigmay()]

        if self.bz != None:
            self.local_energies += [self.bx]
            self.local_ops += [qutip.sigmaz()]
            
        self.hamilonian_model = \
            spinsystems.UniformTwoBodyInteraction(\
                self.interact_ops, \
                self.local_ops, \
                self.interact_energies, self.local_energies, \
                self.interact_graph)
        
        self.decay_rates = []
        self.decay_ops = []
                    
        if self.gammax != None:
            self.decay_rates += [self.gammax]
            self.decay_ops += [qutip.sigmax()]

        if self.gammay != None:
            self.decay_rates += [self.gammay]
            self.decay_ops += [qutip.sigmay()]

        if self.gammaz != None:
            self.decay_rates += [self.gammaz]
            self.decay_ops += [qutip.sigmaz()]

        if self.gammaplus != None:
            self.decay_rates += [self.gammaplus]
            self.decay_ops += [qutip.sigmap()]

        if self.gammaplus != None:
            self.decay_rates += [self.gammaminus]
            self.decay_ops += [qutip.sigmam()]
  

        self.decoherence_model = \
            spinsystems.UniformOneBodyDecoherence(\
                self.decay_ops, self.decay_rates)
        
    def calc_state_dynamics(self):
        if not hasattr(self, 'hamiltonian_model'):
            self._init_dynamics_generators()
            
        if not hasattr(self, 'initial_ket'):
            self._init_initial_state()
            
        if self.FLAG_SOLVE_SE:
            self.sesolver = manybodystateevolve.QutipSESolve(\
                                self.t_initial, self.t_final, self.n_steps, \
                                self.hamilonian_model, \
                                self.n_spins, self.initial_ket)
            self.sesolver.run()
            
        if self.FLAG_SOLVE_MC:
            self.mcsolver = manybodystateevolve.QutipMCSolve(\
                                self.t_initial, self.t_final, self.n_steps, \
                                self.hamilonian_model, self.decoherence_model, \
                                self.n_spins, self.initial_ket, self.n_traj)
            self.mcsolver.run()

        if self.FLAG_SOLVE_ME:
            self.initial_dm = qutip.ket2dm(self.initial_ket)
            self.mesolver = manybodystateevolve.QutipMESolve(\
                                self.t_initial, self.t_final, self.n_steps, \
                                self.hamilonian_model, self.decoherence_model, \
                                self.n_spins, self.initial_dm)
            self.mesolver.run()
            
    def get_dynamics(self):
        if self.FLAG_SOLVE_SE:
            return self.sesolver
        elif self.FLAG_SOLVE_MC:
            return self.mcsolver
        elif self.FLAG_SOLVE_ME:
            return self.mesolver
        else:
            return None
    
    
if __name__ == '__main__':
    import time
    import itertools
    import os
    import pandas
    import pickle
    
    from numpy import pi
        
    # System parameters
    n_spins = 4
    
    # Data saving
    BASE_DIR = '2021-APS-MarchMeet/pkl/'
    
    DATA_DIR = os.path.join(BASE_DIR, '%02d_spin_obc' % (n_spins,))
    
    STATES_DIR = os.path.join(DATA_DIR, 'states')
    TRUNC_DIR = os.path.join(DATA_DIR, 'trunc')
    MOMENTS_DIR = os.path.join(DATA_DIR, 'moments')
    PLOT_DIR = os.path.join(DATA_DIR, 'plots')
    
    for d in [BASE_DIR, PLOT_DIR, DATA_DIR, STATES_DIR, MOMENTS_DIR, TRUNC_DIR]:    
        if not os.path.isdir(d):
            os.makedirs(d)

    INDEX_FILENAME = os.path.join(DATA_DIR, 'index.pkl')

    # Solution parameters
    n_steps = 128
    n_traj = 101

    # Hamiltonian model parameters
    jzz = 1.0
    bx_values = jzz * np.asarray([1/4, 1/2, 2/3, 4/5, 9/10, 1.0, 11/10, 5/4, 3/2, 2, 4])
    bx_values = jzz * np.asarray([1.0])

    # Decoherence model parameters
    gamma_values = jzz * np.asarray([0, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01])

    # Initial condition parameters
    angle_polar_values = [0, pi/2]
    angle_azimuthal = 0

    # Duration parameters
    t_initial = 0
    t_final = n_spins * 2 / jzz

    SOLVE_SE = True
    SOLVE_ME = False #(n_spins <= 8)
    SOLVE_MC = not SOLVE_ME
    
    alpha = 1.5
    interact_graphs =[\
        manybodystateevolve.Interaction1DNearest(n_spins, bc='open'),
        manybodystateevolve.Interaction1DNextNearest(n_spins, bc='open'),
        manybodystateevolve.Interaction1DPowerLaw(n_spins, alpha=alpha, bc='open')]
    
    rows = []
    
    logging.info('START at %s'% (time.strftime('%Y-%m-%d %H:%M:%S'),))
    for bx, gamma, theta, graph in itertools.product(bx_values, gamma_values, angle_polar_values, interact_graphs):
        
        logging.info('%s'% (time.strftime('%Y-%m-%d %H:%M:%S'),))
        
        simulator = QutipManySpinSimulation(\
                    n_spins, graph, \
                    jxx=0, jyy=0, jzz=jzz, bx=bx, by=0, bz=0, \
                    gammax=gamma, gammay=gamma, gammaz=gamma, gammaplus=0, gammaminus=0, \
                    angle_polar=theta, angle_azimuthal=angle_azimuthal, \
                    SOLVE_MC=SOLVE_MC, SOLVE_ME=SOLVE_ME, SOLVE_SE=SOLVE_SE, \
                    t_initial=t_initial, t_final=t_final, n_steps=n_steps)
                   
        logging.info('START %s' % str(simulator))
                   
        simulator.calc_state_dynamics()
                   
        logging.info('END %s' % str(simulator))
        
        uuid_str = simulator.uuid_str
        
        savefilename = os.path.join(STATES_DIR, uuid_str + '.pkl')
        
        with open(savefilename, 'wb') as outputfile:
            pickle.dump(simulator, outputfile)
            
        rows += [{
                  'n_spins': n_spins,
                  'jzz': jzz,
                  'bx': bx,
                  'gamma': gamma,
                  'angle_polar_initial': theta,
                  'description': str(simulator),
                  'uuid_str': simulator.uuid_str
                 }]
    
    logging.info('END at %s'% (time.strftime('%Y-%m-%d %H:%M:%S'),))
    df_index = pandas.DataFrame(rows)
    with open(INDEX_FILENAME, 'wb') as outputfile:
        pickle.dump(df_index, outputfile)
        

