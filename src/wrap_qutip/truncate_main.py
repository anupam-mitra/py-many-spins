import qutip
import numpy as np
import uuid

import manybodystateevolve
import spinsystems
import truncate

from manybody_main import QutipManySpinSimulation

def read_simulation (uuid_str, states_dir):
    inputfilename = os.path.join(states_dir, uuid_str + '.pkl')

    with open(inputfilename, 'rb') as inputfile:
        simulator = pickle.load(inputfile)

    return simulator

def save_truncation(truncator, uuid_str, trunc_dir):
    outputfilename = os.path.join(trunc_dir, uuid_str + '.pkl')

    with open(outputfilename, 'wb') as outputfile:
        pickle.dump(truncator, outputfile)

def save_index_pkl(index, index_filename, index_dir):
    outputfilename = os.path.join(index_dir, index_filename + '.pkl')

    with open(outputfilename, 'wb') as outputfile:
        pickle.dump(index, outputfile)



class QutipSimulationTruncation:
           
    
    def __init__ (self, uuid_str_list, bonddim_values, states_dir, trunc_dir, index_dir):
        self.uuid_str_list = uuid_str_list
        self.bonddim_values = bonddim_values
        
        self.states_dir = states_dir
        self.trunc_dir = trunc_dir
        self.index_dir = index_dir
        
        self.index_filename = 'index_trunc'
        
    def calc_truncations (self):
        '''
        Truncates the dynamics in `self.uuid_str_list`. Each 
        truncated instance creates a `QuantumDynamicsStateEvolution` 
        object
        
        '''
        self.index_rows = []
        
        print('%s: START ' % (time.strftime('%Y-%m-%d %H:%M:%S'),))
        for uuid_str_dynamics in self.uuid_str_list:
            print('Found file', uuid_str_dynamics, \
                  os.path.isfile(os.path.join(self.states_dir, uuid_str_dynamics + '.pkl')))

        print('%s: START ' % (time.strftime('%Y-%m-%d %H:%M:%S'),))
        for uuid_str_dynamics in self.uuid_str_list:
            
            uuid_str_trunc = str(uuid.uuid4())
        
            print('%s: LOAD %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), uuid_str_dynamics))
            
            simulator = read_simulation(uuid_str_dynamics, self.states_dir)
            n_spins = simulator.n_spins
            dynamics = simulator.get_dynamics()
            
            for bonddim in self.bonddim_values:
                print('%s: START truncate %s to %d' \
                      % (time.strftime('%Y-%m-%d %H:%M:%S'), uuid_str_dynamics, bonddim))
            
                if bonddim == (2**n_spins//2):
                    dynamics_trunc = self.dynamics
                else:
                    truncator = truncate.TruncationCalculator(\
                        dynamics, n_spins, bonddim)
                    dynamics_trunc = truncator.calc_trunc_states()
                    
                print('%s: FINISHED truncate %s to %d' \
                      % (time.strftime('%Y-%m-%d %H:%M:%S'), uuid_str_dynamics, bonddim))

                save_truncation(truncator, uuid_str_trunc, self.trunc_dir)
                self.index_rows += [{
                    'uuid_str_dynamics': uuid_str_dynamics,
                    'bonddim': bonddim,
                    'uuid_str_trunc': uuid_str_trunc
                }]
                
        print('%s: SAVE index' % (time.strftime('%Y-%m-%d %H:%M:%S'),))                
            
        self.index_to_df_pkl()
        
        print('%s: START ' % (time.strftime('%Y-%m-%d %H:%M:%S'),))
        
    def index_to_df_pkl (self):
        
        self.index_df = pandas.DataFrame(data=self.index_rows)
        save_index_pkl(self.index_df, self.index_filename, self.index_dir)
        
                
            
        
if __name__ == '__main__':
    
    import time
    import itertools
    import os
    import pandas
    import pickle

    from numpy import pi

    # System parameters
    n_spins = 4
    
    bonddim_values = np.logspace(0, n_spins//2, base=2, num=n_spins//2+1)

    # Data saving
    BASE_DIR = './2021-APS-MarchMeet/pkl/'

    DATA_DIR = os.path.join(BASE_DIR, '%02d_spin_obc' % (n_spins,))

    STATES_DIR = os.path.join(DATA_DIR, 'states')
    TRUNC_DIR = os.path.join(DATA_DIR, 'trunc')
    MOMENTS_DIR = os.path.join(DATA_DIR, 'moments')
    PLOT_DIR = os.path.join(DATA_DIR, 'plots')
    
    for d in [BASE_DIR, PLOT_DIR, DATA_DIR, STATES_DIR, MOMENTS_DIR, TRUNC_DIR]:    
        if not os.path.isdir(d):
            os.makedirs(d)

    INDEX_FILENAME = os.path.join(DATA_DIR, 'index.pkl')
    with open(INDEX_FILENAME, 'rb') as inputfile:
        df_index = pickle.load(inputfile)

    uuid_str_list = [s[:36] for s in df_index.uuid_str]
    
    truncation_manager = QutipSimulationTruncation(\
            uuid_str_list, bonddim_values, STATES_DIR, TRUNC_DIR, DATA_DIR)    
    
    truncation_manager.calc_truncations()
    
    