import numpy as np
import time
import pickle
import os
import pandas
import itertools

import quimb
import quimb.tensor

import sys
sys.path.append('..')
sys.path.append('../Quimb/')

import quimbtebd

def calc_marginal_quimb (mps, locations):
    '''
    Calculates a marginal density matrix
    represented as a dense array
    from a quimb matrix product state
    represented using 
    `quimb.tensor.tensor_1d.MatrixProductState`
    
    Parameters
    ----------
    mps: matrix product state from which to calculate
        the marginal
    
    locations: locations to keep in marginal, all
        other locations are traced out.
        
    Returns
    -------
    marginal_rho: marginal density matrix
    
    '''
    marginal_mpo = mps.partial_trace(keep=locations)
    marginal_rho = marginal_mpo.to_dense()
    
    return marginal_rho


#### Date and time strings to be used for versioning
DATE_STR = time.strftime('%Y-%m-%d')
TIME_STR = time.strftime('%Y-%m-%d@%H:%M:%S')
print(DATE_STR)
print(TIME_STR)

#### Parameters
n_spins_values = [8, 10, 12, 14, 16, 18, 20, 22, 24]
n_spins_values = [32, 40, 56]
INITIAL_STATE = 'upy'

import simulationdictionary
inputs_dict = simulationdictionary.get_input_dicts()

#### Read MPS histories calculated using `quimb.tensor.tensor_1d.TEBD`
print('%s' % time.strftime('%Y-%m-%d %T'))
QUIMB_DATA_DIR = '../../Data/2020-March/Quimb/'
quimb_records_dict = {}

for n_spins in n_spins_values:
    quimb_records_dict[('transverse_fim', 'upy', n_spins)] = []
    for uuid_str in inputs_dict[('transverse_fim', 'upy', n_spins)]:
        inputfilename = os.path.join(QUIMB_DATA_DIR, uuid_str + '.pkl')

        print(inputfilename)
        with open(inputfilename, 'rb') as inputfile:
            record = pickle.load(inputfile)
            quimb_records_dict[('transverse_fim', 'upy', n_spins)] += [record]

    quimb_records_dict[('tilted_fim', 'upy', n_spins)] = []
    for uuid_str in inputs_dict[('tilted_fim', 'upy', n_spins)]:
        inputfilename = os.path.join(QUIMB_DATA_DIR, uuid_str + '.pkl')

        print(inputfilename)
        with open(inputfilename, 'rb') as inputfile:
            record = pickle.load(inputfile)
            quimb_records_dict[('tilted_fim', 'upy', n_spins)] += [record]

            
#### Arrange MPS histories by bond-dimension values used for calculating them
print('%s' % time.strftime('%Y-%m-%d %H:%M:%S'))
bonddim_values = list(range(1, 16+1))
n_bonddim_values = len(bonddim_values)

quimb_tebd_mps_array_dict = {}

for n_spins in n_spins_values:
    n_steps = quimb_records_dict[('tilted_fim', 'upy', n_spins)][0].n_steps
    
    quimb_tebd_mps_array_dict[('tilted_fim', 'upy', n_spins)] = np.empty((n_steps, n_bonddim_values), dtype=object)

    for ix_time in range(n_steps):
        for ix_bonddim in range(n_bonddim_values):
            quimb_tebd_mps_array_dict[('tilted_fim', 'upy', n_spins)][ix_time, ix_bonddim] = \
                quimb_records_dict[('tilted_fim', 'upy', n_spins)][ix_bonddim].states[ix_time]

    quimb_tebd_mps_array_dict[('transverse_fim', 'upy', n_spins)] = np.empty((n_steps, n_bonddim_values), dtype=object)

    for ix_time in range(n_steps):
        for ix_bonddim in range(n_bonddim_values):
            quimb_tebd_mps_array_dict[('transverse_fim', 'upy', n_spins)][ix_time, ix_bonddim] = \
                quimb_records_dict[('transverse_fim', 'upy', n_spins)][ix_bonddim].states[ix_time]
        
#### Functions to loop over approximations and time ####

def loop_timeseries_quimb (mps_timeseries, function, *arguments):
    '''
    Loops over a time series of matrix product
    states and evaluates a function
    
    Parameters
    ----------
    mps_timeseries: 
        time series of matrix product states
        
    function:
        function to apply on each element of
        the time series
        
    arguments:
        arguments to be supplied to the function
        
    Returns
    -------
    output_timeseries:
        timeseries of the calculation output 
        for each element of the time series of
        matrix product states
   
    '''
    
    n_steps = len(mps_timeseries)
    
    output_timeseries = np.empty_like(mps_timeseries, dtype=object)
    
    for ix_time in range(n_steps):
        mps = mps_timeseries[ix_time]
        
        output = function(mps, *arguments)
        
        output_timeseries[ix_time] = output
        
    return output_timeseries


def loop_approximations_quimb (mps_array, function, *arguments):
    '''
    Loops over a time series of matrix product
    states and evaluates a function
    
    Parameters
    ----------
    mps_array: 
        array of matrix product states
        indexed by [ix_time, ix_approx]
        
    function:
        function to apply on each element of
        the time series
        
    arguments:
        arguments to be supplied to the function
        
    Returns
    -------
    output_array:
        array of the calculation output 
        for each element of the time series of
        matrix product states
    '''
    
    n_timesteps, n_approximations = mps_array.shape
    
    output_array = np.empty_like(mps_array, dtype=object)
    
    for ix_approx in range(n_approximations):
        mps_timeseries = mps_array[:, ix_approx]
        
        output_timeseries = loop_timeseries_quimb(mps_timeseries, function, *arguments)
        
        output_array[:, ix_approx] = output_timeseries
        
    return output_array

######################################################################################

#### Calculate marginals
def calc_marginals (marginalsize_max):
    marginals_quimb_tebd_model_dict = {}

    for n_spins in n_spins_values:
        print('%s' % time.strftime('%Y-%m-%d %H:%M:%S'))
        print('%s spins' % n_spins)
        n_steps = quimb_records_dict[('tilted_fim', 'upy', n_spins)][0].n_steps

        locations_list = [tuple(range((n_spins-l+1)//2, (n_spins+l+1)//2)) for l in range (1, marginalsize_max+1)]
        for locations in locations_list:

            marginals_quimb_tebd_model_dict[('transverse_fim', 'upy', n_spins, locations)] = \
                loop_approximations_quimb(\
                    quimb_tebd_mps_array_dict[('transverse_fim', 'upy', n_spins)], 
                    calc_marginal_quimb,
                    *(locations,))

            marginals_quimb_tebd_model_dict[('tilted_fim', 'upy', n_spins, locations)] = \
                loop_approximations_quimb(\
                    quimb_tebd_mps_array_dict[('tilted_fim', 'upy', n_spins)], 
                    calc_marginal_quimb,
                    *(locations,))


    #### Save marginals
    MARGINAL_DIR = os.path.join(QUIMB_DATA_DIR, 'marginals')
    if not os.path.isdir(MARGINAL_DIR):
        os.makedirs(MARGINAL_DIR)

    marginal_txtout = os.path.join(MARGINAL_DIR, TIME_STR + '_marginals' + '.txt')
    with open(marginal_txtout, 'w') as txtout:
        txtout.write("marginalsize_max=%d\nn_spins_values=%s\ninitial=upy\nbondim_values=%s\n" % \
                     (marginalsize_max, str(n_spins_values), str(bonddim_values)))

    marginal_pklout = os.path.join(MARGINAL_DIR, TIME_STR + '_marginals' + '.pkl')
    with open(marginal_pklout, 'wb') as pklout:
        pickle.dump(marginals_quimb_tebd_model_dict, pklout)

    return marginals_quimb_tebd_model_dict

marginalsize_max = 4
print('%s' % time.strftime('%Y-%m-%d %H:%M:%S'))
calc_marginals(marginalsize_max)