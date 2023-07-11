#
#
# 2020-09-15
# 2021-09-13

import quimb as qu
import quimb.tensor as qtn
import numpy as np

from numpy import sqrt, pi

def eval_single_trajectory(\
    psi_initial, h_nni, jump_ops, ts, \
    split_opts, epsilon_trotter, \
    random_numbers, debug=False):
    """
    Evaluates a single quantum trajectory

    Parameters
    ----------
        psi_initial: Initial state

        h_nni: Effective non Hermitian Hamiltonian, represented
            as an `NNI` object

        jump_ops: Sequence of jump operators

        ts: Times at which to save the state

        random_numbers: Random numbers to use for the simulation

        cutoff: Error budget for TEBD

        epsilon_trotter: Error budget for Trotterization


    Returns
    -------
        psi_ts: Sequence of `MatrixProductState` objects 
            as a function of time

        psi_normalized_ts: Sequence of normalized `MatrixProductState` 
            objects as a function of time

        t_jumps: Times at which jumps occurred

    """
    n_jump_ops = len(jump_ops)
    n_steps = len(ts)
    n_spins = psi_initial.nsites

    if debug:
        
        _debug_str = 'n_spins = %d, len(jump_ops) = %d' % (n_spins, n_jump_ops)

        print('DEBUG:\n\t', _debug_str)
    
    
    tebd = qtn.TEBD(psi_initial, h_nni, progbar=False, \
        split_opts=split_opts)

    psi_ts = [psi_initial]

    t_jumps = []
    
    count_random_used_norm = 0
    count_random_used_jump_choice = 0  
    
    dt_mcwf = ts[1] - ts[0]
    h_dk_ops = [qu.dag(jump_op) @ jump_op for jump_op in jump_ops]
    expm_h_dk_ops = [qu.expm(-1/2 * dt_mcwf * h_dk) for h_dk in h_dk_ops]
    
    for t_index in range(1, n_steps):

        t = ts[t_index]

        tebd.update_to(t, tol=epsilon_trotter)
        psit = list(tebd.at_times([t], tol=epsilon_trotter))[0]
        
        for j in range(n_jump_ops):
            for l in range(n_spins):
                psit = qtn.gate_TN_1D(psit, expm_h_dk_ops[j], l, contract=True)
                
        psi_ts += [psit]

        norm = psit.H @ psit
        
        
        # Check the jump condition
        if norm < random_numbers[count_random_used_norm]:
            

            
            t_jumps += [t]
            count_random_used_norm += 1

            # Calculate the probability distribution of jumps
            prob_jumps = np.asarray([\
                        np.abs(psit.H @ qtn.gate_TN_1D(psit, qu.dag(jump_op) @ jump_op, l))
                          for jump_op in jump_ops for l in range(n_spins)])
            
            prob_jumps /= np.sum(prob_jumps)
            
            if debug:
                _debug_str = 'norm = %g' % norm
                _debug_str += 'len(prob_jumps) = %d,'  % \
                (len(prob_jumps))

                print('DEBUG: Performing a jump \n\t', _debug_str)
                
            # Calculate the cumulative probability distribution of jumps
            prob_cum_jumps = np.cumsum(prob_jumps)
            
            index_jump = np.min(np.where(\
                prob_cum_jumps > random_numbers[n_steps + count_random_used_jump_choice])[0])
            
            index_jump_op = index_jump % n_jump_ops
            loc_jump = index_jump // n_jump_ops
            
            if debug:
                _debug_str = 'index_jump = %d, index_jump_op = %d, loc_jump = %d' % \
                    (index_jump, index_jump_op, loc_jump)
                print('\t', _debug_str)

            psit_jumped = qtn.gate_TN_1D(psit, jump_ops[index_jump_op], loc_jump, contract=True)

            count_random_used_jump_choice += 1
            psit_jumped /= np.sqrt(psit_jumped.H @ psit_jumped)

            # Create a new TEBD object
            tebd = qtn.TEBD(psit_jumped, h_nni, t0=t_jumps[-1], progbar=False, \
                split_opts=split_opts)
            
        else:
            tebd = qtn.TEBD(psit, h_nni, t0=t, progbar=False, \
                split_opts=split_opts)
            
    psi_normalized_ts = [psit / np.sqrt(psit.H @ psit) for psit in psi_ts]
    
    return psi_ts, psi_normalized_ts, t_jumps


# '''
# Reading and writing the results
# '''
# import os
# import pickle
# def read_results(string_uuid, datadir='../../Data'):
#     input_file = open(os.path.join(datadir, string_uuid + '.pkl'), 'rb')
#     result_read = pickle.load(input_file)
    
#     input_file.close()
#     return result_read


# # read the result 66d4bc3c-cd35-11ea-80ba-a860b63445d4
# read_dict = read_results('66d4bc3c-cd35-11ea-80ba-a860b63445d4_mcwf')
# read_dict.keys()

# psi_ts = read_dict['psi_ts']
# t_jumps_traj = read_dict['t_jumps_traj']