import quimb as qu
import quimb.tensor as qtn

import numpy as np
import pickle
import uuid
import os
import time

import matplotlib

from numpy import pi


import sys
sys.path.append('../')
sys.path
import QuimbTensor1DResult

############################################################################################
def make_tfim_hamiltonian(j_int, bx, S=1/2):
    sigmax = qu.pauli('X')
    sigmay = qu.pauli('Y')
    sigmaz = qu.pauli('Z')

    local_energies = [bx]
    local_ops = [sigmax]

    interact_energies = [j_int]
    interact_ops = [sigmaz]

    builder = qtn.SpinHam(S=S)

    for ll in range(len(local_ops)):
        builder.add_term(local_energies[ll], local_ops[ll])

    for ll in range(len(interact_ops)):
        builder.add_term(interact_energies[ll], interact_ops[ll], interact_ops[ll])

    h = builder.build_nni(n_spins)
    h = qtn.NNI_ham_ising(n_spins, j=j_int, bx=bx, cyclic=False)

    return h
############################################################################################

############################################################################################
def timeevolve_mps (psi_in, h, tlist, epsilon_trotter=1e-6, \
    cutoff=None, max_bond=None, title_string=None):
    psi_ts = [] # states

    tebd = qtn.TEBD(psi_in, h)

    if cutoff != None:
        tebd.split_opts['cutoff'] = cutoff

    if max_bond != None:
        tebd.split_opts['max_bond'] = max_bond

    # generate the state at each time in tlist
    timestamp_start = time.time()
    for psit in tebd.at_times(tlist * np.pi, tol=epsilon_trotter):
        psit.compress()

        psi_ts += [psit]
    timestamp_end = time.time()

    print('%s: took %g seconds' % (title_string, (timestamp_end - timestamp_start),))
    
    return psi_ts, tebd
############################################################################################

############################################################################################
def save_mps (filename, psi_in, psi_fin, t_in, t_fin, model, params):
    '''
    Save a matrix product state to disk, creating and index entry.

    Parameters
    ----------

    filename: File name

    psi_in: Initial state

    psi_fin: Final state

    model: Model used to propagate the state

    t_in: Initial time

    t_fin: Final time

    params:

    Returns
    -------

    '''
    data = {\
        'psi_in': psi_in, \
        'psi_fin': psi_fin, \
        't_in': t_in, \
        't_fin': t_fin, \
        'model': model, \
        'parameters': params
    }

    with open(filename, 'wb') as f:
        pickle.dump(data, f)
############################################################################################

############################################################################################
def load_mps():
    """
    Loads a matrix product state from a file
    """

    pass
############################################################################################

############################################################################################
def calc_manyspin_moment (state, ops, ls, n_spins=None):
    
    ls_unique = list(set(ls))
    duplicate_flag = (len(ls_unique) != len(ls))
    
    if not duplicate_flag:
        moment = state.H @ qtn.gate_TN_1D(state, qu.kron(*ops), np.asarray(ls))
    else:
        moment = float("nan")
    return moment
############################################################################################

############################################################################################

if __name__ == '__main__':
 
    n_spins = 16
    j_int = 1
    bx = j_int * 1

    sigmaz = qu.pauli('Z')

    string_uuid = uuid.uuid1()

    n_steps = 256
    t_initial = 0

    t_final = n_spins / j_int  * pi * 2
    #time_label_string = r'$ J t / \pi$'

    tlist = np.linspace(t_initial, t_final, n_steps)

    zeros_all = '0' * n_spins
    print('psi_in:', f"|{zeros_all}>")
 
    psi_in = qtn.MPS_computational_state(zeros_all)
    psi_in.show()

    h = make_tfim_hamiltonian(j_int, bx)

    bond_dim_vals = np.logspace(0, n_spins//2, num=n_spins//2+1, base=2)
    mps_histories_bd = np.empty_like(bond_dim_vals, dtype=object)

    innerproducts = np.zeros((len(bond_dim_vals), n_steps))
    zz_corr = np.zeros((len(bond_dim_vals), n_steps))

    dirname = '../_data_/QuimbMPS/B=%gJ/' % (bx/j_int)

    print('Bond dim values = ', bond_dim_vals)

    if not os.path.isdir(dirname):
        print('Directory not found, creating')
        os.makedirs(dirname) 
    else:
        print('Directory found')

    for index_bond_dim, bond_dim in enumerate(bond_dim_vals):
        psi_ts, tebd = timeevolve_mps(psi_in, h, tlist, max_bond=bond_dim)
        
        mps_history = QuimbTensor1DResult.MPSResult(string_uuid, tlist, psi_ts)
        mps_history.parameters = {\
            "n_spins" : n_spins, \
            "j": j_int, \
           "bx": bx, \
        }
        mps_histories_bd[index_bond_dim] = mps_history 

        print('Saving file')
        # for index_t, (t, psi) in enumerate(zip(tlist, psi_ts)):
        #     filename = os.path.join(dirname, 'chi=%d_' %(bond_dim) + str(uuid.uuid4()) + '.pkl')
        #     save_mps(filename, psi, psi, t, t, h, None)

        #     la, lb = n_spins//2, n_spins//2+1
        #     zz_corr_current = \
        #         calc_manyspin_moment(psi, [sigmaz, sigmaz], [la, lb]) \
        #         - calc_manyspin_moment(psi, [sigmaz], [la]) \
        #         * calc_manyspin_moment(psi, [sigmaz], [lb])

        #     zz_corr[index_bond_dim, index_t] = np.abs(zz_corr_current)

    index_ref = -1
    for index_t, (t, psi) in enumerate(zip(tlist, psi_ts)):
        psi_ref = mps_histories_bd[index_ref].psi_t[index_t]

        for index_bond_dim, bond_dim in enumerate(bond_dim_vals):
            psi = mps_histories_bd[index_bond_dim].psi_t[index_t]

            innerproducts[index_bond_dim, index_t] \
                = np.abs(psi_ref.H @ psi)**2 / np.abs(psi_ref.H @ psi_ref.H) / np.abs(psi.H @ psi.H)

    # with open(os.path.join(dirname, 'zz'), 'wb') as f:
    #         pickle.dump(zz_corr, f)


    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.size'] = 12

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    for index_bond_dim, bond_dim in enumerate(bond_dim_vals):
        ax.plot(tlist/2/pi, 1-innerproducts[index_bond_dim, :], label=r'$\chi=%d$' % bond_dim)

    ax.legend(loc='best')
    ax.grid()
    ax.set_xlabel(r'$Jt/(2\pi)$')

    prefix = time.strftime('%Y-%m-%d_%H-%M-%S')
    plt.savefig(prefix + '_quimb_%dspins.pdf' % (n_spins,))

    # fig, (ax_abs, ax_err) = plt.subplots(2, 1, figsize=(12, 12))

    # for index_bond_dim, bond_dim in enumerate(bond_dim_vals):
        
    #     label = r'$\chi=%d$' % bond_dim_vals[index_bond_dim]
            
    #     ax_abs.plot(tlist/2/pi, zz_corr[index_bond_dim, :], \
    #             label=label)

    #     ax_err.plot(tlist/2/pi, np.abs(zz_corr[index_bond_dim, :] - zz_corr[-1, :]), \
    #             label=label)


    # #ax.set_yscale('log')
    # ax_abs.legend(loc='best')
    # ax_abs.grid()
    # ax_abs.set_xlabel(r'$Jt/(2\pi)$')

    # ax_abs.set_ylabel('Z Z')

    # ax_err.set_ylabel('Z Z')
    # ax_err.set_xlabel(r'$Jt/(2\pi)$')
    # ax_err.legend(loc='best')
    # ax_err.grid()

    # plt.savefig('2020-09-22_zzcor_%dspins.pdf' % (n_spins,))
