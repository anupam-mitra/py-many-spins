import numpy as np
import os
import uuid
import time

import tenpy
import tenpy.linalg.np_conserved as npc

import tenpy.networks.mps, tenpy.models.tf_ising, tenpy.algorithms#, tenpy.tools.hdf5_io

from numpy import pi, sqrt

def timeevolve_mps (psi_in, tlist, M, cutoff=None, max_bond=None, \
    title_string=None):
    psi_ts = [psi_in.copy()] # states

    tebd_params = {
    "order": 4,
    #"dt": pi/5,
    "trunc_params": {
        "chi_max": max_bond,
        "svd_min": cutoff,
    },
    "N_steps": 1
    }

    timestamp_start = time.time()

    # generate the state at each time in tlist
    for s in range(1, len(tlist)):
        
        print('>>>>>> Time step %d <<<<<<' % (s,))

        tebd_params['dt'] = tlist[s] - tlist[s-1]
        
        psi_out = psi_in.copy()
        eng = tenpy.algorithms.tebd.Engine(psi_out, M, tebd_params)
        eng.run() 

        psi_ts += [psi_out.copy()]
        psi_in = psi_out
        
    timestamp_end = time.time()

    print('%s: took %g seconds' % (title_string, (timestamp_end - timestamp_start),))
    
    return psi_ts 

############################################################################################
def tebd_simulation_run (psi_in, tlist, model, tebd_params,
    title_string=None):
    '''

    '''

    # Create a simulation id

    # Save simulation details to a file

    # Create an entry to the index

    # Create an id for the initial state
    curr_id = None
    prev_id = None
    next_id = None

    # Save the initial state

    # Loop over time instants
    for s in range(1, len(tlist)):
        
        print('>>>>>> Time step %d <<<<<<' % (s,))

        prev_id = curr_id
        curr_id = next_id
        next_id = None # TODO create a new id
        
        # Evolve the state
        dt = tlist[s] - tlist[s-1]
        psi_fin = tmps_step_evolve(psi_in, model, dt, tebd_params)

        # Create a state id, save the state

        psi_ts += [psi_fin.copy()]
        psi_in = psi_fin
        
    timestamp_end = time.time()

############################################################################################

############################################################################################
def tmps_step_evolve (psi_in, model, dt, tebd_params):
    '''
    Propagate a matrix product state

    Parameters
    ----------
    psi_in: Initial state

    model: Model used to propagate the state

    dt: Time duration

    tebd_params: Parameters used for TEBD

    Returns
    -------
    psi_fin: Final state
    '''

    if tebd_params == None:
        max_bond = None
        svd_cutoff = None

        tebd_params = {
            "order": 4,
            "trunc_params": {
                "chi_max": max_bond,
                "svd_min": svd_cutoff,
            },
            "N_steps": 1
        }

    tebd_params['dt'] = dt

    psi_fin = psi_in.copy()
    eng = tenpy.algorithms.tebd.Engine(psi_fin, model, tebd_params)
    eng.run()

    return psi_fin
############################################################################################

############################################################################################
# def model_save (filename, model, tebd_params, psi_in_ref):
#     '''
#     Save a model used for tebd simulations to disk

#     Parameters
#     ----------
#     filename: File name

#     model: Model used for tebd

#     tebd_params: Parameters used for TEBD

#     psi_in_ref: Reference to the initial state

#     Returns
#     -------
#     '''
#     save_dict = {\
#         'model': model, \
#         'tebd_parms': tebd_params, \
#         'psi_in_ref': psi_in_ref, \
#     }

#     with h5py.File(filename, 'w') as f:
#         tenpy.tools.hdf5_io.save_to_hdf5(f, save_dict)
############################################################################################

############################################################################################
# def tmps_save (filename, psi, t, model_ref):
#     '''
#     Save a matrix product state to disk, creating and index entry.

#     Parameters
#     ----------

#     filename: File name

#     psi: State

#     model_ref: Reference to the model

#     prev_ref: Reference to previous state

#     next_ref: Referene to next state

#     t: Time

#     Returns
#     -------

#     '''
#     data = {\
#         'psi': psi, \
#         't': t, \
#         'model_ref': model_ref, \
#         'prev_ref': prev_ref, \
#         'next_ref': next_ref,
#     }

#     with h5py.File(filename, 'w') as f:
#         tenpy.tools.hdf5_io.save_to_hdf5(f, data)

############################################################################################

############################################################################################
def calc_manyspin_moment (mps, ops, sites, n_spins=None):
    
    sites_unique = list(set(sites))
    duplicate_flag = (len(sites_unique) != len(sites))
    
    if not duplicate_flag:
        moment = mps.expectation_value(ops, sites,)

    else:
        moment = float("nan")
    return moment
############################################################################################


if __name__ == '__main__':
    j_int = 1
    bx_values = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    n_spins = 32
    n_steps = 256

    t_initial = 0

    print('-' * 80)
    print(time.strftime('%Y-%m-%d %T'))

    for bx in bx_values:
        print('-' * 80)
        print(time.strftime('%Y-%m-%d %T'))
        print('Starting bx/jzz = %g' % (bx/j_int,))
        print('-' * 80)
        t_final = n_spins / j_int * 2 * pi if bx < j_int else n_spins / bx * 2 * pi
        title_string = 'bx/jzz = %g, n_spins = %d, jzz t_final / pi = %g' % (bx/j_int, n_spins, j_int * t_final/pi)

        tlist = np.linspace(t_initial, t_final, n_steps)

        M = tenpy.models.tf_ising.TFIChain({"L": n_spins, "J": j_int, "g": bx, "bc_MPS": "finite"}) 
        psi_upz = tenpy.networks.mps.MPS.from_product_state(M.lat.mps_sites(), ['up'] * n_spins, "finite")
        psi_downz = tenpy.networks.mps.MPS.from_product_state(M.lat.mps_sites(), ['down'] * n_spins, "finite")

        psi_in = psi_upz.add(psi_downz, 1/sqrt(2), 1/sqrt(2))
        psi_ts = timeevolve_mps(psi_in, tlist, M, title_string=title_string, max_bond=2)
        print('-' * 80)
        print(time.strftime('%Y-%m-%d %T'))
        print('Finished bx/jzz = %g' % (bx/j_int,))
        print('-' * 80)

    print("Saving state")
    for t, psi in zip(tlist, psi_ts):
        filename = os.path.join('../Data', 'TenPy_MPS_' + str(uuid.uuid4()) + '.h5')
        #save_mps(filename, psi, psi, t, t, M, None)
   
    print("Saved state")
    # SxSx  = npc.outer(psi_in.sites[0].Sx.replace_labels(['p', 'p*'], ['p0', 'p0*']),
    #         psi_in.sites[1].Sx.replace_labels(['p', 'p*'], ['p1', 'p1*']))

    moments = np.zeros_like(tlist)
    la, lb = n_spins//4, n_spins//2
    for index_t, (t, psi) in enumerate(zip(tlist, psi_ts)):
        #moment = psi.expectation_value(SxSx, sites=[0,  1])
        moment = psi.expectation_value_term([('Sx', la), ('Sx', lb)]) \
            - psi.expectation_value_term([('Sx', la)]) *  psi.expectation_value_term([('Sx', lb)])
        moments[index_t] = 4*moment
        #print('At t=%g, <Z Z> is %g',  (t, moment))
        

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.size'] = 12
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))

    ax.plot(tlist/2/pi, moments)
    ax.grid()
    ax.set_xlabel(r'$Jt/(2\pi)$')

    ax.set_ylabel('Z Z')
    plt.savefig('2020-12-11_momments_TenPy.pdf')