import numpy as np
import pickle
import uuid
import os
import time
import itertools

import logging

import tenpy
import tenpy.networks.mps
import tenpy.models.tf_ising
import tenpy.algorithms

import qutip

from numpy import pi

DATA_DIR = '../../Data/2020-March/TenPy/'

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def uuid_gen ():
    uuid_str = str(uuid.uuid4())
    return uuid_str

def tenpy_mps_to_qutip_qobj (mps):
    '''
    Converts a matrix product state represented
    using a `tenpy` implementation in `tenpy.networks.mps.MPS`
    to a ket represented as `qutip.qobj.Qobj`
    by calculating each probability amplitude

    
    '''
    
    L = mps.L
    dimensions = mps.dim
    
    ket_labels = list(itertools.product(*[tuple(range(d)) for d in dimensions]))
    dim_manybody_state = 2**L
    
    amp = np.empty((dim_manybody_state), dtype=complex)
    
    for j in range(dim_manybody_state):
        label = ket_labels[j]
        basis_mps = tenpy.networks.mps.MPS.from_product_state(\
                        mps.sites, label, "finite")
    
        amp[j] = basis_mps.overlap(mps)
     
    qutip_qobj = qutip.qobj.Qobj(amp, dims=[mps.dim, [1]*len(mps.dim)])
    
    return qutip_qobj

def timeevolve_mps (psi_in, tlist, M, cutoff=None, max_bond=None):
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
        
        eng = tenpy.algorithms.tebd.Engine(psi_in, M, tebd_params)
        eng.run() 
        
        psi_ts += [psi_in.copy()]
        
    timestamp_end = time.time()

    walltime = (timestamp_end - timestamp_start)
    
    return psi_ts, walltime


logger = logging.getLogger('tebd')
logger.setLevel(logging.WARNING)

jxx = 1
bz = 1

n_spins = 8
n_steps = 256

if n_spins <= 12:
    bonddim_values = range(2, (1 << (n_spins//2)) + 1, 2)
else:
    bonddim_values = [4] + list(range(4, 256 + 1, 4))

bonddim_values = [None]
    
walltime_values = np.zeros_like(bonddim_values, dtype=float)

t_initial = 0

t_final = n_spins / jxx
title_string = 'B/J = %g, n_spins = %d, J t_final = %g' % (bz/jxx, n_spins, jxx * t_final)

tlist = np.linspace(t_initial, t_final, n_steps)

M = tenpy.models.tf_ising.TFIChain({"L": n_spins, "J": jxx, "g": bz, "bc_MPS": "finite", "conserve": None})

psi_in = tenpy.networks.mps.MPS.from_product_state(M.lat.mps_sites(), ['up'] * n_spins, "finite")

uuid_strings = []

for ix, bonddim in enumerate(bonddim_values):
    psi_ts, wt = timeevolve_mps(psi_in, tlist, M, max_bond=bonddim)
    
    walltime_values[ix] = wt
    uuid_str = uuid_gen()
    uuid_strings += [uuid_str]
    psi_ts_qutip = [tenpy_mps_to_qutip_qobj(psi) for psi in psi_ts]

    mpspklfilename = os.path.join(DATA_DIR, uuid_str + '.pkl')    
    with open(mpspklfilename, 'wb') as pklout:
        pickle.dump(psi_ts_qutip, pklout)

import pandas
import pickle

data_bench = []
for ix in range(len(bonddim_values)):
    data_bench += [{
        'uuid_str': uuid_strings[ix],
        'n_spins': n_spins,
        'procedure': 'tenpy.algorithms.tebd.Engine',
        'chi_max': bonddim_values[ix],
        'wall_time_second': walltime_values[ix],
        'jxx': jxx,
        'bz': bz,
        't_initial_jxx': 0,
        't_final_jxx': jxx * t_final,
        'n_steps': n_steps
    }]
df_bench = pandas.DataFrame(data=data_bench)

pklfilename = '%02d_spins_bench.pkl' % (n_spins,)

with open(pklfilename, 'wb') as pklout:
    pickle.dump(df_bench, pklout)
