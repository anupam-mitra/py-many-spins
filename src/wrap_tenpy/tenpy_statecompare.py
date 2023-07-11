#
# History `tenpy_statecompare.py`
# 2020-09-15: File created.
# 2022-11-14:


import numpy as np
import pickle
import uuid
import time

import tenpy

import tenpy.networks.mps, tenpy.models.tf_ising, tenpy.algorithms

from numpy import pi, sqrt

def timeevolve_mps (psi_in, tlist, M, cutoff=None, max_bond=None, title_string=None):
    print(']]]]]] Initial State [[[[[[', psi_in.chi)
    psi_ts = [psi_in.copy()] # states

    if cutoff == None:
        cutoff = 1e-12

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

    print('%s: took %g seconds' % (title_string, (timestamp_end - timestamp_start),))
    
    return psi_ts

j_int = 1
bx = 1

n_spins = 16
n_steps = 256

t_initial = 0

t_final = n_spins / j_int * 2 * pi
title_string = 'B/J = %g, n_spins = %d, J t_final / pi = %g' % (bx/j_int, n_spins, j_int * t_final/pi)

tlist = np.linspace(t_initial, t_final, n_steps)

M = tenpy.models.tf_ising.TFIChain({"L": n_spins, "J": j_int, "g": bx, "bc_MPS": "finite"}) 
psi_in = tenpy.networks.mps.MPS.from_product_state(M.lat.mps_sites(), ['up'] * n_spins, "finite")

bond_dim_vals = np.logspace(0, n_spins//2, num=n_spins//2+1, base=2)
mps_histories_bd = np.empty_like(bond_dim_vals, dtype=object)

for index_bond_dim, bond_dim in enumerate(bond_dim_vals):
    
    print('bond_dim = %d @ %s' % (bond_dim, time.asctime()))
    
    psi_ts = timeevolve_mps(psi_in.copy(), tlist, M, max_bond=int(bond_dim), title_string=title_string)
    
    mps_histories_bd[index_bond_dim] = np.asarray(psi_ts, dtype=object)   


# Compare states
inner_products_bd = np.zeros((n_steps, len(bond_dim_vals)), dtype=float)
index_ref = -1

for index_t, t in enumerate(tlist):    
    psi_ref = mps_histories_bd[index_ref][index_t]
   
    for index_bond_dim, bond_dim in enumerate(bond_dim_vals):
        psi = mps_histories_bd[index_bond_dim][index_t]
        
        inner_products_bd[index_t, index_bond_dim] = \
            np.abs(psi_ref.overlap(psi)) / np.abs(psi_ref.overlap(psi_ref))


# Plots
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['font.size'] = 12
fig, ax = plt.subplots(1, 1, figsize=(12, 10))

for index_bond_dim, bond_dim in enumerate(bond_dim_vals):
    
    label = r'$\chi=%d$' % bond_dim_vals[index_bond_dim]
        
    ax.plot(tlist/2/pi, 1-inner_products_bd[:, index_bond_dim], \
            label=label)
    
#ax.set_yscale('log')
ax.legend(loc='best')
ax.grid()
ax.set_xlabel(r'$Jt/(2\pi)$')
ax.set_ylabel('InFidelity')
plt.savefig('2020-09-15_infidelity_bonddim_%dspins.pdf' % (n_spins,))

