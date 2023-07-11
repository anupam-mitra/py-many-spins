import numpy as np
import h5py
import os
import uuid
import time

import tenpy
import tenpy.linalg.np_conserved as npc

import tenpy.networks.mps, tenpy.models.tf_ising, tenpy.algorithms, tenpy.tools.hdf5_io

from numpy import pi, sqrt

class TFIChainDecoherent(tenpy.models.tf_ising.TFIModel):
    r"""Transverse field Ising model on a general lattice with local decoherence

    Decoherence is defined by local jump operators

    """
    pass

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

    tebd_params = {
        "order": 4,
        #"dt": pi/5,
        "trunc_params": {
            "chi_max": max_bond,
            "svd_min": cutoff,
        },
        "N_steps": 1
    }

    # TODO the model object needs to have the non Hermitian part of the Hamiltonian

    # while the norm is less than generated random number
    # evolve the state using tebd.

    psi_out = psi_in.copy()
    eng = tenpy.algorithms.tebd.Engine(psi_out, model, tebd_params)
    eng.run() 
        
    
    psi_ts += [psi_out.copy()]
    psi_in = psi_out

    if norm < random_numbers[count_random_used_norm]:
        pass