import numpy as np
import quimb as qu
import quimb.tensor as qtn

import time
import scipy
import scipy.linalg

import sys
import logging
import pandas
import pickle
import uuid
import os

from numpy import pi, sin, cos, sqrt, exp

##### Functions #####
    
def uuid_gen ():
    uuid_str = str(uuid.uuid4())
    return uuid_str

def spinhalf_state (ang_polar, ang_azimuth):
    return [cos(ang_polar/2),\
           sin(ang_polar/2) * exp(1j * ang_azimuth)]

from circuit1d import AlternateOneQubitTwoQubit1D
from quimbtebd import QuimbTEBD1DSolver

def make_builder (interact_energies, interact_ops):
    
    builder = qtn.SpinHam1D(S=1/2, cyclic=False)

    for ll in range(len(local_ops)):
        if abs(local_energies[ll]) > 1e-8:
            builder.add_term(local_energies[ll], local_ops[ll])

    for ll in range(len(interact_ops)):
        if abs(interact_energies[ll]) > 1e-8:
            builder.add_term(interact_energies[ll], interact_ops[ll], interact_ops[ll])
       
    return builder


def timeevolve_tebd ():

    split_opts['max_bond'] = bonddim

    solver = QuimbTEBD1DSolver(
        psi_in, n_spins, t_initial, t_final, n_steps, \
        hamiltonian_builder=builder, \
        trotter_opts=trotter_opts, split_opts=split_opts)

    uuid_str = uuid_gen()
    logger.info('n_spins = %d, chi=%d, jxx=%g, bz=%g, uuid=%s' % (n_spins, bonddim, jxx, bz, uuid_str))
    logger.info('Executing TEBD at', time.strftime('%Y-%m-%d %T %s'))

    timestamp_start = time.time()
    solver.run()
    timestamp_end = time.time()

    walltime = (timestamp_end - timestamp_start)
    
    t_list = solver.t_list
    states = solver.states
    
    
#### End Functions #####

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    DATA_DIR = '../../Data/2020-March/Quimb/'

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    n_spins = 36
    n_steps = 512

    if n_spins <= 16:
        bonddim_values = range(1, (1 << (n_spins//2)) + 1)
    else:
        bonddim_values = range(1, (1 << (n_spins//2)) + 1)

    bonddim_values = range(1, 33)

    #### Initial conditions
    zeros_all = '0' * n_spins
    zero_one_alt = '01' * n_spins

    psi_in = qtn.MPS_computational_state(zero_one_alt)

    theta = pi/2
    phi = pi/2

    psi_in = qtn.MPS_product_state([spinhalf_state(theta, phi)]*n_spins, cyclic=False)

    #### Hamiltonian
    jxx = 1
    bx_values = jxx * np.asarray([0, 1])
    bz_values = jxx * np.asarray([0.5, -0.5, 1.0, -1.0, 2.0, -2.0])
    bz_values = jxx * np.asarray([1])

    identity = qu.pauli('I')
    x_gate_gen = qu.pauli('X')
    z_gate_gen = qu.pauli('Z')

    xx_gate_gen = qu.kronpow(qu.pauli('X'), 2)
    zz_gate_gen = qu.kronpow(qu.pauli('Z'), 2)

    #### Time
    t_initial = 0
    t_final = n_spins / jxx

    data_index = []

    if False:
        # A circuit using a first order Trotter formula

        for ix_bz, bz in enumerate(bz_values):
            ang_x = t_final / n_steps * bx
            ang_z = t_final / n_steps * bz
            ang_xx = t_final / n_steps * jxx

            onequbitgate = scipy.linalg.expm(-1j * z_gate_gen * ang_z) \
                        @ scipy.linalg.expm(-1j * x_gate_gen * ang_x)
            twoqubitgate = scipy.linalg.expm(-1j * xx_gate_gen * ang_xx)

            for ix_bonddim, bonddim in enumerate(bonddim_values):

                circuit = AlternateOneQubitTwoQubit1D(onequbitgate, twoqubitgate, n_spins, max_bond=bonddim)

                uuid_str = uuid_gen()
                logger.info('n_spins = %d, chi=%d, jxx=%g, bz=%g, uuid=%s' % (n_spins, bonddim, jxx, bz, uuid_str))
                logger.info('Evolving circuit at', time.strftime('%Y-%m-%d %T %s'))

                timestamp_start = time.time()
                circuit.evolve_circuit(psi_in, n_steps)
                timestamp_end = time.time()

                walltime = (timestamp_end - timestamp_start)

                logger.info('Saving circuit at', time.strftime('%Y-%m-%d %T %s'))

                data_index += [{
                    'uuid_str': uuid_str,
                    'model': 'titled-field-Ising',
                    'procedure': '[quimb.tensor.gate]',
                    'n_spins': n_spins,
                    'chi_max': bonddim,
                    'wall_time_second': walltime,
                    'jxx': jxx,
                    'bz': bz,
                    'bx': bx,
                    't_initial_jxx': 0,
                    't_final_jxx': jxx * t_final,
                    'n_steps': n_steps
                }]

                mpspklfilename = os.path.join(DATA_DIR, uuid_str + '.pkl')
                with open(mpspklfilename, 'wb') as pklout:
                    pickle.dump(circuit, pklout)


    # TEBD using quimb's TEBD
    local_ops = [z_gate_gen, x_gate_gen]
    interact_ops = [x_gate_gen]

    trotter_opts = {
        'tol' : None,
        'dt' : t_final / n_steps,
    }
    split_opts = {
        'max_bond' : None,
        'cutoff' : None
    }


    for ix_bz, bz in enumerate(bz_values):
        for ix_bx, bx in enumerate(bx_values):
            interact_energies = [jxx]
            local_energies = [bz, bx]

            builder = qtn.SpinHam1D(S=1/2, cyclic=False)

            for ll in range(len(local_ops)):
                if abs(local_energies[ll]) > 1e-8:
                    builder.add_term(local_energies[ll], local_ops[ll])

            for ll in range(len(interact_ops)):
                if abs(interact_energies[ll]) > 1e-8:
                    builder.add_term(interact_energies[ll], interact_ops[ll], interact_ops[ll])

            for ix_bonddim, bonddim in enumerate(bonddim_values):

                split_opts['max_bond'] = bonddim

                solver = QuimbTEBD1DSolver(
                    psi_in, n_spins, t_initial, t_final, n_steps, \
                    hamiltonian_builder=builder, \
                    trotter_opts=trotter_opts, split_opts=split_opts)

                uuid_str = uuid_gen()
                logger.info('n_spins = %d, chi=%d, jxx=%g, bz=%g, uuid=%s' % (n_spins, bonddim, jxx, bz, uuid_str))
                logger.info('Executing TEBD at', time.strftime('%Y-%m-%d %T %s'))

                timestamp_start = time.time()
                solver.run()
                timestamp_end = time.time()

                walltime = (timestamp_end - timestamp_start)

                logger.info('Saving TEBD at', time.strftime('%Y-%m-%d %T %s'))

                data_index += [{
                    'uuid_str': uuid_str,
                    'model': 'titled-field-Ising-nn',
                    'procedure': 'quimb.tensor.TEBD',
                    'n_spins': n_spins,
                    'chi_max': bonddim,
                    'wall_time_second': walltime,
                    'jxx': jxx,
                    'bz': bz,
                    'bx': bx,
                    't_initial_jxx': 0,
                    't_final_jxx': jxx * t_final,
                    'n_steps': n_steps
                }]

                mpspklfilename = os.path.join(DATA_DIR, uuid_str + '.pkl')
                with open(mpspklfilename, 'wb') as pklout:
                    pickle.dump(solver, pklout)

    df_index = pandas.DataFrame(data=data_index)

    dfpklfilename = '%02d_spins_%d_%d_fim_init_upy.pkl' % (n_spins, np.min(bonddim_values), np.max(bonddim_values))

    with open(dfpklfilename, 'wb') as pklout:
        pickle.dump(df_index, pklout)
