import argparse
import logging
import os
import pickle
import sys
import time
import uuid

import numpy as np

logging.basicConfig(
    format='%(asctime)s: %(levelname)s: %(message)s',
    level=logging.INFO,
)

sys.path.append("../src")
logging.info(sys.path)

import config

from manybody_util.spinmodel import tilted_field_ising_1d
from wrap_quspin.spinmodel import to_quspin_basis, to_quspin_hamiltonian
from wrap_quspin.timeevolution import manyspin_product_state, solve_state_history


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(
        prog="quspin_stateevolve.py",
        description="Calculates exact time evolution of a many-body state using QuSpin.",
        epilog="",
    )

    argument_parser.add_argument("--systemsize", type=int)
    argument_parser.add_argument("--bonddim", type=int)

    args = argument_parser.parse_args()
    logging.info("Input arguments = %s" % vars(args))

    if args.bonddim is not None:
        argument_parser.error("--bonddim is not supported for QuSpin exact evolution")

    systemsize = args.systemsize
    bonddim = None

    algorithm = 'evolve'

    theta = 0.5 * np.pi
    phi = 0.5 * np.pi

    theta_bfield = np.pi / 3
    j_int = -1.0
    b_field = -1.0
    b_parallel = b_field * np.cos(theta_bfield)
    b_perp = b_field * np.sin(theta_bfield)

    j_int = 0.1
    b_parallel = 0.15
    b_perp = 1.0

    logging.info("Using %s" % (algorithm,))
    logging.info("systemsize = %d" % (systemsize,))
    logging.info(
        "j_int = %g, b_parallel = %g, b_perp = %g"
        % (j_int, b_parallel, b_perp)
    )

    spin_model = tilted_field_ising_1d(
        systemsize,
        j_xx=j_int,
        b_z=b_perp,
        b_x=b_parallel,
        bc="open",
    )
    basis = to_quspin_basis(spin_model)
    hamiltonian = to_quspin_hamiltonian(spin_model, basis=basis)

    initial_state = manyspin_product_state(systemsize, theta, phi, basis=basis)

    t_initial = 0.0
    t_final = 20.0 / np.abs(j_int)
    n_steps = 2 * int(t_final) + 1

    t_list = np.linspace(t_initial, t_final, n_steps)
    t_steps = np.diff(t_list)
    logging.info("t_step = %s" % (t_steps,))

    walltime_begin = time.time()
    uuid_string_bonddim = '%s' % uuid.uuid4()

    df_states, state_list = solve_state_history(
        hamiltonian,
        initial_state,
        t_list,
        metadata={"bonddim": bonddim},
    )
    df_states["uuid_bonddim"] = uuid_string_bonddim

    param_dict = {
        'uuid_bonddim': uuid_string_bonddim,
        'j_int': j_int,
        'b_field': b_field,
        'theta_bfield': theta_bfield,
        'systemsize': systemsize,
        'bonddim': bonddim,
        't_initial': t_initial,
        't_final': t_final,
        'algorithm': algorithm,
        'library': 'QuSpin',
    }
    logging.info("param_dict = %s" % param_dict)

    filename_index = os.path.join(
        config.index_directory, "%s.pkl" % (uuid_string_bonddim,)
    )
    with open(filename_index, "wb") as iofile:
        pickle.dump(param_dict, iofile)

    logging.info("Saving states")

    filename_state_df = os.path.join(
        config.mps_directory, "%s_index.pkl" % (uuid_string_bonddim,)
    )
    with open(filename_state_df, "wb") as iofile:
        pickle.dump(df_states, iofile)

    for row, state in zip(df_states.itertuples(), state_list):
        filename_state = os.path.join(
            config.mps_directory, "%s.pkl" % (row.uuid_str,)
        )
        with open(filename_state, "wb") as iofile:
            pickle.dump(state, iofile)

    logging.info("Finished saving states")

    walltime_end = time.time()
    walltime_duration = walltime_end - walltime_begin
    logging.info("Time taken = %g s" % (walltime_duration,))
