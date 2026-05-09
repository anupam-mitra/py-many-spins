import numpy as np

import uuid
import pickle
import os
import argparse

import time
import logging

import quimb
import quimb.tensor

logging.basicConfig(
    format='%(asctime)s: %(levelname)s: %(message)s',
    level=logging.INFO,
)

import sys
sys.path.append("../src")
logging.info(sys.path)

import config

from manybody_util.spinmodel import tilted_field_ising_1d
from manybody_backends.quimb.spinmodel import to_quimb_spinham1d
from manybody_backends.quimb.quimbtebd import TEBDWrapper, spinhalf_state


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(
        prog="quimb_tebdmain.py",
        description="Calculates time evolution of a matrix product state using Quimb TEBD.",
        epilog="",
    )

    argument_parser.add_argument("--systemsize", type=int)
    argument_parser.add_argument("--bonddim", type=int)

    args = argument_parser.parse_args()
    logging.info("Input arguments = %s" % vars(args))

    systemsize = args.systemsize
    bonddim = args.bonddim

    logging.info("Using quimb version %s" % (quimb.__version__,))

    algorithm = 'TEBD'

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
    builder = to_quimb_spinham1d(spin_model)

    mps_in = quimb.tensor.MPS_product_state(
        [spinhalf_state(theta, phi)] * systemsize,
        cyclic=False,
    )

    t_initial = 0.0
    t_final = 20.0 / np.abs(j_int)
    n_steps = 2 * int(t_final) + 1

    t_list = np.linspace(t_initial, t_final, n_steps)
    t_steps = np.diff(t_list)
    logging.info("t_step = %s" % (t_steps,))

    trotter_params = {
        "order": 4,
        "dt": t_list[1] - t_list[0],
    }

    trunc_params = {
        "chi_max": bonddim,
        "degeneracy_tol": 1e-6,
        "svd_min": None,
    }

    walltime_begin = time.time()
    uuid_string_bonddim = '%s' % uuid.uuid4()

    logging.info("%s: Using trunc_params = %s" % (algorithm, trunc_params,))

    wrap = TEBDWrapper(builder, mps_in, t_list, trotter_params, trunc_params)
    wrap.evolve()
    logging.info("wrap = %s" % (wrap,))

    df_mps, mps_list = wrap.get_mps_history_df()
    df_mps["uuid_bonddim"] = uuid_string_bonddim

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
        'library': 'Quimb',
    }
    logging.info("param_dict = %s" % param_dict)

    filename_index = os.path.join(
        config.index_directory, "%s.pkl" % (uuid_string_bonddim,)
    )
    with open(filename_index, "wb") as iofile:
        pickle.dump(param_dict, iofile)

    logging.info("Saving MPS")

    filename_mps_df = os.path.join(
        config.mps_directory, "%s_index.pkl" % (uuid_string_bonddim,)
    )
    with open(filename_mps_df, "wb") as iofile:
        pickle.dump(df_mps, iofile)

    for row, mps in zip(df_mps.itertuples(), mps_list):
        filename_mps = os.path.join(
            config.mps_directory, "%s.pkl" % (row.uuid_str,)
        )
        with open(filename_mps, "wb") as iofile:
            pickle.dump(mps, iofile)

    logging.info("Finished saving MPS")

    walltime_end = time.time()
    walltime_duration = walltime_end - walltime_begin
    logging.info("Time taken = %g s" % (walltime_duration,))
