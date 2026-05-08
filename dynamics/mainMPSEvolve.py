import argparse
import logging
import os
import pickle
import sys
import time
import uuid

import numpy as np
import tenpy

logging.basicConfig(
    format='%(asctime)s: %(levelname)s: %(message)s',
    level=logging.INFO,
)

sys.path.append("../src")
logging.info(sys.path)

import config

from manybody_util.spinmodel import tilted_field_ising_1d
from wrap_tenpy.spinmodel import to_tenpy_model
from wrap_tenpy.timeevolution import manyspin_product_mps, solve_mps_history


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(
        prog="tenpy_mps_evolve",
        description="Calculates MPS time evolution using TenPy.",
        epilog="",
    )
    argument_parser.add_argument("--systemsize", type=int, required=True)
    argument_parser.add_argument("--bonddim", type=int)
    argument_parser.add_argument(
        "--algorithm",
        choices=("TEBD", "TDVP"),
        default="TEBD",
    )

    args = argument_parser.parse_args()
    logging.info("Input arguments = %s" % vars(args))
    logging.info("Using tenpy version %s" % (tenpy.__version__,))

    theta = 0.5 * np.pi
    phi = 0.5 * np.pi

    j_int = 0.1
    b_parallel = 0.15
    b_perp = 1.0

    logging.info("Using %s" % (args.algorithm,))
    logging.info("systemsize = %d" % (args.systemsize,))
    logging.info(
        "j_int = %g, b_parallel = %g, b_perp = %g"
        % (j_int, b_parallel, b_perp)
    )

    spin_model = tilted_field_ising_1d(
        args.systemsize,
        j_xx=j_int,
        b_z=b_perp,
        b_x=b_parallel,
        bc="open",
    )
    tenpy_model = to_tenpy_model(spin_model, bc_mps="finite", conserve=None)
    mps_in = manyspin_product_mps(
        tenpy_model.lat.mps_sites(),
        theta,
        phi,
        bc=tenpy_model.lat.bc_MPS,
        unit_cell_width=tenpy_model.lat.mps_unit_cell_width,
    )

    t_initial = 0.0
    t_final = 20.0 / abs(j_int)
    n_steps = 2 * int(t_final) + 1
    t_list = np.linspace(t_initial, t_final, n_steps)

    evolution_params = {"N_steps": 1}
    if args.algorithm == "TEBD":
        evolution_params["order"] = 4

    trunc_params = {
        "chi_max": args.bonddim,
        "degeneracy_tol": 1e-6,
    }

    walltime_begin = time.time()
    uuid_string_bonddim = "%s" % uuid.uuid4()

    logging.info("%s: evolution_params = %s" % (args.algorithm, evolution_params))
    logging.info("%s: trunc_params = %s" % (args.algorithm, trunc_params))

    df_mps, mps_list = solve_mps_history(
        tenpy_model,
        mps_in,
        t_list,
        algorithm=args.algorithm,
        evolution_params=evolution_params,
        trunc_params=trunc_params,
        metadata={"bonddim": args.bonddim},
    )
    df_mps["uuid_bonddim"] = uuid_string_bonddim

    param_dict = {
        "uuid_bonddim": uuid_string_bonddim,
        "j_int": j_int,
        "b_parallel": b_parallel,
        "b_perp": b_perp,
        "systemsize": args.systemsize,
        "bonddim": args.bonddim,
        "t_initial": t_initial,
        "t_final": t_final,
        "algorithm": args.algorithm,
        "library": "TenPy",
    }
    logging.info("param_dict = %s" % param_dict)

    filename_index = os.path.join(
        config.index_directory,
        "%s.pkl" % (uuid_string_bonddim,),
    )
    with open(filename_index, "wb") as iofile:
        pickle.dump(param_dict, iofile)

    logging.info("Saving MPS")

    filename_mps_df = os.path.join(
        config.mps_directory,
        "%s_index.pkl" % (uuid_string_bonddim,),
    )
    with open(filename_mps_df, "wb") as iofile:
        pickle.dump(df_mps, iofile)

    for row, mps in zip(df_mps.itertuples(), mps_list):
        filename_mps = os.path.join(
            config.mps_directory,
            "%s.pkl" % (row.uuid_str,),
        )
        with open(filename_mps, "wb") as iofile:
            pickle.dump(mps, iofile)

    logging.info("Finished saving MPS")

    walltime_duration = time.time() - walltime_begin
    logging.info("Time taken = %g s" % (walltime_duration,))
