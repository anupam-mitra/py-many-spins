import numpy as np
import itertools

import uuid
import pandas
import pickle
import os
import argparse

import time
import logging

import tenpy
import tenpy.models
import tenpy.algorithms
import tenpy.simulations
import tenpy.networks
import tenpy.networks.site

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.INFO)

import sys
sys.path.append("../src")
logging.info(sys.path)

import config

### TODO ###
### 1. Refactor the time evolution wrappers for `tenpy.algorithms.tdvp.TwoSiteTDVPEngine`
###    and `tenpy.algorithms.tdvp.TEBDEngine`, which are currently the classes
###    `timeevolution.TDVPWrapper` and `timeevolution.TEBDWrapper`
###    respectively.

from wrap_tenpy.timeevolution import TEBDWrapper, TDVPWrapper, spinhalf_state

####################################################################################################
if __name__ == '__main__':
    
    argument_parser = argparse.ArgumentParser(
        prog="tebdmain.py",
        description="Calculates time evolution of a matrix product state using TEBD.",
        epilog=""
    )

    argument_parser.add_argument("--systemsize", type=int)
    argument_parser.add_argument("--bonddim", type=int)

    args = argument_parser.parse_args()
    logging.info("Input arguments = %s" % vars(args))

    systemsize:int = args.systemsize 
    bonddim:int = args.bonddim

    string_start_date = time.strftime("%Y-%j_%H-%M-%S", time.localtime())
    uuid_string_simulation = '%s' % uuid.uuid4()

    logging.info("date = %s" % (string_start_date,))
    logging.info("uuid = %s" % (uuid_string_simulation,))

    algorithm = np.random.choice(['TEBD',])

    logging.info("Using tenpy version %s" % (tenpy.__version__))
    string_desc = "%s_sfim_nspins=%d" % (algorithm, systemsize)

    # TenPy's uses XX interactions and Z as the transverse field.
    # Thus `theta` = 0.0 corresponds to the paramagnetic ground state.
    # Thus `theta` = `0.5*pi` and `1.5*pi` corresponds to the ferromagnetic ground states.
    theta:float = 0.5 * np.pi
    phi:float = 0.5 * np.pi

    theta_bfield:float = np.pi / 3
    j_int:float = - 1.0
    b_field: float = -1.0
    b_parallel:float = b_field * np.cos(theta_bfield)
    b_perp:float = b_field * np.sin(theta_bfield)

    logging.info("Using %s" % (algorithm))
    logging.info("systemsize = %d" % (systemsize))
    logging.info("j_int = %g, b_parallel = %g, b_perp = %g" % \
            (j_int, b_parallel, b_perp))

    site:tenpy.networks.site.SpinHalfSite \
            = tenpy.networks.site.SpinHalfSite(conserve=None)

    sfim_parameters:dict = {
        "L": systemsize,
        "J": j_int,
        "g": b_perp,
        "bc_MPS": "finite",
        "conserve": None
    }

    sfim:tenpy.models.tf_ising.TFIChain \
        = tenpy.models.tf_ising.TFIChain(sfim_parameters)

    if theta_bfield != np.pi/2: 
        sfim.manually_call_init_H = True
        sfim.add_onsite(b_parallel, 0, "Sigmax")
        sfim.init_H_from_terms()

    mps_in = tenpy.networks.mps.MPS.from_product_state(
        [site]*systemsize, p_state=[spinhalf_state(theta, phi)]*systemsize,
        bc="finite", dtype=complex)

    t_initial:float = 0
    t_final:float = 20.0 / np.abs(j_int)
    n_steps:int = 2*int(t_final) + 1

    t_list:list = np.linspace(t_initial, t_final, n_steps)
    t_steps:np.ndarray = np.diff(t_list)
    logging.info("t_step = %s" % (t_steps))

    if algorithm == 'TEBD':

        trotter_params:dict = {
            "order": 4,
            "dt": t_list[1] - t_list[0],
            "N_steps": 1
        }

        trunc_params:dict = {
            "chi_max": 256 ,
            "degeneracy_tol": 1e-6,
            "svd_min": None,
        }

    elif algorithm == 'TDVP':
        trunc_params = {
            "chi_max": 64 ,
            "degeneracy_tol": 1e-6,
            "svd_min": None,
        }
        tdvp_params = {
            "trunc_params" : trunc_params,
        }

    bonddim_list:np.ndarray = np.array([bonddim])
    n_bonddims:int = len(bonddim_list)
    logging.info(bonddim_list)

    walltime_begin:float = time.time()

    evolve_wrappers:np.ndarray = np.empty(bonddim_list.shape, dtype=object)
    df_mps:np.ndarray = np.empty(bonddim_list.shape, dtype=object)

    for ix_bonddim in range(n_bonddims):
        trunc_params["chi_max"] = bonddim_list[ix_bonddim]
        uuid_string_bonddim = '%s' % uuid.uuid4()

        logging.info("%s: Using trunc_params = %s" % (algorithm, trunc_params,))

        if algorithm == 'TEBD':
            wrap:TEBDWrapper = TEBDWrapper(sfim, mps_in, t_list, 
                                           trotter_params, trunc_params)
            wrap.evolve()
            evolve_wrappers[ix_bonddim] = wrap
            logging.info("wrap = %s" % (wrap))

        elif algorithm == 'TDVP':
            wrap:TDVPWrapper = TDVPWrapper(sfim, mps_in, t_list, 
                                           trunc_params, tdvp_params)
            wrap.evolve()
            evolve_wrappers[ix_bonddim] = wrap
            logging.info("wrap = %s" % (wrap))

        df_mps_current, mps_list = evolve_wrappers[ix_bonddim].get_mps_history_df()

        param_dict = {
            'uuid_simulation': uuid_string_simulation,
            'uuid_bonddim': uuid_string_bonddim,
            'j_int': j_int,
            'b_field': b_field,
            'theta_bfield': theta_bfield,
            'systemsize': systemsize,
            'bonddim': bonddim_list[ix_bonddim],
            't_initial': t_initial,
            't_final': t_final,
            'algorithm': algorithm,
            'library': 'TenPy',
        }
        logging.info("param_dict = %s" % param_dict)
        
        filename_index = os.path.join(
             config.index_directory, "%s.pkl" % (uuid_string_bonddim,))
        with open(filename_index, "wb") as iofile:
            pickle.dump(param_dict, iofile)

        logging.info("Saving MPS")

        filename_mps_df = os.path.join(
            config.mps_directory, "%s_mpsHistory.pkl" % (uuid_string_bonddim,))
        with open(filename_mps_df, "wb") as iofile:
            pickle.dump(df_mps, iofile)

        logging.info("Finished saving MPS")

    walltime_end:float = time.time()
    walltime_duration:float = walltime_end - walltime_begin
    logging.info("Time taken = %g s" % (walltime_duration))

