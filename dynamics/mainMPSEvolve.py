import numpy as np
import itertools

import uuid
import pandas
import pickle
import os

import time
import logging

import tenpy
import tenpy.models
import tenpy.algorithms
import tenpy.simulations
import tenpy.networks
import tenpy.networks.site

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s', level=logging.DEBUG)

import sys
sys.path.append("../src")
logging.info(sys.path)

### TODO ###
### 1. [DONE] Rename the module `TenPy_OnlineCompress` to something better,
### 2. Refactor the time evolution wrappers for `tenpy.algorithms.tdvp.TwoSiteTDVPEngine`
###    and `tenpy.algorithms.tdvp.TEBDEngine`, which are currently the classes
###    `TenPy_OnlineCompress.TDVPWrapper` and `TenPy.OnlineCompress.TEBDWrapper`
###    respectively.
### 3.

#from wrap_tenpy.TenPy_OnlineCompress import *
from wrap_tenpy.timeevolution import TEBDWrapper, TDVPWrapper, spinhalf_state

####################################################################################################
if __name__ == '__main__':

    string_start_date = time.strftime("%Y-%j_%H-%M-%S", time.localtime())
    string_uuid_simulation = '%s' % uuid.uuid4()

    logging.info("date = %s" % (string_start_date,))
    logging.info("uuid = %s" % (string_uuid_simulation,))

    algorithm = np.random.choice(['TEBD', 'TEBD'])

    n_spins:int = 81

    logging.info("Using tenpy version %s" % (tenpy.__version__))
    string_desc = "%s_sfim_nspins=%d" % (algorithm, n_spins)

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
    logging.info("n_spins = %d" % (n_spins))
    logging.info("j_int = %g, b_parallel = %g, b_perp = %g" % \
            (j_int, b_parallel, b_perp))

    site:tenpy.networks.site.SpinHalfSite \
            = tenpy.networks.site.SpinHalfSite(conserve=None)

    sfim_parameters:dict = {
        "L": n_spins,
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
        [site]*n_spins, p_state=[spinhalf_state(theta, phi)]*n_spins,
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

    bonddim_list:np.ndarray = np.logspace(3, 8, 6, base=2, dtype=int)
    n_bonddims:int = len(bonddim_list)
    logging.info(bonddim_list)

    walltime_begin:float = time.time()

    evolve_wrappers:np.ndarray = np.empty(bonddim_list.shape, dtype=object)
    df_mps:np.ndarray = np.empty(bonddim_list.shape, dtype=object)

    for ix_bonddim in range(n_bonddims):
        trunc_params["chi_max"] = bonddim_list[ix_bonddim]
        string_uuid_bonddim = '%s' % uuid.uuid4()

        logging.info("%s: Using trunc_params = %s" % (algorithm, trunc_params,))

        if algorithm == 'TEBD':
            wrap:TEBDWrapper = TEBDWrapper(sfim, mps_in, t_list, 
                                           trotter_params, trunc_params)
            wrap.evolve()
            evolve_wrappers[ix_bonddim] = wrap

        elif algorithm == 'TDVP':
            wrap:TDVPWrapper = TDVPWrapper(sfim, mps_in, t_list, 
                                           trunc_params, tdvp_params)
            wrap.evolve()
            evolve_wrappers[ix_bonddim] = wrap

        df_mps[ix_bonddim] = evolve_wrappers[ix_bonddim].get_mps_history_df()

        param_dict = {
            'uuid_simulation': string_uuid_simulation,
            'uuid_bonddim': string_uuid_bonddim,
            'j_int': j_int,
            'b_field': b_field,
            'theta_bfield': theta_bfield,
            'n_spins': n_spins,
            'bonddim': bonddim_list[ix_bonddim],
            't_initial': t_initial,
            't_final': t_final,
            'algorithm': algorithm,
            'library': 'TenPy',
        }
        logging.info("param_dict = %s" % param_dict)
        
        filename_index = os.path.join(
                "pkl", "index", "%s.pkl" % (string_uuid_bonddim,))
        with open(filename_index, "wb") as iofile:
            pickle.dump(param_dict, iofile)

        logging.info("Saving MPS")

        filename_mps_df = os.path.join(
            "pkl", "mps", "%s_mpsHistory.pkl" % (string_uuid_bonddim,))
        with open(filename_mps_df, "wb") as iofile:
            pickle.dump(df_mps, iofile)

        logging.info("Finished saving MPS")

    walltime_end:float = time.time()
    walltime_duration:float = walltime_end - walltime_begin
    logging.info("Time taken = %g s" % (walltime_duration))

    for ix_bonddim in range(n_bonddims):
        df = df_mps[ix_bonddim]
        df["bonddim"] = [bonddim_list[ix_bonddim] for _ in range(n_steps)]

    df_mps_all = pandas.concat(df_mps, ignore_index=True)
    logging.info("df_mps_all = \n%s", (df_mps_all))

    logging.info("Saving MPS")
    filename_mps_df = os.path.join(
        "pkl", "mps", "%s_mpsHistory.pkl" % (string_uuid_simulation))
    with open(filename_mps_df, "wb") as iofile:
        pickle.dump(df_mps_all, iofile)
