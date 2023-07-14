import numpy as np
import itertools

import uuid
import time
import itertools
import logging

import tenpy
import tenpy.models
import tenpy.algorithms
import tenpy.simulations
import tenpy.networks
import tenpy.networks.site

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

import sys
sys.path.append("../src")
logging.info(sys.path)

from wrap_tenpy.TenPy_OnlineCompress import *
from wrap_tenpy.distancemeasures import hilbertschmidt_distance

####################################################################################################
if __name__ == '__main__':

    string_date:str = "2023-058_13-37-35"
    n_spins:int = 20

    string_date:str = "2023-088_20-20-15"
    n_spins:int = 80

    string_date:str = "a33cdc3b-dd7b-4ad5-aaa2-79bd3d4f0969"
    n_spins:int = 40

    size_marginal:int = 1

    filename_mps_df = "pkl/mps/%s_tebd_mps.pkl" % (string_date)

    READ_FLAGS = "rb"

    with open(filename_mps_df, READ_FLAGS) as iofile:
        df_mps_all = pickle.load(iofile)

    logging.info("df_mps_all = \n%s" % (df_mps_all,))

    bonddim_list = df_mps_all["bonddim"].sort_values().unique()
    ix_time_list = df_mps_all["ix_time"].sort_values().unique()
    time_list = df_mps_all["time"].sort_values().unique()

    logging.info("bonddim_list = %s" % bonddim_list)
    logging.info("ix_time_list = %s" % ix_time_list)

    logging.info("Evaluating %d-spin reduced density operators" % (size_marginal))

    walltime_begin = time.time()
    rows_reduced_dm = []

    sites_sel_list = [s for s in itertools.combinations(range(n_spins), size_marginal)]
    logging.info("sites_set_list = %s" % sites_sel_list)

    for row_iteration in df_mps_all.itertuples():
        for sites_sel in sites_sel_list:

            logging.info("Reduced state from bonddim=%d for %s in %d at ix_time = %d" \
                         % (row_iteration.bonddim, sites_sel, n_spins, row_iteration.ix_time))
            mps = row_iteration.mps

            rho = mps.get_rho_segment(sites_sel)

            row_created = {
                "ix_time": row_iteration.ix_time,
                "time": row_iteration.ix_time,
                "bonddim": row_iteration.bonddim,
                "sites_sel": sites_sel,
                "rho": rho,
            }

            rows_reduced_dm.append(row_created)

    df_reduced_dm = pandas.DataFrame(rows_reduced_dm)

    walltime_end:float = time.time()
    walltime_duration:float = walltime_end - walltime_begin
    logging.info("Time taken = %g s" % (walltime_duration))
    logging.info("df_reduced_dm =\n%s" % (df_reduced_dm))

    WRITE_FLAGS = "wb"
    logging.info("Saving %d-spin marginals" % (size_marginal))
    filename_df = "pkl/marginals/%s_%d-spin.pkl" % (string_date, size_marginal)
    with open(filename_df, "wb") as iofile:
        pickle.dump(df_reduced_dm, iofile)

    logging.info("Finished saving %d-spin marginals" % (size_marginal))
