import numpy as np

import uuid
import time
import itertools
import logging
import os
import pickle
import pandas

import tenpy
import tenpy.models
import tenpy.algorithms
import tenpy.simulations
import tenpy.networks
import tenpy.networks.site

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s',
        level=logging.DEBUG)

import sys
sys.path.append("../src")
logging.info(sys.path)

import config

####################################################################################################
def tuple_to_bitstring(t, length_bitstring):
    """
    Parameters
    ----------
    t: 
    tuple with positions of bits to be set to 1

    length_bitstring:
    Length of the bitstring to generate
    """

    bitlist = ['0' for _ in range(length_bitstring)]

    for ix in t:
        bitlist[ix] = '1'

    bitlist.reverse()
    bitstring = ''.join([b for b in bitlist])
 
    bitinteger = int(bitstring, base=2)

    return bitinteger

####################################################################################################
if __name__ == '__main__':
    n_spins:int = 60

    string_uuid:str = "305a3b4a-8abe-41b1-a130-5084a5a29151"
    bonddim:int = 256
    
    string_uuid:str = "165236ce-d7d7-4b74-ac31-52fd54384335"
    bonddim:int = 128
    
    string_uuid:str = "079a0881-3a7f-4803-b018-59c9e6e4c0e2"
    bonddim:int = 64
    
    string_uuid:str = "8cbb6b94-ea57-42b7-865a-8638ceef4ec9"
    bonddim:int = 32
    
    string_uuid:str = "f9d94016-212d-4a93-8407-52819cdb9a5d"
    bonddim:int = 8
    
    string_uuid:str = "553055ff-9d26-41c4-90db-5403a7170b4b"
    bonddim:int = 16

    size_marginal:int = 2

    filename_mps_df = os.path.join(
            config.mps_directory, "%s_mpsHistory.pkl" % (string_uuid,))

    READ_FLAGS = "rb"

    with open(filename_mps_df, READ_FLAGS) as iofile:
        df_mps_all = pickle.load(iofile)

    ## This is a bit of a hack. `df_mps_all` is a `numpy.ndarray` of `pandas.DataFrame` with only
    ## one element. I may need to change the way I save the MPS history.
    df_mps_all = df_mps_all[0]

    logging.info("df_mps_all = \n%s" % (df_mps_all,))

    ix_time_list = df_mps_all["ix_time"].sort_values().unique()
    time_list = df_mps_all["time"].sort_values().unique()

    logging.info("ix_time_list = %s" % ix_time_list)

    logging.info("Evaluating %d-spin reduced density operators" % (size_marginal))

    walltime_begin = time.time()
    rows_reduced_dm = []

    sites_sel_list = [s for s in itertools.combinations(range(n_spins), size_marginal)]
    logging.info("sites_set_list = %s" % sites_sel_list)

    for row_iteration in df_mps_all.itertuples():
        for sites_sel in sites_sel_list:

            logging.info("Reduced state from bonddim = %d for %s in %d at ix_time = %d" \
                         % (bonddim, sites_sel, n_spins, row_iteration.ix_time))
            mps = row_iteration.mps
            logging.info("mps.chi = %s" % (mps.chi))
            rho = mps.get_rho_segment(sites_sel)

            row_created = {
                "ix_time": row_iteration.ix_time,
                "time": row_iteration.ix_time,
                "bonddim": bonddim,
                "sites_sel": sites_sel,
                "sites_sel_int": tuple_to_bitstring(sites_sel, n_spins),
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
    filename_df = os.path.join(
            config.reducedstate_directory, "%s_%d-spin.pkl" % (string_uuid, size_marginal))

    with open(filename_df, "wb") as iofile:
        pickle.dump(df_reduced_dm, iofile)

    logging.info("Finished saving %d-spin marginals" % (size_marginal))
