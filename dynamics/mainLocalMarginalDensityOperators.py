import numpy as np

import uuid
import time
import itertools
import os
import pickle
import pandas
import h5py
import argparse
import logging

import tenpy
import tenpy.models
import tenpy.algorithms
import tenpy.simulations
import tenpy.networks
import tenpy.networks.site
import tenpy.tools.hdf5_io

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

    argument_parser = argparse.ArgumentParser(
        prog="marginalize",
        description="Generates random tenpy.networks.site.mps.MPS using TenPy's RandomUnitaryEvolution",
        epilog=""
    )

    argument_parser.add_argument("--mpsfilename", type=str)
    argument_parser.add_argument("--marginalsize", type=int)
    argument_parser.add_argument("--which", type=str)

    args = argument_parser.parse_args()
    logging.info("Input arguments = %s" % vars(args))

    mpsfilepath = os.path.join(config.mps_directory, "%s" % args.mpsfilename)
    logging.info("mpsfilepath = %s" % (mpsfilepath))
    
    if mpsfilepath.endswith(".pkl"):
        with open(mpsfilepath, "rb") as infile:
            mps = pickle.load(infile)

    elif mpsfilepath.endswith(".h5"):
        h5filepath = os.path.join(config.mps_directory, "%s" % (args.mpsfilename))
        h5filehandle = h5py.File(h5filepath, "r")
        h5group = h5filehandle["/"]
        h5tenpyloader = tenpy.tools.hdf5_io.Hdf5Loader(h5filehandle)
        mps = tenpy.networks.mps.MPS.from_hdf5(h5tenpyloader, h5group, "/")

    marginalsize = args.marginalsize
    which = args.which

    systemsize = mps.L
        
    if which == 'all' or which == None:
        selectsites_list = itertools.combinations(
            range(systemsize), marginalsize)
    else:
        selectsites_list = [eval(item) for item in which.split(";")]

    logging.info("sites_set_list = %s" % selectsites_list)

    bonddim = np.max(mps.chi)

    uuid_string = args.mpsfilename.split(".")[0]
    logging.info("uuid_string = %s" % (uuid_string))

    #filename_mps_df = os.path.join(
    #        config.mps_directory, "%s_mpsHistory.pkl" % (uuid_string,))

    READ_FLAGS = "rb"

    #with open(filename_mps_df, READ_FLAGS) as iofile:
    #    df_mps_all = pickle.load(iofile)

    ## This is a bit of a hack. `df_mps_all` is a `numpy.ndarray` of `pandas.DataFrame` with only
    ## one element. I may need to change the way I save the tenpy.networks.site.mps.MPS history.
    #df_mps_all = df_mps_all[0]

    #logging.info("df_mps_all = \n%s" % (df_mps_all,))

    #ix_time_list = df_mps_all["ix_time"].sort_values().unique()
    #time_list = df_mps_all["time"].sort_values().unique()

    #logging.info("ix_time_list = %s" % ix_time_list)

    logging.info("Evaluating %d-spin reduced density operators" % (marginalsize))

    walltime_begin = time.time()
    rows_reduced_dm = []
    #for row_iteration in df_mps_all.itertuples():
    for sites_sel in selectsites_list:

        #logging.info("Reduced state from bonddim = %d for %s in %d at ix_time = %d" \
        #                 % (bonddim, sites_sel, systemsize, row_iteration.ix_time))

        logging.info("Reduced state from bonddim = %d for %s in %d" \
                         % (bonddim, sites_sel, systemsize,))
        #mps = row_iteration.mps
        logging.info("mps.chi = %s" % (mps.chi))
        rho = mps.get_rho_segment(sites_sel)

        row_created = {
            "bonddim": bonddim,
            "sites_sel": sites_sel,
            "sites_sel_int": tuple_to_bitstring(sites_sel, systemsize),
            "rho": rho,
        }

        rows_reduced_dm.append(row_created)

    df_reduced_dm = pandas.DataFrame(rows_reduced_dm)

    walltime_end:float = time.time()
    walltime_duration:float = walltime_end - walltime_begin
    logging.info("Time taken = %g s" % (walltime_duration))
    #logging.info("df_reduced_dm =\n%s" % (df_reduced_dm))

    WRITE_FLAGS = "wb"
    logging.info("Saving %d-spin marginals" % (marginalsize))

    filename_df = os.path.join(
            config.marginal_directory, "%s_%d-spin.pkl" % (uuid_string, marginalsize))

    with open(filename_df, WRITE_FLAGS) as iofile:
        pickle.dump(df_reduced_dm, iofile)

    logging.info("Finished saving %d-spin marginals" % (marginalsize))
