import itertools
import os
import pickle
import time
import argparse
import logging

import pandas

logging.basicConfig(
    format='%(asctime)s: %(levelname)s: %(message)s',
    level=logging.DEBUG,
)

import sys
sys.path.append("../src")
logging.info(sys.path)

import config

from manybody_backends.quimb.quimbtebd import (
    local_marginal_density_matrix,
    max_bond_dimension,
)


def tuple_to_bitstring(t, length_bitstring):
    """Return an integer bitstring label for the selected site tuple."""
    bitlist = ['0' for _ in range(length_bitstring)]

    for ix in t:
        bitlist[ix] = '1'

    bitlist.reverse()
    bitstring = ''.join([b for b in bitlist])

    return int(bitstring, base=2)


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(
        prog="quimb_marginalize",
        description="Calculates local marginals from a pickled Quimb MPS.",
        epilog="",
    )

    argument_parser.add_argument("--mpsfilename", type=str)
    argument_parser.add_argument("--marginalsize", type=int)
    argument_parser.add_argument("--which", type=str)

    args = argument_parser.parse_args()
    logging.info("Input arguments = %s" % vars(args))

    mpsfilepath = os.path.join(config.mps_directory, "%s" % args.mpsfilename)
    logging.info("mpsfilepath = %s" % (mpsfilepath,))

    with open(mpsfilepath, "rb") as infile:
        mps = pickle.load(infile)

    marginalsize = args.marginalsize
    which = args.which

    systemsize = mps.L

    if which == 'all' or which is None:
        selectsites_list = itertools.combinations(range(systemsize), marginalsize)
    else:
        selectsites_list = [eval(item) for item in which.split(";")]

    logging.info("sites_set_list = %s" % (selectsites_list,))

    bonddim = max_bond_dimension(mps)
    uuid_string = args.mpsfilename.split(".")[0]
    logging.info("uuid_string = %s" % (uuid_string,))

    logging.info("Evaluating %d-spin reduced density operators" % (marginalsize,))

    walltime_begin = time.time()
    rows_reduced_dm = []
    for sites_sel in selectsites_list:
        logging.info(
            "Reduced state from bonddim = %d for %s in %d"
            % (bonddim, sites_sel, systemsize)
        )

        rho = local_marginal_density_matrix(mps, sites_sel)

        rows_reduced_dm.append({
            "bonddim": bonddim,
            "sites_sel": sites_sel,
            "sites_sel_int": tuple_to_bitstring(sites_sel, systemsize),
            "rho": rho,
        })

    df_reduced_dm = pandas.DataFrame(rows_reduced_dm)

    walltime_end = time.time()
    walltime_duration = walltime_end - walltime_begin
    logging.info("Time taken = %g s" % (walltime_duration,))

    logging.info("Saving %d-spin marginals" % (marginalsize,))

    filename_df = os.path.join(
        config.marginal_directory,
        "%s_%d-spin.pkl" % (uuid_string, marginalsize),
    )

    with open(filename_df, "wb") as iofile:
        pickle.dump(df_reduced_dm, iofile)

    logging.info("Finished saving %d-spin marginals" % (marginalsize,))
