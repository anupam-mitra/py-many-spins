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

from manybody_backends.qutip.timeevolution import local_marginal_density_matrix


def tuple_to_bitstring(t, length_bitstring):
    """Return an integer bitstring label for the selected site tuple."""
    bitlist = ['0' for _ in range(length_bitstring)]

    for ix in t:
        bitlist[ix] = '1'

    bitlist.reverse()
    bitstring = ''.join([b for b in bitlist])

    return int(bitstring, base=2)


def lookup_state_metadata(uuid_string):
    """Find saved metadata for a state UUID from the run index pickles."""
    for filename in os.listdir(config.mps_directory):
        if not filename.endswith("_index.pkl"):
            continue

        filepath = os.path.join(config.mps_directory, filename)
        with open(filepath, "rb") as infile:
            df_states = pickle.load(infile)

        if not isinstance(df_states, pandas.DataFrame):
            continue

        if "uuid_str" not in df_states.columns:
            continue

        matches = df_states.loc[df_states["uuid_str"] == uuid_string]
        if len(matches) == 0:
            continue

        return matches.iloc[0].to_dict()

    return {}


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(
        prog="qutip_marginalize",
        description="Calculates local marginals from a pickled QuTiP state.",
        epilog="",
    )

    argument_parser.add_argument("--mpsfilename", type=str)
    argument_parser.add_argument("--marginalsize", type=int)
    argument_parser.add_argument("--which", type=str)

    args = argument_parser.parse_args()
    logging.info("Input arguments = %s" % vars(args))

    statefilepath = os.path.join(config.mps_directory, "%s" % args.mpsfilename)
    logging.info("statefilepath = %s" % (statefilepath,))

    with open(statefilepath, "rb") as infile:
        state = pickle.load(infile)

    marginalsize = args.marginalsize
    which = args.which

    systemsize = len(state.dims[0])

    if which == 'all' or which is None:
        selectsites_list = itertools.combinations(range(systemsize), marginalsize)
    else:
        selectsites_list = [eval(item) for item in which.split(";")]

    logging.info("sites_set_list = %s" % (selectsites_list,))

    uuid_string = args.mpsfilename.split(".")[0]
    state_metadata = lookup_state_metadata(uuid_string)
    logging.info("uuid_string = %s" % (uuid_string,))

    logging.info("Evaluating %d-spin reduced density operators" % (marginalsize,))

    walltime_begin = time.time()
    rows_reduced_dm = []
    for sites_sel in selectsites_list:
        logging.info(
            "Reduced state for %s in %d"
            % (sites_sel, systemsize)
        )

        rho = local_marginal_density_matrix(state, sites_sel)

        row_created = {
            "bonddim": state_metadata.get("bonddim"),
            "ix_time": state_metadata.get("ix_time"),
            "time": state_metadata.get("time"),
            "sites_sel": sites_sel,
            "sites_sel_int": tuple_to_bitstring(sites_sel, systemsize),
            "rho": rho,
        }

        rows_reduced_dm.append(row_created)

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
