import argparse
import ast
import itertools
import logging
import os
import pickle
import sys
import time

import h5py
import pandas
from tenpy.networks.mps import MPS
from tenpy.tools.hdf5_io import Hdf5Loader

logging.basicConfig(
    format='%(asctime)s: %(levelname)s: %(message)s',
    level=logging.DEBUG,
)

sys.path.append("../src")
logging.info(sys.path)

import config

from manybody_backends.tenpy.timeevolution import (
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


def load_mps(filepath):
    """Load a TenPy MPS from pickle or TenPy HDF5."""
    if filepath.endswith(".pkl"):
        with open(filepath, "rb") as infile:
            return pickle.load(infile)

    if filepath.endswith(".h5"):
        with h5py.File(filepath, "r") as h5filehandle:
            loader = Hdf5Loader(h5filehandle)
            return MPS.from_hdf5(loader, h5filehandle["/"], "/")

    raise ValueError("unsupported TenPy MPS file extension: %s" % (filepath,))


def parse_site_selection(which, systemsize, marginalsize):
    """Parse the marginal site selector used by the dynamics scripts."""
    if which == 'all' or which is None:
        return list(itertools.combinations(range(systemsize), marginalsize))
    return [tuple(ast.literal_eval(item)) for item in which.split(";")]


def lookup_state_metadata(uuid_string):
    """Find saved metadata for an MPS UUID from the run index pickles."""
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
        if len(matches) != 0:
            return matches.iloc[0].to_dict()

    return {}


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(
        prog="tenpy_marginalize",
        description="Calculates local marginals from a pickled or HDF5 TenPy MPS.",
        epilog="",
    )
    argument_parser.add_argument("--mpsfilename", type=str, required=True)
    argument_parser.add_argument("--marginalsize", type=int, required=True)
    argument_parser.add_argument("--which", type=str)

    args = argument_parser.parse_args()
    logging.info("Input arguments = %s" % vars(args))

    mpsfilepath = os.path.join(config.mps_directory, "%s" % args.mpsfilename)
    logging.info("mpsfilepath = %s" % (mpsfilepath,))

    mps = load_mps(mpsfilepath)
    systemsize = mps.L
    selectsites_list = parse_site_selection(
        args.which,
        systemsize,
        args.marginalsize,
    )
    logging.info("sites_set_list = %s" % (selectsites_list,))

    bonddim = max_bond_dimension(mps)
    uuid_string = args.mpsfilename.split(".")[0]
    state_metadata = lookup_state_metadata(uuid_string)
    logging.info("uuid_string = %s" % (uuid_string,))

    logging.info("Evaluating %d-spin reduced density operators" % (args.marginalsize,))

    walltime_begin = time.time()
    rows_reduced_dm = []
    for sites_sel in selectsites_list:
        logging.info(
            "Reduced state from bonddim = %d for %s in %d"
            % (bonddim, sites_sel, systemsize)
        )

        rows_reduced_dm.append({
            "bonddim": state_metadata.get("bonddim", bonddim),
            "ix_time": state_metadata.get("ix_time"),
            "time": state_metadata.get("time"),
            "sites_sel": sites_sel,
            "sites_sel_int": tuple_to_bitstring(sites_sel, systemsize),
            "rho": local_marginal_density_matrix(mps, sites_sel),
        })

    df_reduced_dm = pandas.DataFrame(rows_reduced_dm)

    walltime_duration = time.time() - walltime_begin
    logging.info("Time taken = %g s" % (walltime_duration,))

    logging.info("Saving %d-spin marginals" % (args.marginalsize,))
    filename_df = os.path.join(
        config.marginal_directory,
        "%s_%d-spin.pkl" % (uuid_string, args.marginalsize),
    )

    with open(filename_df, "wb") as iofile:
        pickle.dump(df_reduced_dm, iofile)

    logging.info("Finished saving %d-spin marginals" % (args.marginalsize,))
