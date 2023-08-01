import numpy as np

import pandas
import pickle
import os
import time
import uuid

import tenpy

import logging

from numpy import pi, sqrt, cos, sin, exp

######################################################################################
# TODO Move to a separate file
def spinhalf_state (ang_polar, ang_azimuth):
    """Creates a spin 1/2 pure state pointing
    along directions `ang_polar` and `ang_azimuth`.
    """
    return np.asarray([cos(ang_polar/2),\
           sin(ang_polar/2) * exp(1j * ang_azimuth)])
######################################################################################

######################################################################################
# TODO Move to a separate file
def mps_spincoherent_state (sites, ang_polar, ang_azimuth, cyclic=False):
    """Creates a matrix product state representation of a spin
    coherent state pointing along directions `ang_polar` and
    `ang_azimuth`.
    Boundary conditions are set by the flag `cyclic`
    """
    one_spin_state = spinhalf_state(ang_polar, ang_azimuth)
    n_spins = len(sites)
    site = tenpy.networks.site.SpinHalfSite(conserve=None)
    if cyclic:
        bc = "infinite"
    else:
        bc = "finite"

    mps = tenpy.networks.mps.MPS.from_product_state(
        sites, [one_spin_state]*n_spins, bc)
    return mps
#
#
#####################################################################################

######################################################################################
# TODO Move to a separate file
def select_central_sites(n, m):
    """
    Selects the `m` central contiguous sites
    from `n sites labeled `0, 1, ..., (n-1)`.

    In an even system size, that is when `n % 2 == 0`,
    the sites chosen are the smaller ones, that is closer
    to the left edge at `0` than to the right edge at `n-1`
    """

    locations = tuple(range((n - m + 1)//2, (n + m + 1)//2))

    return locations
#####################################################################################

######################################################################################
######################################################################################
# TODO Move to a separate file
# TODO Implement saving and indexing functionality
import pandas
import time

######################################################################################
######################################################################################
class TEBDWrapper:
    """
    Represents a wrapper around `tenpy.algorithms.tebd.TEBDEngine` to calculate the
    state represented as `tenpy.networks.mps.MPS` as a function of time.

    Parameters
    ----------
    model: tenpy.models.model.NearestNeighborModel
    Model used to generate the Hamiltonian

    mps_in: tenpy.networks.mps.MPS
    Initial state, represented as a matrix product state

    tlist: iterable
    List of times at which to save the state

    trotter_params: dict
    Dictionary with parameters for Trotter-Suzuki decomposition

    compress_params: dict
    Dictionary with parameters for compression of matrix product state tensors

    """
    def __init__(self, model, mps_in, tlist,
        trotter_params, trunc_params):

        self.model = model
        self.mps_in = mps_in
        self.tlist = tlist

        self.trotter_params = trotter_params
        self.compress_params = trunc_params

        self.tebd_params = {
            "trunc_params" : trunc_params,
            **trotter_params,
        }

        logging.info("Created TEBD object with \n tebd_params = %s" % (self.tebd_params))

        self.mps_list = []
        self.rows = []

    def evolve(self):
        """
        Calculate time evolution of the matrix product state.
        """
        self.rows += [{
            "ix_time": 0,
            "time": self.tlist[0],
            "bonddim":self.compress_params.get("chi_max"),
            "uuid_str": "%s" % uuid.uuid4(),
            "walltime": time.time()
        }]
        self.mps_list.append(self.mps_in)

        logging.info("Calculating state at self.tlist[ix_time=%d] = %g" % (0, self.tlist[0]))

        mps_current = self.mps_in.copy()

        for ix_time in range(1, len(self.tlist)):
            logging.info("Calculating state at self.tlist[ix_time=%d] = %g" % (ix_time, self.tlist[ix_time]))
            engine = tenpy.algorithms.tebd.TEBDEngine(
                mps_current, self.model, self.tebd_params)
            engine.run()

            row = {
                "ix_time": ix_time,
                "time": self.tlist[ix_time],
                "bonddim":self.compress_params.get("chi_max"),
                "uuid_str": "%s" % uuid.uuid4(),
                "walltime": time.time()
            }

            self.mps_list.append(mps_current.copy())
            self.rows.append(row)
            logging.info("mps.chi = %s" % (mps_current.chi))
            
    def get_mps_history_df(self):
        """
        Returns the mps history index in an object of type `pandas.DataFrame`
        """

        if not hasattr(self, "df"):
            self.df = pandas.DataFrame(self.rows)

        return self.df, self.mps_list
######################################################################################

######################################################################################
class TDVPWrapper:
    def __init__(self, model, mps_in, tlist,
        trunc_params, tdvp_params):

        self.model = model
        self.mps_in = mps_in
        self.tlist = tlist

        self.compress_params = trunc_params
        if tdvp_params is None:
            self.tdvp_params = {
                "trunc_params" : trunc_params,
            }
        else:
            self.tdvp_params = tdvp_params

        logging.info("Created TDVP object with \n tdvp_params = %s" % (self.tdvp_params))
        
        self.rows = []

    def evolve(self):

        self.rows += [{
            "ix_time": 0,
            "time": 0,
            "mps": self.mps_in,
            "walltime": time.time()
        }]
        
        logging.info("Calculating state at self.tlist[ix_time=%d] = %g" % (0, self.tlist[0]))

        mps_current = self.mps_in.copy()

        for ix_time in range(1, len(self.tlist)):
            logging.info("Calculating state at self.tlist[ix_time=%d] = %g" % (ix_time, self.tlist[ix_time]))
            engine = tenpy.algorithms.tdvp.TwoSiteTDVPEngine(
                mps_current, self.model, self.tdvp_params)
            engine.run()

            row = {
                "ix_time": ix_time,
                "time": self.tlist[ix_time],
                "mps": mps_current.copy(),
                "walltime": time.time()
            }

            self.rows += [row]

    def get_mps_history_df(self):

        if not hasattr(self, "df"):
            self.df = pandas.DataFrame(self.rows)

        return self.df
######################################################################################
