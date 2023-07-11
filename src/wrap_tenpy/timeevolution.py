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
#
#
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
#
#
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

        self.rows = []

    def evolve(self):

        self.rows += [{
            "ix_time": 0,
            "time": 0,
            "mps": self.mps_in.copy(),
            #"walltime": time.time()
        }]

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
                "mps": mps_current.copy(),
                #"walltime": time.time()
            }

            self.rows += [row]

    def get_mps_history_df(self):

        if not hasattr(self, "df"):
            self.df = pandas.DataFrame(self.rows)

        return self.df


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
#
#
######################################################################################
######################################################################################

if __name__ == '__main__':
    n_spins = 14

    theta = pi/2
    phi = pi/2

    j_int = 1
    b_parallel = 1
    b_perp = 1

    S=1/2

    site = tenpy.networks.site.SpinHalfSite(conserve=None)

    sfim_parameters = {
        "L": n_spins,
        "J": j_int,
        "g": b_perp,
        "bc_MPS": "finite"
    }

    sfim = tenpy.models.tf_ising.TFIChain(sfim_parameters)
    sfim.add_onsite(b_parallel, 0, "Sx")

    t_initial = 0
    t_final = n_spins * n_spins / j_int
    n_steps = 256

    t_list = np.linspace(t_initial, t_final, n_steps)
