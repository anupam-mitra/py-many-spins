import numpy as np
import pandas
import quimb.tensor as qtn

from manybody_backends.quimb.options import (
    split_options,
    tebd_evolution_options,
    tebd_init_options,
)
from manybody_util.history import history_record


def _n_sites(mps):
    if hasattr(mps, "nsites"):
        return mps.nsites
    return mps.L


def max_bond_dimension(mps):
    """Return the maximum bond dimension of a Quimb MPS."""
    return mps.max_bond()


def local_marginal_density_matrix(mps, locations):
    """Return a dense reduced density matrix for selected MPS sites."""
    return mps.partial_trace_to_mpo(keep=tuple(locations)).to_dense()


def spinhalf_state(ang_polar, ang_azimuth):
    """Create a spin-half pure state vector for Quimb product MPS input."""
    return [
        np.cos(ang_polar / 2),
        np.sin(ang_polar / 2) * np.exp(1j * ang_azimuth),
    ]


class TEBDWrapper:
    """Quimb TEBD wrapper mirroring the TenPy workflow API."""

    def __init__(self, hamiltonian_builder, initial_mps, tlist,
        trotter_params, trunc_params):

        self.hamiltonian_builder = hamiltonian_builder
        self.initial_mps = initial_mps
        self.tlist = tlist
        self.trotter_params = trotter_params
        self.trunc_params = trunc_params
        self.n_sites = _n_sites(initial_mps)

        self.split_opts = split_options(trunc_params)

        self.mps_list = []
        self.rows = []

    def evolve(self):
        self.hamiltonian_local = self.hamiltonian_builder.build_local_ham(self.n_sites)
        self.tebd = qtn.TEBD(
            self.initial_mps,
            self.hamiltonian_local,
            split_opts=self.split_opts,
            **tebd_init_options(self.trotter_params),
        )

        self.mps_list = []
        self.rows = []
        for ix_time, state in enumerate(
            self.tebd.at_times(
                self.tlist,
                **tebd_evolution_options(self.trotter_params),
            )
        ):
            self.mps_list.append(state.copy() if hasattr(state, "copy") else state)
            self.rows.append(
                history_record(
                    ix_time,
                    self.tlist[ix_time],
                    {"bonddim": self.trunc_params.get("chi_max")},
                )
            )

        self.df = pandas.DataFrame(self.rows)

    def get_mps_history_df(self):
        if not hasattr(self, "df"):
            self.df = pandas.DataFrame(self.rows)

        return self.df, self.mps_list
