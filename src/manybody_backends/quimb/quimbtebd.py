import numpy as np
import pandas
import quimb.tensor as qtn
from typing import Any, Tuple

from manybody_backends.quimb.options import (
    split_options,
    tebd_evolution_options,
    tebd_init_options,
)
from manybody_util.history import history_record

 
def _n_sites(mps: Any) -> int:
    """
    Get the number of sites in a Quimb MPS.

    Parameters
    ----------
    mps : Any
        The Quimb MPS.

    Returns
    -------
    int
        The number of sites.
    """
    if hasattr(mps, "nsites"):
        return mps.nsites
    return mps.L

 
def max_bond_dimension(mps: Any) -> int:
    """
    Return the maximum bond dimension of a Quimb MPS.

    Parameters
    ----------
    mps : Any
        The Quimb MPS.

    Returns
    -------
    int
        The maximum bond dimension.
    """
    return mps.max_bond()

 
def local_marginal_density_matrix(mps: Any, locations: Tuple[int, ...]) -> Any:
    """
    Return a dense reduced density matrix for selected MPS sites.

    Parameters
    ----------
    mps : Any
        The Quimb MPS.
    locations : Tuple[int, ...]
        The site indices.

    Returns
    -------
    Any
        The reduced density matrix.
    """
    return mps.partial_trace_to_mpo(keep=tuple(locations)).to_dense()

 
def spinhalf_state(ang_polar: float, ang_azimuth: float) -> list[complex]:
    """
    Create a spin-half pure state vector for Quimb product MPS input.

    Parameters
    ----------
    ang_polar : float
        Polar angle.
    ang_azimuth : float
        Azimuthal angle.

    Returns
    -------
    list[complex]
        The spin-half state vector.
    """
    return [
        np.cos(ang_polar / 2),
        np.sin(ang_polar / 2) * np.exp(1j * ang_azimuth),
    ]

 
class TEBDWrapper:
    """
    Quimb TEBD wrapper mirroring the TenPy workflow API.

    Attributes
    ----------
    hamiltonian_builder : Any
        The Hamiltonian builder.
    initial_mps : Any
        The initial MPS state.
    tlist : np.ndarray
        The time grid.
    trotter_params : dict[str, Any]
        The Trotter evolution parameters.
    trunc_params : dict[str, Any]
        The truncation parameters.
    n_sites : int
        The number of sites in the system.
    split_opts : dict[str, Any]
        The Quimb split options.
    mps_list : list[Any]
        The history of evolved MPS states.
    rows : list[dict[str, Any]]
        The evolution records.
    """

    def __init__(
        self,
        hamiltonian_builder: Any,
        initial_mps: Any,
        tlist: np.ndarray,
        trotter_params: dict[str, Any],
        trunc_params: dict[str, Any],
    ) -> None:
        """
        Initialize the TEBDWrapper.

        Parameters
        ----------
        hamiltonian_builder : Any
            The Hamiltonian builder.
        initial_mps : Any
            The initial MPS state.
        tlist : np.ndarray
            The time grid.
        trotter_params : dict[str, Any]
            The Trotter evolution parameters.
        trunc_params : dict[str, Any]
            The truncation parameters.
        """
        self.hamiltonian_builder = hamiltonian_builder
        self.initial_mps = initial_mps
        self.tlist = tlist
        self.trotter_params = trotter_params
        self.trunc_params = trunc_params
        self.n_sites = _n_sites(initial_mps)

        self.split_opts = split_options(trunc_params)

        self.mps_list = []
        self.rows = []

    def evolve(self) -> None:
        """
        Run the TEBD evolution and store the MPS history.
        """
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

    def get_mps_history_df(self) -> tuple[pandas.DataFrame, list[Any]]:
        """
        Return the evolution records as a DataFrame and the MPS history.

        Returns
        -------
        tuple[pandas.DataFrame, list[Any]]
            A tuple containing the records DataFrame and the MPS history.
        """
        if not hasattr(self, "df"):
            self.df = pandas.DataFrame(self.rows)

        return self.df, self.mps_list

