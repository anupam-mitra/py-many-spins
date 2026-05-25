import itertools
from typing import Any, Tuple, List, Dict

import numpy as np
import pandas
from quspin.basis import spin_basis_1d

from manybody_util.history import history_records

 
def spinhalf_basis(n_spins: int) -> Any:
    """
    Create a full spin-half QuSpin basis using Pauli operators.

    Parameters
    ----------
    n_spins : int
        The number of spins.

    Returns
    -------
    Any
        The QuSpin spin_basis_1d object.
    """
    return spin_basis_1d(L=n_spins, S="1/2", pauli=1)

 
def spinhalf_state(ang_polar: float, ang_azimuth: float) -> Dict[str, complex]:
    """
    Create single-site amplitudes in QuSpin's spin string convention.

    Parameters
    ----------
    ang_polar : float
        Polar angle.
    ang_azimuth : float
        Azimuthal angle.

    Returns
    -------
    Dict[str, complex]
        The amplitudes for '0' and '1' states.
    """
    return {
        "1": np.cos(ang_polar / 2),
        "0": np.sin(ang_polar / 2) * np.exp(1j * ang_azimuth),
    }

 
def manyspin_product_state(n_spins: int, ang_polar: float, ang_azimuth: float, basis: Any = None) -> np.ndarray:
    """
    Create a dense product-state vector in a full QuSpin spin-half basis.

    Parameters
    ----------
    n_spins : int
        The number of spins.
    ang_polar : float
        Polar angle.
    ang_azimuth : float
        Azimuthal angle.
    basis : Any, optional
        The QuSpin basis. If None, a default basis is created, by default None.

    Returns
    -------
    np.ndarray
        The product-state vector.
    """
    if basis is None:
        basis = spinhalf_basis(n_spins)

    one_spin_state = spinhalf_state(ang_polar, ang_azimuth)
    state = np.zeros(basis.Ns, dtype=np.complex128)
    for bits in itertools.product(("0", "1"), repeat=n_spins):
        bitstring = "".join(bits)
        amplitude = 1.0 + 0.0j
        for bit in bits:
            amplitude *= one_spin_state[bit]
        state[basis.index(bitstring)] = amplitude

    return state

 
def local_marginal_density_matrix(state: np.ndarray, locations: Tuple[int, ...], basis: Any) -> np.ndarray:
    """
    Return the dense reduced density matrix for selected sites.

    Parameters
    ----------
    state : np.ndarray
        The state vector.
    locations : Tuple[int, ...]
        The site indices.
    basis : Any
        The QuSpin basis.

    Returns
    -------
    np.ndarray
        The reduced density matrix.
    """
    return basis.partial_trace(
        state,
        sub_sys_A=list(locations),
        return_rdm="A",
        sparse=False,
    )

 
def solve_state_history(hamiltonian: Any, initial_state: np.ndarray, tlist: np.ndarray, metadata: dict[str, Any] | None = None) -> Tuple[pandas.DataFrame, List[np.ndarray]]:
    """
    Evolve a QuSpin state and return the workflow index plus state history.

    Parameters
    ----------
    hamiltonian : Any
        The QuSpin Hamiltonian.
    initial_state : np.ndarray
        The initial state vector.
    tlist : np.ndarray
        The time grid.
    metadata : dict[str, Any] | None, optional
        Evolution metadata, by default None.

    Returns
    -------
    Tuple[pandas.DataFrame, List[np.ndarray]]
        A tuple containing the records DataFrame and the list of evolved states.

    Raises
    ------
    ValueError
        If QuSpin returns a different number of states than requested times.
    """
    evolved_states = hamiltonian.evolve(
        initial_state,
        tlist[0],
        tlist,
        eom="SE",
        iterate=True,
    )
    states = [np.asarray(state, dtype=np.complex128).copy() for state in evolved_states]
    if len(states) != len(tlist):
        raise ValueError(
            "QuSpin returned %d states for %d requested times" % (len(states), len(tlist))
        )

    return pandas.DataFrame(history_records(tlist)), states

