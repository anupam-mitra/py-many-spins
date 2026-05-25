import numpy as np
import pandas
import qutip
from typing import Any, Tuple, List, Dict

from manybody_util.history import history_records

 
def spinhalf_state(ang_polar: float, ang_azimuth: float) -> Any:
    """
    Create a single-spin pure state pointing along the given angles.

    Parameters
    ----------
    ang_polar : float
        Polar angle.
    ang_azimuth : float
        Azimuthal angle.

    Returns
    -------
    Any
        The QuTiP spin-coherent state (ket).
    """
    return qutip.spin_coherent(j=1 / 2, theta=ang_polar, phi=ang_azimuth, type="ket")

 
def manyspin_product_state(n_spins: int, ang_polar: float, ang_azimuth: float) -> Any:
    """
    Create a product state of identically prepared spin-half kets.

    Parameters
    ----------
    n_spins : int
        The number of spins.
    ang_polar : float
        Polar angle.
    ang_azimuth : float
        Azimuthal angle.

    Returns
    -------
    Any
        The product state as a QuTiP Qobj.
    """
    one_spin_state = spinhalf_state(ang_polar, ang_azimuth)
    return qutip.tensor([one_spin_state] * n_spins)

 
def local_marginal_density_matrix(state: Any, locations: Tuple[int, ...]) -> Any:
    """
    Return the local marginal density matrix for the selected sites.

    Parameters
    ----------
    state : Any
        The state (ket or density operator).
    locations : Tuple[int, ...]
        The site indices.

    Returns
    -------
    Any
        The reduced density matrix as a QuTiP Qobj.
    """
    return state.ptrace(list(locations))



def _history_rows(tlist: np.ndarray) -> pandas.DataFrame:
    """
    Create history records for a given time list.

    Parameters
    ----------
    tlist : np.ndarray
        The time grid.

    Returns
    -------
    pandas.DataFrame
        The history records DataFrame.
    """
    return pandas.DataFrame(history_records(tlist))

 
def _state_copies(states: List[Any]) -> List[Any]:
    """
    Create copies of QuTiP states.

    Parameters
    ----------
    states : List[Any]
        The list of QuTiP states.

    Returns
    -------
    List[Any]
        The list of copied states.
    """
    return [state.copy() for state in states]



def _solver_options(options: dict[str, Any] | None = None, keep_runs_results: bool = False) -> dict[str, Any]:
    """
    Create solver options for QuTiP evolution.

    Parameters
    ----------
    options : dict[str, Any] | None, optional
        Base solver options, by default None.
    keep_runs_results : bool, optional
        Whether to keep results for each run, by default False.

    Returns
    -------
    dict[str, Any]
        The final solver options.
    """
    solver_options = dict(options or {})
    solver_options["store_states"] = True
    solver_options.setdefault("progress_bar", False)
    if keep_runs_results:
        solver_options["keep_runs_results"] = True
    return solver_options



def _require_state_count(states: List[Any], tlist: np.ndarray, solver_name: str) -> None:
    """
    Verify that the number of evolved states matches the requested time list.

    Parameters
    ----------
    states : List[Any]
        The evolved states.
    tlist : np.ndarray
        The time grid.
    solver_name : str
        The name of the solver for the error message.

    Raises
    ------
    ValueError
        If state count mismatch occurs.
    """
    if len(states) != len(tlist):
        raise ValueError(
            "%s returned %d states for %d requested times"
            % (solver_name, len(states), len(tlist))
        )



def _mcsolve_metadata(result: Any) -> dict[str, Any]:
    """
    Extract metadata from a QuTiP mcsolve result.

    Parameters
    ----------
    result : Any
        The result object from qutip.mcsolve.

    Returns
    -------
    dict[str, Any]
        The extracted metadata.
    """
    return {
        "ntraj": int(getattr(result, "num_trajectories", 0)),
        "col_times": [
            [float(time_value) for time_value in trajectory]
            for trajectory in getattr(result, "col_times", [])
        ],
        "col_which": [
            [int(which) for which in trajectory]
            for trajectory in getattr(result, "col_which", [])
        ],
        "seeds": ["%s" % seed for seed in getattr(result, "seeds", [])],
    }




def solve_state_history(hamiltonian: Any, initial_state: Any, tlist: np.ndarray, metadata: dict[str, Any] | None = None) -> Tuple[pandas.DataFrame, List[Any]]:
    """
    Evolve a QuTiP state and return the workflow index plus state history.

    Parameters
    ----------
    hamiltonian : Any
        The QuTiP Hamiltonian.
    initial_state : Any
        The initial state.
    tlist : np.ndarray
        The time grid.
    metadata : dict[str, Any] | None, optional
        Evolution metadata, by default None.

    Returns
    -------
    Tuple[pandas.DataFrame, List[Any]]
        A tuple containing the records DataFrame and the list of evolved states.
    """
    result = qutip.sesolve(hamiltonian, initial_state, tlist)
    states = list(result.states)
    _require_state_count(states, tlist, "QuTiP sesolve")
    return _history_rows(tlist), _state_copies(states)



def solve_master_history(
    hamiltonian: Any,
    initial_state: Any,
    tlist: np.ndarray,
    collapse_ops: List[Any] | None = None,
    options: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Tuple[pandas.DataFrame, List[Any]]:
    """
    Evolve a QuTiP density operator history using ``qutip.mesolve``.

    Parameters
    ----------
    hamiltonian : Any
        The QuTiP Hamiltonian.
    initial_state : Any
        The initial state (ket or density operator).
    tlist : np.ndarray
        The time grid.
    collapse_ops : List[Any] | None, optional
        The collapse operators, by default None.
    options : dict[str, Any] | None, optional
        Solver options, by default None.
    metadata : dict[str, Any] | None, optional
        Evolution metadata, by default None.

    Returns
    -------
    Tuple[pandas.DataFrame, List[Any]]
        A tuple containing the records DataFrame and the list of evolved states.
    """
    result = qutip.mesolve(
        hamiltonian,
        initial_state,
        tlist,
        c_ops=list(collapse_ops or ()),
        options=_solver_options(options),
    )
    states = list(result.states)
    _require_state_count(states, tlist, "QuTiP mesolve")
    return _history_rows(tlist), _state_copies(states)



def solve_monte_carlo_history(
    hamiltonian: Any,
    initial_state: Any,
    tlist: np.ndarray,
    collapse_ops: List[Any] | None = None,
    ntraj: int = 500,
    seeds: List[int] | None = None,
    target_tol: float | None = None,
    timeout: float | None = None,
    options: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Tuple[pandas.DataFrame, List[Any], dict[str, Any]]:
    """
    Evolve averaged QuTiP MCWF states using ``qutip.mcsolve``.

    Parameters
    ----------
    hamiltonian : Any
        The QuTiP Hamiltonian.
    initial_state : Any
        The initial state.
    tlist : np.ndarray
        The time grid.
    collapse_ops : List[Any] | None, optional
        The collapse operators, by default None.
    ntraj : int, optional
        The number of trajectories, by default 500.
    seeds : List[int] | None, optional
        Random seeds, by default None.
    target_tol : float | None, optional
        Target tolerance, by default None.
    timeout : float | None, optional
        Timeout in seconds, by default None.
    options : dict[str, Any] | None, optional
        Solver options, by default None.
    metadata : dict[str, Any] | None, optional
        Evolution metadata, by default None.

    Returns
    -------
    Tuple[pandas.DataFrame, List[Any], dict[str, Any]]
        A tuple containing the records DataFrame, the list of averaged states, and the metadata.
    """
    result = qutip.mcsolve(
        hamiltonian,
        initial_state,
        tlist,
        c_ops=list(collapse_ops or ()),
        ntraj=int(ntraj),
        seeds=seeds,
        target_tol=target_tol,
        timeout=timeout,
        options=_solver_options(options, keep_runs_results=True),
    )
    states = list(result.average_states)
    _require_state_count(states, tlist, "QuTiP mcsolve")
    return _history_rows(tlist), _state_copies(states), _mcsolve_metadata(result)

