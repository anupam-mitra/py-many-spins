import numpy as np
import pandas
from tenpy.algorithms.mpo_evolution import ExpMPOEvolution
from tenpy.algorithms.tebd import TEBDEngine
from tenpy.algorithms.tdvp import TwoSiteTDVPEngine
from tenpy.networks.mps import MPS
from typing import Any, Tuple, Type

from manybody_util.history import history_record

 
def spinhalf_state(ang_polar: float, ang_azimuth: float) -> np.ndarray:
    """
    Create a spin-half pure state vector for TenPy product MPS input.

    Parameters
    ----------
    ang_polar : float
        Polar angle.
    ang_azimuth : float
        Azimuthal angle.

    Returns
    -------
    np.ndarray
        The spin-half state vector.
    """
    return np.asarray([
        np.cos(ang_polar / 2),
        np.sin(ang_polar / 2) * np.exp(1j * ang_azimuth),
    ], dtype=complex)

 
def manyspin_product_mps(
    sites: Any, ang_polar: float, ang_azimuth: float, bc: str = "finite", unit_cell_width: int | None = None
) -> MPS:
    """
    Create an MPS product state of identically prepared spin-half sites.

    Parameters
    ----------
    sites : Any
        The TenPy sites object.
    ang_polar : float
        Polar angle.
    ang_azimuth : float
        Azimuthal angle.
    bc : str, optional
        Boundary conditions, by default "finite".
    unit_cell_width : int, optional
        Unit cell width, by default None.

    Returns
    -------
    MPS
        The created MPS product state.
    """
    state = spinhalf_state(ang_polar, ang_azimuth)
    kwargs = {
        "bc": bc,
        "dtype": complex,
    }
    if unit_cell_width is not None:
        kwargs["unit_cell_width"] = unit_cell_width
    return MPS.from_product_state(sites, [state] * len(sites), **kwargs)

 
def max_bond_dimension(mps: MPS) -> int:
    """
    Return the maximum bond dimension of a TenPy MPS.

    Parameters
    ----------
    mps : MPS
        The TenPy MPS.

    Returns
    -------
    int
        The maximum bond dimension.
    """
    return max(mps.chi) if mps.chi else 1

 
def local_marginal_density_matrix(mps: MPS, locations: Tuple[int, ...]) -> Any:
    """
    Return the reduced density matrix for selected MPS sites.

    Parameters
    ----------
    mps : MPS
        The TenPy MPS.
    locations : Tuple[int, ...]
        The site indices.

    Returns
    -------
    Any
        The reduced density matrix.
    """
    return mps.get_rho_segment(tuple(locations))

 
def _clean_dict(params: dict[str, Any] | None) -> dict[str, Any]:
    """
    Remove None values from a dictionary.

    Parameters
    ----------
    params : dict[str, Any] | None
        The dictionary to clean.

    Returns
    -------
    dict[str, Any]
        The cleaned dictionary.
    """
    return {key: value for key, value in (params or {}).items() if value is not None}

 
def _engine_class(algorithm: str) -> Type:
    """
    Get the TenPy engine class for a given algorithm.

    Parameters
    ----------
    algorithm : str
        The algorithm name ('TEBD', 'TDVP', or 'ExpMPO').

    Returns
    -------
    Type
        The TenPy engine class.

    Raises
    ------
    ValueError
        If the algorithm is unsupported.
    """
    if algorithm == "TEBD":
        return TEBDEngine
    if algorithm == "TDVP":
        return TwoSiteTDVPEngine
    if algorithm == "ExpMPO":
        return ExpMPOEvolution
    raise ValueError("unsupported TenPy evolution algorithm %r" % (algorithm,))

 
def _engine_defaults(algorithm: str) -> dict[str, Any]:
    """
    Get default engine options for a given algorithm.

    Parameters
    ----------
    algorithm : str
        The algorithm name.

    Returns
    -------
    dict[str, Any]
        The default options.
    """
    if algorithm == "ExpMPO":
        return {
            "approximation": "II",
            "compression_method": "SVD",
            "order": 2,
        }
    return {}

 
def _engine_options(algorithm: str, evolution_params: dict[str, Any] | None, trunc_params: dict[str, Any] | None) -> dict[str, Any]:
    """
    Build engine options for the TenPy evolution.

    Parameters
    ----------
    algorithm : str
        The algorithm name.
    evolution_params : dict[str, Any] | None
        Evolution parameters.
    trunc_params : dict[str, Any] | None
        Truncation parameters.

    Returns
    -------
    dict[str, Any]
        The final engine options.
    """
    options = {
        **_engine_defaults(algorithm),
        **_clean_dict(evolution_params),
    }
    options.pop("dt", None)
    options.pop("N_steps", None)
    if algorithm == "TDVP":
        options.pop("order", None)
    options["trunc_params"] = _clean_dict(trunc_params)
    return options

 
def _substeps(evolution_params: dict[str, Any] | None) -> int:
    """
    Determine the number of substeps for evolution.

    Parameters
    ----------
    evolution_params : dict[str, Any] | None
        Evolution parameters.

    Returns
    -------
    int
        The number of substeps.

    Raises
    ------
    ValueError
        If N_steps is less than 1.
    """
    n_substeps = int((evolution_params or {}).get("N_steps", 1))
    if n_substeps < 1:
        raise ValueError("N_steps must be at least 1")
    return n_substeps

 
def solve_mps_history(
    model: Any,
    initial_mps: MPS,
    tlist: list[float] | np.ndarray,
    algorithm: str = "TEBD",
    evolution_params: dict[str, Any] | None = None,
    trunc_params: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> tuple[pandas.DataFrame, list[MPS]]:
    """
    Evolve a TenPy MPS and return the workflow index plus MPS history.

    Parameters
    ----------
    model : Any
        The TenPy model.
    initial_mps : MPS
        The initial MPS state.
    tlist : list[float] | np.ndarray
        The time grid.
    algorithm : str, optional
        The evolution algorithm, by default "TEBD".
    evolution_params : dict[str, Any] | None, optional
        Additional evolution parameters, by default None.
    trunc_params : dict[str, Any] | None, optional
        Truncation parameters, by default None.
    metadata : dict[str, Any] | None, optional
        Simulation metadata, by default None.

    Returns
    -------
    tuple[pandas.DataFrame, list[MPS]]
        A tuple containing the evolution records (as a DataFrame) and the MPS history.

    Raises
    ------
    ValueError
        If tlist is empty.
    """
    tlist = np.asarray(tlist, dtype=float)
    if len(tlist) == 0:
        raise ValueError("tlist must contain at least one time")

    metadata = metadata or {}
    trunc_params = trunc_params or {}
    bonddim = metadata.get("bonddim", trunc_params.get("chi_max"))

    mps = initial_mps.copy()
    mps_list = [mps.copy()]
    rows = [history_record(0, tlist[0], {"bonddim": bonddim})]

    engine = _engine_class(algorithm)(
        mps,
        model,
        _engine_options(algorithm, evolution_params, trunc_params),
    )
    n_substeps = _substeps(evolution_params)

    for ix_time in range(1, len(tlist)):
        interval = tlist[ix_time] - tlist[ix_time - 1]
        engine.run_evolution(n_substeps, interval / n_substeps)
        mps_list.append(mps.copy())
        rows.append(history_record(ix_time, tlist[ix_time], {"bonddim": bonddim}))

    return pandas.DataFrame(rows), mps_list

