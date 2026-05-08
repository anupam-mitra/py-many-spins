import time
import uuid

import numpy as np
import pandas
from tenpy.algorithms.mpo_evolution import ExpMPOEvolution
from tenpy.algorithms.tebd import TEBDEngine
from tenpy.algorithms.tdvp import TwoSiteTDVPEngine
from tenpy.networks.mps import MPS


def spinhalf_state(ang_polar, ang_azimuth):
    """Create a spin-half pure state vector for TenPy product MPS input."""
    return np.asarray([
        np.cos(ang_polar / 2),
        np.sin(ang_polar / 2) * np.exp(1j * ang_azimuth),
    ], dtype=complex)


def manyspin_product_mps(sites, ang_polar, ang_azimuth, bc="finite", unit_cell_width=None):
    """Create an MPS product state of identically prepared spin-half sites."""
    state = spinhalf_state(ang_polar, ang_azimuth)
    kwargs = {
        "bc": bc,
        "dtype": complex,
    }
    if unit_cell_width is not None:
        kwargs["unit_cell_width"] = unit_cell_width
    return MPS.from_product_state(sites, [state] * len(sites), **kwargs)


def max_bond_dimension(mps):
    """Return the maximum bond dimension of a TenPy MPS."""
    return max(mps.chi) if mps.chi else 1


def local_marginal_density_matrix(mps, locations):
    """Return the reduced density matrix for selected MPS sites."""
    return mps.get_rho_segment(tuple(locations))


def _clean_dict(params):
    return {key: value for key, value in (params or {}).items() if value is not None}


def _engine_class(algorithm):
    if algorithm == "TEBD":
        return TEBDEngine
    if algorithm == "TDVP":
        return TwoSiteTDVPEngine
    if algorithm == "ExpMPO":
        return ExpMPOEvolution
    raise ValueError("unsupported TenPy evolution algorithm %r" % (algorithm,))


def _engine_defaults(algorithm):
    if algorithm == "ExpMPO":
        return {
            "approximation": "II",
            "compression_method": "SVD",
            "order": 2,
        }
    return {}


def _engine_options(algorithm, evolution_params, trunc_params):
    options = {
        **_engine_defaults(algorithm),
        **_clean_dict(evolution_params),
    }
    options.pop("dt", None)
    options.pop("N_steps", None)
    options["trunc_params"] = _clean_dict(trunc_params)
    return options


def _substeps(evolution_params):
    n_substeps = int((evolution_params or {}).get("N_steps", 1))
    if n_substeps < 1:
        raise ValueError("N_steps must be at least 1")
    return n_substeps


def solve_mps_history(
    model,
    initial_mps,
    tlist,
    algorithm="TEBD",
    evolution_params=None,
    trunc_params=None,
    metadata=None,
):
    """Evolve a TenPy MPS and return the workflow index plus MPS history."""
    tlist = np.asarray(tlist, dtype=float)
    if len(tlist) == 0:
        raise ValueError("tlist must contain at least one time")

    metadata = metadata or {}
    trunc_params = trunc_params or {}
    bonddim = metadata.get("bonddim", trunc_params.get("chi_max"))

    mps = initial_mps.copy()
    mps_list = [mps.copy()]
    rows = [{
        "ix_time": 0,
        "time": tlist[0],
        "bonddim": bonddim,
        "uuid_str": "%s" % uuid.uuid4(),
        "walltime": time.time(),
    }]

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
        rows.append({
            "ix_time": ix_time,
            "time": tlist[ix_time],
            "bonddim": bonddim,
            "uuid_str": "%s" % uuid.uuid4(),
            "walltime": time.time(),
        })

    return pandas.DataFrame(rows), mps_list
