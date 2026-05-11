import time
import uuid

import pandas
import qutip


def spinhalf_state(ang_polar, ang_azimuth):
    """Create a single-spin pure state pointing along the given angles."""
    return qutip.spin_coherent(j=1 / 2, theta=ang_polar, phi=ang_azimuth, type="ket")


def manyspin_product_state(n_spins, ang_polar, ang_azimuth):
    """Create a product state of identically prepared spin-half kets."""
    one_spin_state = spinhalf_state(ang_polar, ang_azimuth)
    return qutip.tensor([one_spin_state] * n_spins)


def local_marginal_density_matrix(state, locations):
    """Return the local marginal density matrix for the selected sites."""
    return state.ptrace(list(locations))


def _history_rows(tlist):
    rows = []
    for ix_time, time_value in enumerate(tlist):
        rows.append({
            "ix_time": ix_time,
            "time": time_value,
            "uuid_str": "%s" % uuid.uuid4(),
            "walltime": time.time(),
        })
    return pandas.DataFrame(rows)


def _state_copies(states):
    return [state.copy() for state in states]


def _solver_options(options=None, keep_runs_results=False):
    solver_options = dict(options or {})
    solver_options["store_states"] = True
    solver_options.setdefault("progress_bar", False)
    if keep_runs_results:
        solver_options["keep_runs_results"] = True
    return solver_options


def _require_state_count(states, tlist, solver_name):
    if len(states) != len(tlist):
        raise ValueError(
            "%s returned %d states for %d requested times"
            % (solver_name, len(states), len(tlist))
        )


def _mcsolve_metadata(result):
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


def solve_state_history(hamiltonian, initial_state, tlist, metadata=None):
    """Evolve a QuTiP state and return the workflow index plus state history."""
    result = qutip.sesolve(hamiltonian, initial_state, tlist)
    states = list(result.states)
    _require_state_count(states, tlist, "QuTiP sesolve")
    return _history_rows(tlist), _state_copies(states)


def solve_master_history(
    hamiltonian,
    initial_state,
    tlist,
    collapse_ops=None,
    options=None,
    metadata=None,
):
    """Evolve a QuTiP density operator history using ``qutip.mesolve``."""
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
    hamiltonian,
    initial_state,
    tlist,
    collapse_ops=None,
    ntraj=500,
    seeds=None,
    target_tol=None,
    timeout=None,
    options=None,
    metadata=None,
):
    """Evolve averaged QuTiP MCWF states using ``qutip.mcsolve``."""
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
