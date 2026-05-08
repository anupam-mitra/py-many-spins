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


def solve_state_history(hamiltonian, initial_state, tlist, metadata=None):
    """Evolve a QuTiP state and return the workflow index plus state history."""
    result = qutip.sesolve(hamiltonian, initial_state, tlist)
    states = list(result.states)
    if len(states) != len(tlist):
        raise ValueError(
            "QuTiP returned %d states for %d requested times" % (len(states), len(tlist))
        )

    metadata = metadata or {}
    rows = []
    for ix_time, time_value in enumerate(tlist):
        rows.append({
            "ix_time": ix_time,
            "time": time_value,
            "bonddim": metadata.get("bonddim"),
            "uuid_str": "%s" % uuid.uuid4(),
            "walltime": time.time(),
        })

    return pandas.DataFrame(rows), [state.copy() for state in states]
