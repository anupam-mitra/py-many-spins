import itertools
import time
import uuid

import numpy as np
import pandas
from quspin.basis import spin_basis_1d


def spinhalf_basis(n_spins):
    """Create a full spin-half QuSpin basis using Pauli operators."""
    return spin_basis_1d(L=n_spins, S="1/2", pauli=1)


def spinhalf_state(ang_polar, ang_azimuth):
    """Create single-site amplitudes in QuSpin's spin string convention."""
    return {
        "1": np.cos(ang_polar / 2),
        "0": np.sin(ang_polar / 2) * np.exp(1j * ang_azimuth),
    }


def manyspin_product_state(n_spins, ang_polar, ang_azimuth, basis=None):
    """Create a dense product-state vector in a full QuSpin spin-half basis."""
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


def local_marginal_density_matrix(state, locations, basis):
    """Return the dense reduced density matrix for selected sites."""
    return basis.partial_trace(
        state,
        sub_sys_A=list(locations),
        return_rdm="A",
        sparse=False,
    )


def solve_state_history(hamiltonian, initial_state, tlist, metadata=None):
    """Evolve a QuSpin state and return the workflow index plus state history."""
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

    return pandas.DataFrame(rows), states
