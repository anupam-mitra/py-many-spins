import time
import uuid

import pandas
import qutip

from .manybodystateevolve import QutipSESolve


def spinhalf_state(ang_polar, ang_azimuth):
    """Create a single-spin pure state pointing along the given angles."""
    return qutip.spin_coherent(j=1 / 2, theta=ang_polar, phi=ang_azimuth, type='ket')


def manyspin_product_state(n_spins, ang_polar, ang_azimuth):
    """Create a product state of identically prepared spin-half kets."""
    one_spin_state = spinhalf_state(ang_polar, ang_azimuth)
    return qutip.tensor([one_spin_state] * n_spins)


def local_marginal_density_matrix(state, locations):
    """Return the local marginal density matrix for the selected sites."""
    return state.ptrace(list(locations))


class SESolveWrapper:
    """A QuTiP state-evolution wrapper mirroring the TenPy workflow API."""

    def __init__(self, hamiltonian_model, initial_state, tlist, metadata=None):
        self.hamiltonian_model = hamiltonian_model
        self.initial_state = initial_state
        self.tlist = tlist
        self.metadata = metadata or {}

        self.state_list = []
        self.rows = []

    def evolve(self):
        n_dof = len(self.initial_state.dims[0])
        solver = QutipSESolve(
            self.tlist[0],
            self.tlist[-1],
            len(self.tlist),
            self.hamiltonian_model,
            n_dof,
            self.initial_state,
        )
        solver.run()

        states = list(solver.states)
        if len(states) != len(self.tlist):
            raise ValueError(
                "QuTiP returned %d states for %d requested times"
                % (len(states), len(self.tlist))
            )

        self.state_list = [state.copy() for state in states]
        self.rows = []
        for ix_time, time_value in enumerate(self.tlist):
            self.rows.append({
                "ix_time": ix_time,
                "time": time_value,
                "bonddim": self.metadata.get("bonddim"),
                "uuid_str": "%s" % uuid.uuid4(),
                "walltime": time.time(),
            })

        self.df = pandas.DataFrame(self.rows)

    def get_mps_history_df(self):
        if not hasattr(self, "df"):
            self.df = pandas.DataFrame(self.rows)

        return self.df, self.state_list

    def get_state_history_df(self):
        return self.get_mps_history_df()
