from dataclasses import dataclass, field
from typing import Any
import uuid

import quimb as qu
import quimb.tensor as qtn
import numpy as np


@dataclass
class MPSTrajectoryResult:
    """Container for one MPS quantum trajectory history."""

    str_uuid: str
    tlist: Any
    psi_t: Any
    tjumps: Any
    whichjumps: Any
    random_numbers: Any
    parameters: dict[str, Any] = field(default_factory=dict)
    psi_unnormalized_t: Any = field(default_factory=list)


class _RandomDraws:
    def __init__(self, random_numbers=None, rng=None, seed=None):
        if random_numbers is not None and rng is not None:
            raise ValueError("provide either random_numbers or rng, not both")
        self._values = None
        if random_numbers is not None:
            self._values = list(np.asarray(random_numbers, dtype=float).ravel())
        self._index = 0
        self._rng = rng if rng is not None else np.random.default_rng(seed)
        self.jump_thresholds = []
        self.jump_choices = []

    def _draw(self):
        if self._values is None:
            value = float(self._rng.random())
        else:
            if self._index >= len(self._values):
                raise ValueError("not enough random numbers supplied for trajectory")
            value = float(self._values[self._index])
            self._index += 1
        if value < 0.0 or value > 1.0:
            raise ValueError("random draws must lie in [0, 1]")
        return value

    def threshold(self):
        value = self._draw()
        self.jump_thresholds.append(value)
        return value

    def choice(self):
        value = self._draw()
        self.jump_choices.append(value)
        return value

    def to_dict(self):
        return {
            "jump_thresholds": list(self.jump_thresholds),
            "jump_choices": list(self.jump_choices),
        }


def _n_sites(mps):
    return mps.nsites if hasattr(mps, "nsites") else mps.L


def _copy_mps(mps):
    return mps.copy() if hasattr(mps, "copy") else mps


def _mps_norm(mps):
    value = np.real_if_close(mps.H @ mps)
    return max(float(np.real(value)), 0.0)


def _normalize_mps(mps):
    norm = _mps_norm(mps)
    if norm <= 0.0:
        raise ValueError("cannot normalize an MPS with zero norm")
    return mps / np.sqrt(norm)


def _validate_tlist(tlist):
    tlist = np.asarray(tlist, dtype=float)
    if len(tlist) == 0:
        raise ValueError("tlist must contain at least one time")
    if len(tlist) > 1 and np.any(np.diff(tlist) <= 0.0):
        raise ValueError("tlist must be strictly increasing")
    return tlist


def _validate_collapse_ops(collapse_ops):
    operators = []
    for operator in collapse_ops or ():
        operator = np.asarray(operator, dtype=complex)
        if operator.shape != (2, 2):
            raise ValueError("collapse operators must be single-site 2x2 arrays")
        operators.append(operator)
    return tuple(operators)


def _tebd_init_options(tebd_params):
    params = tebd_params or {}
    options = {}
    for key in ("dt", "tol", "progbar"):
        if params.get(key) is not None:
            options[key] = params[key]
    options.setdefault("progbar", False)
    return options


def _tebd_evolution_options(tebd_params):
    params = tebd_params or {}
    options = {}
    for key in ("dt", "tol", "order", "progbar"):
        if params.get(key) is not None:
            options[key] = params[key]
    options.setdefault("progbar", False)
    return options


def _gate_options(split_opts):
    params = split_opts or {}
    options = {}
    for key in ("max_bond", "cutoff"):
        if params.get(key) is not None:
            options[key] = params[key]
    return options


def _coherent_step(mps, hamiltonian, t0, t1, tebd_params, split_opts):
    if hamiltonian is None or t1 == t0:
        return _copy_mps(mps)
    tebd = qtn.TEBD(
        mps,
        hamiltonian,
        t0=t0,
        split_opts=split_opts,
        **_tebd_init_options(tebd_params),
    )
    return next(iter(tebd.at_times([t1], **_tebd_evolution_options(tebd_params))))


def _apply_no_jump_damping(mps, collapse_ops, dt, gate_options):
    state = mps
    for collapse_op in collapse_ops:
        damping_gate = qu.expm(-0.5 * dt * (qu.dag(collapse_op) @ collapse_op))
        for site in range(_n_sites(state)):
            state = state.gate(damping_gate, site, contract=True, **gate_options)
    return state


def _jump_probabilities(mps, collapse_ops, gate_options):
    probabilities = []
    jump_states = []
    n_sites = _n_sites(mps)
    for op_index, collapse_op in enumerate(collapse_ops):
        for site in range(n_sites):
            jump_state = mps.gate(collapse_op, site, contract=True, **gate_options)
            jump_states.append((op_index, site, jump_state))
            probabilities.append(_mps_norm(jump_state))

    probabilities = np.asarray(probabilities, dtype=float)
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise ValueError("jump selected but all jump probabilities are zero")
    return probabilities / total, jump_states


def solve_mps_trajectory(
    initial_mps,
    hamiltonian,
    collapse_ops,
    tlist,
    *,
    n_substeps=1,
    tebd_params=None,
    split_opts=None,
    random_numbers=None,
    rng=None,
    seed=None,
    str_uuid=None,
    metadata=None,
    debug=False,
):
    """Evolve one MCWF trajectory using Quimb MPS states.

    The implementation uses TEBD for coherent evolution and local no-jump
    damping gates ``exp(-0.5 * dt * C^dagger C)`` for collapse operators.
    Saved states are normalized; ``psi_unnormalized_t`` keeps diagnostic copies
    of the states before save-time normalization.
    """
    tlist = _validate_tlist(tlist)
    collapse_ops = _validate_collapse_ops(collapse_ops)
    n_substeps = int(n_substeps)
    if n_substeps < 1:
        raise ValueError("n_substeps must be at least 1")

    gate_options = _gate_options(split_opts)
    random_draws = _RandomDraws(random_numbers=random_numbers, rng=rng, seed=seed)

    state = _normalize_mps(_copy_mps(initial_mps))
    psi_t = [_copy_mps(state)]
    psi_unnormalized_t = [_copy_mps(state)]
    t_jumps = []
    jump_records = []
    jump_threshold = (
        random_draws.threshold() if collapse_ops and len(tlist) > 1 else None
    )

    total_substeps = max(len(tlist) - 1, 0) * n_substeps
    completed_substeps = 0

    for ix_time in range(1, len(tlist)):
        interval_start = tlist[ix_time - 1]
        dt = (tlist[ix_time] - interval_start) / n_substeps

        for ix_substep in range(n_substeps):
            t0 = interval_start + ix_substep * dt
            t1 = t0 + dt
            state = _coherent_step(state, hamiltonian, t0, t1, tebd_params, split_opts)

            if collapse_ops:
                state = _apply_no_jump_damping(state, collapse_ops, dt, gate_options)
                norm = _mps_norm(state)
                if debug:
                    print("MCWF step t=%g norm=%g threshold=%g" % (t1, norm, jump_threshold))
                if norm <= jump_threshold:
                    probabilities, jump_states = _jump_probabilities(
                        state,
                        collapse_ops,
                        gate_options,
                    )
                    choice = random_draws.choice()
                    cumulative = np.cumsum(probabilities)
                    event_index = int(np.searchsorted(cumulative, choice, side="right"))
                    if event_index >= len(jump_states):
                        event_index = len(jump_states) - 1
                    op_index, site, jump_state = jump_states[event_index]
                    state = _normalize_mps(jump_state)
                    t_jumps.append(float(t1))
                    jump_records.append({
                        "time": float(t1),
                        "ix_time": ix_time,
                        "ix_substep": ix_substep,
                        "op_index": op_index,
                        "site": site,
                        "probability": float(probabilities[event_index]),
                    })

                    if completed_substeps + 1 < total_substeps:
                        jump_threshold = random_draws.threshold()

            completed_substeps += 1

        psi_unnormalized_t.append(_copy_mps(state))
        psi_t.append(_copy_mps(_normalize_mps(state)))

    parameters = {
        "n_substeps": n_substeps,
        "tebd_params": dict(tebd_params or {}),
        "split_opts": dict(split_opts or {}),
        **dict(metadata or {}),
    }
    return MPSTrajectoryResult(
        str_uuid=str_uuid or "%s" % uuid.uuid4(),
        tlist=tlist.copy(),
        psi_t=psi_t,
        tjumps=t_jumps,
        whichjumps=jump_records,
        random_numbers=random_draws.to_dict(),
        parameters=parameters,
        psi_unnormalized_t=psi_unnormalized_t,
    )


def eval_single_trajectory(
    psi_initial,
    h_nni,
    jump_ops,
    ts,
    split_opts=None,
    epsilon_trotter=1e-6,
    random_numbers=None,
    debug=False,
):
    """Compatibility wrapper returning raw states, normalized states, and jumps."""
    result = solve_mps_trajectory(
        psi_initial,
        h_nni,
        jump_ops,
        ts,
        split_opts=split_opts,
        tebd_params={"tol": epsilon_trotter},
        random_numbers=random_numbers,
        debug=debug,
    )
    return result.psi_unnormalized_t, result.psi_t, result.tjumps
