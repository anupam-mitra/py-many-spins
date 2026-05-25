from dataclasses import dataclass, field
from typing import Any, Tuple

import uuid
import quimb as qu
import quimb.tensor as qtn
import numpy as np

from manybody_backends.quimb.options import (
    gate_options,
    tebd_evolution_options,
    tebd_init_options,
)

 
@dataclass
class MPSTrajectoryResult:
    """
    Container for one MPS quantum trajectory history.

    Attributes
    ----------
    str_uuid : str
        Unique identifier for the trajectory.
    tlist : Any
        The time grid.
    psi_t : Any
        The normalized MPS states at each time step.
    tjumps : Any
        The times at which jumps occurred.
    whichjumps : Any
        Records of which jump operator and site were selected.
    random_numbers : Any
        The random numbers used for threshold and choices.
    parameters : dict[str, Any]
        Simulation parameters.
    psi_unnormalized_t : Any
        The unnormalized MPS states before save-time normalization.
    """

    str_uuid: str
    tlist: Any
    psi_t: Any
    tjumps: Any
    whichjumps: Any
    random_numbers: Any
    parameters: dict[str, Any] = field(default_factory=dict)
    psi_unnormalized_t: Any = field(default_factory=list)



class _RandomDraws:
    """
    Handle random draws for MCWF trajectories.

    Attributes
    ----------
    _values : list[float] | None
        Pre-supplied random numbers.
    _index : int
        The current index in _values.
    _rng : Any
        The random number generator.
    jump_thresholds : list[float]
        The thresholds drawn for each step.
    jump_choices : list[float]
        The choices drawn when a jump occurred.
    """

    def __init__(self, random_numbers: Any = None, rng: Any = None, seed: int | None = None) -> None:
        """
        Initialize the random draws handler.

        Parameters
        ----------
        random_numbers : Any, optional
            Pre-supplied random numbers, by default None.
        rng : Any, optional
            Existing random number generator, by default None.
        seed : int, optional
            Seed for the random number generator, by default None.

        Raises
        ------
        ValueError
            If both random_numbers and rng are provided.
        """
        if random_numbers is not None and rng is not None:
            raise ValueError("provide either random_numbers or rng, not both")
        self._values = None
        if random_numbers is not None:
            self._values = list(np.asarray(random_numbers, dtype=float).ravel())
        self._index = 0
        self._rng = rng if rng is not None else np.random.default_rng(seed)
        self.jump_thresholds = []
        self.jump_choices = []

    def _draw(self) -> float:
        """
        Draw a random number between 0 and 1.

        Returns
        -------
        float
            The drawn value.

        Raises
        ------
        ValueError
            If pre-supplied random numbers are exhausted or values are outside [0, 1].
        """
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

    def threshold(self) -> float:
        """
        Draw and record a jump threshold.

        Returns
        -------
        float
            The threshold value.
        """
        value = self._draw()
        self.jump_thresholds.append(value)
        return value

    def choice(self) -> float:
        """
        Draw and record a jump choice.

        Returns
        -------
        float
            The choice value.
        """
        value = self._draw()
        self.jump_choices.append(value)
        return value

    def to_dict(self) -> dict[str, list[float]]:
        """
        Convert the random draws history to a dictionary.

        Returns
        -------
        dict[str, list[float]]
            A dictionary containing jump_thresholds and jump_choices.
        """
        return {
            "jump_thresholds": list(self.jump_thresholds),
            "jump_choices": list(self.jump_choices),
        }



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
    return mps.nsites if hasattr(mps, "nsites") else mps.L

 
def _copy_mps(mps: Any) -> Any:
    """
    Copy a Quimb MPS if possible.

    Parameters
    ----------
    mps : Any
        The MPS to copy.

    Returns
    -------
    Any
        The copied MPS or the original.
    """
    return mps.copy() if hasattr(mps, "copy") else mps

 
def _mps_norm(mps: Any) -> float:
    """
    Compute the norm of a Quimb MPS.

    Parameters
    ----------
    mps : Any
        The Quimb MPS.

    Returns
    -------
    float
        The norm.
    """
    value = np.real_if_close(mps.H @ mps)
    return max(float(np.real(value)), 0.0)

 
def _normalize_mps(mps: Any) -> Any:
    """
    Normalize a Quimb MPS.

    Parameters
    ----------
    mps : Any
        The MPS to normalize.

    Returns
    -------
    Any
        The normalized MPS.

    Raises
    ------
    ValueError
        If the MPS has zero norm.
    """
    norm = _mps_norm(mps)
    if norm <= 0.0:
        raise ValueError("cannot normalize an MPS with zero norm")
    return mps / np.sqrt(norm)

 
def _validate_tlist(tlist: Any) -> np.ndarray:
    """
    Validate the time list.

    Parameters
    ----------
    tlist : Any
        The time list to validate.

    Returns
    -------
    np.ndarray
        The validated time list as a NumPy array.

    Raises
    ------
    ValueError
        If tlist is empty or not strictly increasing.
    """
    tlist = np.asarray(tlist, dtype=float)
    if len(tlist) == 0:
        raise ValueError("tlist must contain at least one time")
    if len(tlist) > 1 and np.any(np.diff(tlist) <= 0.0):
        raise ValueError("tlist must be strictly increasing")
    return tlist



def _validate_collapse_sites(sites: Any, n_sites: int) -> Tuple[int, ...]:
    """
    Validate and expand the sites for a collapse operator.

    Parameters
    ----------
    sites : Any
        The sites specification.
    n_sites : int
        The total number of sites.

    Returns
    -------
    Tuple[int, ...]
        The expanded tuple of site indices.

    Raises
    ------
    ValueError
        If the sites specification is invalid or out of range.
    """
    if sites == "all":
        return tuple(range(n_sites))
    sites = tuple(sites)
    if len(sites) == 0:
        raise ValueError("collapse operator sites must be non-empty")
    for site in sites:
        if site < 0 or site >= n_sites:
            raise ValueError("collapse operator site %d exceeds n_sites" % (site,))
    return sites

 
def _validate_collapse_ops(collapse_ops: Any, n_sites: int) -> Tuple[dict[str, Any], ...]:
    """
    Validate the collapse operators.

    Parameters
    ----------
    collapse_ops : Any
        The collapse operator specifications.
    n_sites : int
        The total number of sites.

    Returns
    -------
    Tuple[dict[str, Any], ...]
        A tuple of validated collapse operator specifications.

    Raises
    ------
    ValueError
        If any collapse operator is invalid.
    """
    operators = []
    for operator in collapse_ops or ():
        sites = "all"
        if isinstance(operator, dict):
            sites = operator.get("sites", "all")
            operator = operator["operator"]
        operator = np.asarray(operator, dtype=complex)
        if operator.shape != (2, 2):
            raise ValueError("collapse operators must be single-site 2x2 arrays")
        operators.append({
            "operator": operator,
            "sites": _validate_collapse_sites(sites, n_sites),
        })
    return tuple(operators)



def _coherent_step(mps: Any, hamiltonian: Any, t0: float, t1: float, tebd_params: dict[str, Any] | None, split_opts: dict[str, Any] | None) -> Any:
    """
    Perform a single coherent step of evolution.

    Parameters
    ----------
    mps : Any
        The current MPS state.
    hamiltonian : Any
        The Hamiltonian builder.
    t0 : float
        The start time.
    t1 : float
        The end time.
    tebd_params : dict[str, Any] | None
        Trotter evolution parameters.
    split_opts : dict[str, Any] | None
        Truncation parameters.

    Returns
    -------
    Any
        The evolved MPS state.
    """
    if hamiltonian is None or t1 == t0:
        return _copy_mps(mps)
    tebd = qtn.TEBD(
        mps,
        hamiltonian,
        t0=t0,
        split_opts=split_opts,
        **tebd_init_options(tebd_params),
    )
    return next(iter(tebd.at_times([t1], **tebd_evolution_options(tebd_params))))

 
def _apply_no_jump_damping(mps: Any, collapse_ops: Tuple[dict[str, Any], ...], dt: float, gate_opts: dict[str, Any]) -> Any:
    """
    Apply the no-jump damping evolution.

    Parameters
    ----------
    mps : Any
        The current MPS state.
    collapse_ops : Tuple[dict[str, Any], ...]
        The validated collapse operators.
    dt : float
        The time step.
    gate_opts : dict[str, Any]
        The gate options.

    Returns
    -------
    Any
        The damped MPS state.
    """
    state = mps
    for collapse_spec in collapse_ops:
        collapse_op = collapse_spec["operator"]
        damping_gate = qu.expm(-0.5 * dt * (qu.dag(collapse_op) @ collapse_op))
        for site in collapse_spec["sites"]:
            state = state.gate(damping_gate, site, contract=True, **gate_opts)
    return state

 
def _jump_probabilities(mps: Any, collapse_ops: Tuple[dict[str, Any], ...], gate_opts: dict[str, Any]) -> Tuple[np.ndarray, list[Tuple[int, int, Any]]]:
    """
    Compute jump probabilities and corresponding jump states.

    Parameters
    ----------
    mps : Any
        The current MPS state.
    collapse_ops : Tuple[dict[str, Any], ...]
        The validated collapse operators.
    gate_opts : dict[str, Any]
        The gate options.

    Returns
    -------
    Tuple[np.ndarray, list[Tuple[int, int, Any]]]:
        A tuple containing the jump probabilities and a list of (op_index, site, jump_state).

    Raises
    ------
    ValueError
        If all jump probabilities are zero.
    """
    probabilities = []
    jump_states = []
    for op_index, collapse_spec in enumerate(collapse_ops):
        collapse_op = collapse_spec["operator"]
        for site in collapse_spec["sites"]:
            jump_state = mps.gate(collapse_op, site, contract=True, **gate_opts)
            jump_states.append((op_index, site, jump_state))
            probabilities.append(_mps_norm(jump_state))

    probabilities = np.asarray(probabilities, dtype=float)
    total = float(np.sum(probabilities))
    if total <= 0.0:
        raise ValueError("jump selected but all jump probabilities are zero")
    return probabilities / total, jump_states



def solve_mps_trajectory(
    initial_mps: Any,
    hamiltonian: Any,
    collapse_ops: Any,
    tlist: Any,
    *,
    n_substeps: int = 1,
    tebd_params: dict[str, Any] | None = None,
    split_opts: dict[str, Any] | None = None,
    random_numbers: Any = None,
    rng: Any = None,
    seed: int | None = None,
    str_uuid: str | None = None,
    metadata: dict[str, Any] | None = None,
    debug: bool = False,
) -> MPSTrajectoryResult:
    """
    Evolve one MCWF trajectory using Quimb MPS states.

    The implementation uses TEBD for coherent evolution and local no-jump
    damping gates ``exp(-0.5 * dt * C^dagger C)`` for collapse operators.
    Saved states are normalized; ``psi_unnormalized_t`` keeps diagnostic copies
    of the states before save-time normalization.

    Parameters
    ----------
    initial_mps : Any
        The initial MPS state.
    hamiltonian : Any
        The Hamiltonian builder.
    collapse_ops : Any
        The collapse operator specifications.
    tlist : Any
        The time grid.
    n_substeps : int, optional
        Number of substeps per time step, by default 1.
    tebd_params : dict[str, Any] | None, optional
        Trotter evolution parameters, by default None.
    split_opts : dict[str, Any] | None, optional
        Truncation parameters, by default None.
    random_numbers : Any, optional
        Pre-supplied random numbers, by default None.
    rng : Any, optional
        Random number generator, by default None.
    seed : int, optional
        Seed for the random number generator, by default None.
    str_uuid : str | None, optional
        Unique identifier for the trajectory, by default None.
    metadata : dict[str, Any] | None, optional
        Simulation metadata, by default None.
    debug : bool, optional
        Whether to print debug information, by default False.

    Returns
    -------
    MPSTrajectoryResult
        The result of the trajectory evolution.

    Raises
    ------
    ValueError
        If tlist is invalid, n_substeps is less than 1, or if a jump is selected
        but all jump probabilities are zero.
    """
    tlist = _validate_tlist(tlist)
    collapse_ops = _validate_collapse_ops(collapse_ops, _n_sites(initial_mps))
    n_substeps = int(n_substeps)
    if n_substeps < 1:
        raise ValueError("n_substeps must be at least 1")

    gate_opts = gate_options(split_opts)
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
                state = _apply_no_jump_damping(state, collapse_ops, dt, gate_opts)
                norm = _mps_norm(state)
                if debug:
                    print("MCWF step t=%g norm=%g threshold=%g" % (t1, norm, jump_threshold))
                if norm <= jump_threshold:
                    probabilities, jump_states = _jump_probabilities(
                        state,
                        collapse_ops,
                        gate_opts,
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
                state = _apply_no_jump_damping(state, collapse_ops, dt, gate_opts)
                norm = _mps_norm(state)
                if debug:
                    print("MCWF step t=%g norm=%g threshold=%g" % (t1, norm, jump_threshold))
                if norm <= jump_threshold:
                    probabilities, jump_states = _jump_probabilities(
                        state,
                        collapse_ops,
                        gate_opts,
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
    psi_initial: Any,
    h_nni: Any,
    jump_ops: Any,
    ts: Any,
    split_opts: Any | None = None,
    epsilon_trotter: float = 1e-6,
    random_numbers: Any | None = None,
    debug: bool = False,
) -> Tuple[Any, Any, Any]:
    """
    Compatibility wrapper returning raw states, normalized states, and jumps.

    Parameters
    ----------
    psi_initial : Any
        The initial MPS state.
    h_nni : Any
        The Hamiltonian.
    jump_ops : Any
        The jump operators.
    ts : Any
        The time grid.
    split_opts : Any, optional
        Truncation parameters, by default None.
    epsilon_trotter : float, optional
        Trotter error tolerance, by default 1e-6.
    random_numbers : Any, optional
        Pre-supplied random numbers, by default None.
    debug : bool, optional
        Whether to print debug information, by default False.

    Returns
    -------
    Tuple[Any, Any, Any]
        A tuple containing raw states, normalized states, and jump times.
    """
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

