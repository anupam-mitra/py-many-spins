from dataclasses import replace
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from manybody_util.history import history_records
from manybody_util.spinmodel import tilted_field_ising_1d, SpinHalfPauliModel
from time_evolution.persistence import RunStore
from time_evolution.results import EvolutionResult
from time_evolution.specs import EXACT_BACKENDS, SimulationSpec, ModelSpec, TimeGridSpec

T = TypeVar("T")


def build_spin_model(model_spec: ModelSpec) -> SpinHalfPauliModel:
    """
    Build a spin model from its specification.

    Parameters
    ----------
    model_spec : ModelSpec
        The model specification.

    Returns
    -------
    SpinHalfPauliModel
        The built spin model.

    Raises
    ------
    ValueError
        If the model kind is unsupported.
    """
    model_spec.validate()
    if model_spec.kind == "tilted_field_ising_1d":
        return tilted_field_ising_1d(
            n_sites=model_spec.n_sites,
            j_xx=model_spec.j_xx,
            b_z=model_spec.b_z,
            b_x=model_spec.b_x,
            bc=model_spec.bc,
        )
    raise ValueError("unsupported model kind %r" % (model_spec.kind,))


def build_time_grid(time_grid_spec: TimeGridSpec) -> np.ndarray:
    """
    Build a time grid from its specification.

    Parameters
    ----------
    time_grid_spec : TimeGridSpec
        The time grid specification.

    Returns
    -------
    np.ndarray
        The built time grid as a NumPy array.

    Raises
    ------
    ValueError
        If the time grid kind is unsupported.
    """
    time_grid_spec.validate()
    if time_grid_spec.kind == "linspace":
        return np.linspace(
            time_grid_spec.start,
            time_grid_spec.stop,
            time_grid_spec.num,
        )
    raise ValueError("unsupported time grid kind %r" % (time_grid_spec.kind,))


def run_simulation(spec: SimulationSpec) -> EvolutionResult:
    """
    Run a time evolution simulation.

    Parameters
    ----------
    spec : SimulationSpec
        The full simulation specification.

    Returns
    -------
    EvolutionResult
        The result of the simulation.

    Raises
    ------
    ValueError
        If the backend is unsupported.
    """
    spec.validate()
    backend = spec.method.backend.lower()
    spin_model = build_spin_model(spec.model)
    tlist = build_time_grid(spec.time_grid)

    if backend == "tenpy":
        return _run_tenpy(spec, spin_model, tlist)
    if backend == "quimb":
        return _run_quimb(spec, spin_model, tlist)
    if backend == "qutip":
        return _run_qutip(spec, spin_model, tlist)
    if backend == "quspin":
        return _run_quspin(spec, spin_model, tlist)
    raise ValueError("unsupported backend %r" % (spec.method.backend,))


def run_and_store(spec: SimulationSpec) -> tuple[RunStore, dict[str, Any]]:
    """
    Run a simulation and store the result on disk.

    Parameters
    ----------
    spec : SimulationSpec
        The simulation specification.

    Returns
    -------
    tuple[RunStore, dict[str, Any]]
        A tuple containing the RunStore and the index of the saved result.
    """
    store = RunStore(spec.output.base_dir, run_id=spec.output.run_id)
    stored_spec = spec
    if spec.output.run_id != store.run_id:
        stored_spec = replace(spec, output=replace(spec.output, run_id=store.run_id))

    result = run_simulation(stored_spec)
    index = store.save_evolution_result(stored_spec, result)
    return store, index


def _base_metadata(spec: SimulationSpec, tlist: np.ndarray) -> dict[str, Any]:
    """
    Generate basic metadata for a simulation run.

    Parameters
    ----------
    spec : SimulationSpec
        The simulation specification.
    tlist : np.ndarray
        The time grid.

    Returns
    -------
    dict[str, Any]
        The generated metadata.
    """
    backend = spec.method.backend.lower()
    metadata = {
        "backend": backend,
        "algorithm": spec.method.algorithm,
        "n_sites": spec.model.n_sites,
        "time_count": int(len(tlist)),
    }
    if backend not in EXACT_BACKENDS:
        metadata["bonddim"] = spec.method.trunc_params.get("chi_max")
    if spec.collapse_operators:
        metadata["collapse_operators"] = [
            collapse.to_dict() for collapse in spec.collapse_operators
        ]
    return metadata


def _records_from_dataframe(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Convert a pandas DataFrame to a list of records.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame to convert.

    Returns
    -------
    list[dict[str, Any]]
        The list of records.
    """
    return dataframe.to_dict(orient="records")


def _result_from_dataframe(
    dataframe: pd.DataFrame, states: list[Any], metadata: dict[str, Any]
) -> EvolutionResult:
    """
    Create an EvolutionResult from a DataFrame, states, and metadata.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing evolution records.
    states : list[Any]
        The evolution states.
    metadata : dict[str, Any]
        The simulation metadata.

    Returns
    -------
    EvolutionResult
        The constructed EvolutionResult.
    """
    return EvolutionResult(
        records=_records_from_dataframe(dataframe),
        states=states,
        metadata=metadata,
    )


def _state_records(
    tlist: np.ndarray, bonddim: int | None = None, trajectory_id: str | None = None
) -> list[dict[str, Any]]:
    """
    Generate records for the evolution states.

    Parameters
    ----------
    tlist : np.ndarray
        The time grid.
    bonddim : int, optional
        The bond dimension, by default None.
    trajectory_id : str, optional
        The trajectory ID for MCWF, by default None.

    Returns
    -------
    list[dict[str, Any]]
        The generated records.
    """
    return history_records(
        tlist,
        {"bonddim": bonddim, "trajectory_id": trajectory_id},
        omit_none=True,
    )


def _local_spin_matrix(operator: str) -> np.ndarray:
    """
    Get the local spin matrix for a given operator string.

    Parameters
    ----------
    operator : str
        The operator string (e.g., 'sigmam', 'x', 'z').

    Returns
    -------
    np.ndarray
        The corresponding 2x2 matrix.

    Raises
    ------
    ValueError
        If the operator is unsupported.
    """
    operator = operator.lower()
    if operator == "sigmam":
        return np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    if operator == "sigmap":
        return np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    if operator == "x":
        return np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    if operator == "y":
        return np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    if operator == "z":
        return np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    raise ValueError("unsupported collapse operator %r" % (operator,))


def _build_quimb_collapse_ops(
    collapse_specs: tuple[Any, ...], n_sites: int
) -> list[dict[str, Any]]:
    """
    Build Quimb collapse operators from specifications.

    Parameters
    ----------
    collapse_specs : tuple[Any, ...]
        The collapse operator specifications.
    n_sites : int
        The number of sites in the model.

    Returns
    -------
    list[dict[str, Any]]
        The built Quimb collapse operators.
    """
    collapse_ops = []
    for collapse in collapse_specs:
        if collapse.rate == 0.0:
            continue
        collapse_ops.append({
            "operator": np.sqrt(collapse.rate) * _local_spin_matrix(collapse.operator),
            "sites": collapse.expanded_sites(n_sites),
        })
    return collapse_ops


def _qutip_local_spin_operator(operator: str) -> Any:
    """
    Get the QuTiP local spin operator.

    Parameters
    ----------
    operator : str
        The operator string (e.g., 'sigmam', 'x', 'z').

    Returns
    -------
    Any
        The QuTiP operator.

    Raises
    ------
    ValueError
        If the operator is unsupported.
    """
    import qutip

    operator = operator.lower()
    if operator == "sigmam":
        return qutip.sigmam()
    if operator == "sigmap":
        return qutip.sigmap()
    if operator == "x":
        return qutip.sigmax()
    if operator == "y":
        return qutip.sigmay()
    if operator == "z":
        return qutip.sigmaz()
    raise ValueError("unsupported collapse operator %r" % (operator,))


def _qutip_embed_local_operator(n_sites: int, operator: Any, site: int) -> Any:
    """
    Embed a local operator into the full system Hilbert space.

    Parameters
    ----------
    n_sites : int
        The number of sites in the model.
    operator : Any
        The local operator.
    site : int
        The site index.

    Returns
    -------
    Any
        The embedded operator.
    """
    import qutip

    factors = [qutip.qeye(2) for _ in range(n_sites)]
    factors[site] = operator
    return qutip.tensor(factors)


def _build_qutip_collapse_ops(
    collapse_specs: tuple[Any, ...], n_sites: int
) -> list[Any]:
    """
    Build QuTiP collapse operators from specifications.

    Parameters
    ----------
    collapse_specs : tuple[Any, ...],
        The collapse operator specifications.
    n_sites : int
        The number of sites in the model.

    Returns
    -------
    list[Any]
        The built QuTiP collapse operators.
    """
    collapse_ops = []
    for collapse in collapse_specs:
        if collapse.rate == 0.0:
            continue
        local_operator = np.sqrt(collapse.rate) * _qutip_local_spin_operator(
            collapse.operator
        )
        for site in collapse.expanded_sites(n_sites):
            collapse_ops.append(
                _qutip_embed_local_operator(n_sites, local_operator, site)
            )
    return collapse_ops


def _run_tenpy(spec: SimulationSpec, spin_model: SpinHalfPauliModel, tlist: np.ndarray) -> EvolutionResult:
    """
    Run the TenPy time evolution.

    Parameters
    ----------
    spec : SimulationSpec
        The simulation specification.
    spin_model : SpinHalfPauliModel
        The built spin model.
    tlist : np.ndarray
        The time grid.

    Returns
    -------
    EvolutionResult
        The TenPy evolution result.
    """
    from manybody_backends.tenpy.spinmodel import to_tenpy_model
    from manybody_backends.tenpy.timeevolution import (
        manyspin_product_mps,
        solve_mps_history,
    )

    tenpy_model = to_tenpy_model(spin_model, bc_mps="finite", conserve=None)
    initial_mps = manyspin_product_mps(
        tenpy_model.lat.mps_sites(),
        spec.initial_state.theta,
        spec.initial_state.phi,
        bc=tenpy_model.lat.bc_MPS,
        unit_cell_width=tenpy_model.lat.mps_unit_cell_width,
    )
    metadata = _base_metadata(spec, tlist)
    dataframe, states = solve_mps_history(
        tenpy_model,
        initial_mps,
        tlist,
        algorithm=spec.method.algorithm,
        evolution_params=spec.method.evolution_params,
        trunc_params=spec.method.trunc_params,
        metadata=metadata,
    )
    return _result_from_dataframe(dataframe, states, metadata)


def _quimb_evolution_params(spec: SimulationSpec, tlist: np.ndarray) -> dict[str, Any]:
    """
    Build evolution parameters for Quimb.

    Parameters
    ----------
    spec : SimulationSpec
        The simulation specification.
    tlist : np.ndarray
        The time grid.

    Returns
    -------
    dict[str, Any]
        The evolution parameters.
    """
    params = dict(spec.method.evolution_params)
    if len(tlist) > 1 and params.get("dt") is None and params.get("tol") is None:
        params["dt"] = float(tlist[1] - tlist[0])
    return params


def _quimb_mcwf_tebd_params(
    evolution_params: dict[str, Any], tlist: np.ndarray, n_substeps: int
) -> dict[str, Any]:
    """
    Build TEBD parameters for Quimb MCWF.

    Parameters
    ----------
    evolution_params : dict[str, Any]
        The evolution parameters.
    tlist : np.ndarray
        The time grid.
    n_substeps : int
        The number of substeps.

    Returns
    -------
    dict[str, Any]
        The TEBD parameters.
    """
    tebd_params = dict(evolution_params.get("tebd_params", {}))
    for key in ("dt", "tol", "order", "progbar"):
        if key in evolution_params and evolution_params[key] is not None:
            tebd_params[key] = evolution_params[key]
    if (
        len(tlist) > 1
        and tebd_params.get("dt") is None
        and tebd_params.get("tol") is None
    ):
        tebd_params["dt"] = float(tlist[1] - tlist[0]) / n_substeps
    return tebd_params


def _run_quimb(spec: SimulationSpec, spin_model: SpinHalfPauliModel, tlist: np.ndarray) -> EvolutionResult:
    """
    Run the Quimb time evolution.

    Parameters
    ----------
    spec : SimulationSpec
        The simulation specification.
    spin_model : SpinHalfPauliModel
        The built spin model.
    tlist : np.ndarray
        The time grid.

    Returns
    -------
    EvolutionResult
        The Quimb evolution result.
    """
    import quimb.tensor as qtn

    from manybody_backends.quimb.options import split_options
    from manybody_backends.quimb.quimbtebd import TEBDWrapper, spinhalf_state
    from manybody_backends.quimb.quimbmpstrajectory import solve_mps_trajectory
    from manybody_backends.quimb.spinmodel import to_quimb_spinham1d

    initial_mps = qtn.MPS_product_state(
        [spinhalf_state(spec.initial_state.theta, spec.initial_state.phi)]
        * spin_model.n_sites,
        cyclic=(spec.model.bc == "periodic"),
    )
    builder = to_quimb_spinham1d(spin_model)
    metadata = _base_metadata(spec, tlist)
    if spec.method.algorithm == "MCWF":
        evolution_params = dict(spec.method.evolution_params)
        n_substeps = int(evolution_params.get("n_substeps", 1))
        tebd_params = _quimb_mcwf_tebd_params(evolution_params, tlist, n_substeps)

        trajectory = solve_mps_trajectory(
            initial_mps,
            builder.build_local_ham(spin_model.n_sites),
            _build_quimb_collapse_ops(spec.collapse_operators, spin_model.n_sites),
            tlist,
            n_substeps=n_substeps,
            tebd_params=tebd_params,
            split_opts=split_options(spec.method.trunc_params),
            random_numbers=evolution_params.get("random_numbers"),
            seed=evolution_params.get("seed"),
            metadata=metadata,
        )
        metadata = {
            **metadata,
            "trajectory_id": trajectory.str_uuid,
            "tjumps": trajectory.tjumps,
            "whichjumps": trajectory.whichjumps,
            "random_numbers": trajectory.random_numbers,
            "trajectory_parameters": trajectory.parameters,
        }
        return EvolutionResult(
            records=_state_records(
                tlist,
                bonddim=spec.method.trunc_params.get("chi_max"),
                trajectory_id=trajectory.str_uuid,
            ),
            states=trajectory.psi_t,
            metadata=metadata,
        )

    wrapper = TEBDWrapper(
        builder,
        initial_mps,
        tlist,
        _quimb_evolution_params(spec, tlist),
        spec.method.trunc_params,
    )
    wrapper.evolve()
    dataframe, states = wrapper.get_mps_history_df()
    return _result_from_dataframe(dataframe, states, metadata)


def _run_qutip(spec: SimulationSpec, spin_model: SpinHalfPauliModel, tlist: np.ndarray) -> EvolutionResult:
    """
    Run the QuTiP time evolution.

    Parameters
    ----------
    spec : SimulationSpec
        The simulation specification.
    spin_model : SpinHalfPauliModel
        The built spin model.
    tlist : np.ndarray
        The time grid.

    Returns
    -------
    EvolutionResult
        The QuTiP evolution result.
    """
    from manybody_backends.qutip.spinmodel import to_qutip_hamiltonian
    from manybody_backends.qutip.timeevolution import (
        manyspin_product_state,
        solve_master_history,
        solve_monte_carlo_history,
        solve_state_history,
    )

    initial_state = manyspin_product_state(
        spin_model.n_sites,
        spec.initial_state.theta,
        spec.initial_state.phi,
    )
    hamiltonian = to_qutip_hamiltonian(spin_model)
    metadata = _base_metadata(spec, tlist)
    collapse_ops = _build_qutip_collapse_ops(
        spec.collapse_operators,
        spin_model.n_sites,
    )

    if spec.method.algorithm == "mesolve":
        dataframe, states = solve_master_history(
            hamiltonian,
            initial_state,
            tlist,
            collapse_ops=collapse_ops,
            options=spec.method.evolution_params.get("options"),
            metadata=metadata,
        )
    elif spec.method.algorithm == "mcsolve":
        evolution_params = dict(spec.method.evolution_params)
        dataframe, states, solver_metadata = solve_monte_carlo_history(
            hamiltonian,
            initial_state,
            tlist,
            collapse_ops=collapse_ops,
            ntraj=evolution_params.get("ntraj") or 500,
            seeds=evolution_params.get("seeds"),
            target_tol=evolution_params.get("target_tol"),
            timeout=evolution_params.get("timeout"),
            options=evolution_params.get("options"),
            metadata=metadata,
        )
        metadata = {**metadata, "mcsolve": solver_metadata}
    else:
        dataframe, states = solve_state_history(
            hamiltonian,
            initial_state,
            tlist,
            metadata=metadata,
        )

    return _result_from_dataframe(dataframe, states, metadata)


def _run_quspin(spec: SimulationSpec, spin_model: SpinHalfPauliModel, tlist: np.ndarray) -> EvolutionResult:
    """
    Run the QuSpin time evolution.

    Parameters
    ----------
    spec : SimulationSpec
        The simulation specification.
    spin_model : SpinHalfPauliModel
        The built spin model.
    tlist : np.ndarray
        The time grid.

    Returns
    -------
    EvolutionResult
        The QuSpin evolution result.
    """
    from manybody_backends.quspin.spinmodel import (
        to_quspin_basis,
        to_quspin_hamiltonian,
    )
    from manybody_backends.quspin.timeevolution import (
        manyspin_product_state,
        solve_state_history,
    )

    basis = to_quspin_basis(spin_model)
    metadata = _base_metadata(spec, tlist)
    initial_state = manyspin_product_state(
        spin_model.n_sites,
        spec.initial_state.theta,
        spec.initial_state.phi,
        basis=basis,
    )
    dataframe, states = solve_state_history(
        to_quspin_hamiltonian(spin_model, basis=basis),
        initial_state,
        tlist,
        metadata=metadata,
    )
    return _result_from_dataframe(dataframe, states, metadata)

    raise ValueError("unsupported model kind %r" % (model_spec.kind,))


def build_time_grid(time_grid_spec: TimeGridSpec) -> np.ndarray:
    """
    Build a time grid from its specification.

    Parameters
    ----------
    time_grid_spec : TimeGridSpec
        The time grid specification.

    Returns
    -------
    np.ndarray
        The built time grid as a NumPy array.

    Raises
    ------
    ValueError
        If the time grid kind is unsupported.
    """
    time_grid_spec.validate()
    if time_grid_spec.kind == "linspace":
        return np.linspace(
            time_grid_spec.start,
            time_grid_spec.stop,
            time_grid_spec.num,
        )
    raise ValueError("unsupported time grid kind %r" % (time_grid_spec.kind,))


def run_simulation(spec: SimulationSpec) -> EvolutionResult:
    """
    Run a time evolution simulation.

    Parameters
    ----------
    spec : SimulationSpec
        The full simulation specification.

    Returns
    -------
    EvolutionResult
        The result of the simulation.

    Raises
    ------
    ValueError
        If the backend is unsupported.
    """
    spec.validate()
    backend = spec.method.backend.lower()
    spin_model = build_spin_model(spec.model)
    tlist = build_time_grid(spec.time_grid)

    if backend == "tenpy":
        return _run_tenpy(spec, spin_model, tlist)
    if backend == "quimb":
        return _run_quimb(spec, spin_model, tlist)
    if backend == "qutip":
        return _run_qutip(spec, spin_model, tlist)
    if backend == "quspin":
        return _run_quspin(spec, spin_model, tlist)
    raise ValueError("unsupported backend %r" % (spec.method.backend,))


def run_and_store(spec: SimulationSpec) -> tuple[RunStore, dict[str, Any]]:
    """
    Run a simulation and store the result on disk.

    Parameters
    ----------
    spec : SimulationSpec
        The simulation specification.

    Returns
    -------
    tuple[RunStore, dict[str, Any]]
        A tuple containing the RunStore and the index of the saved result.
    """
    store = RunStore(spec.output.base_dir, run_id=spec.output.run_id)
    stored_spec = spec
    if spec.output.run_id != store.run_id:
        stored_spec = replace(spec, output=replace(spec.output, run_id=store.run_id))

    result = run_simulation(stored_spec)
    index = store.save_evolution_result(stored_spec, result)
    return store, index


def _base_metadata(spec: SimulationSpec, tlist: np.ndarray) -> dict[str, Any]:
    """
    Generate basic metadata for a simulation run.

    Parameters
    ----------
    spec : SimulationSpec
        The simulation specification.
    tlist : np.ndarray
        The time grid.

    Returns
    -------
    dict[str, Any]
        The generated metadata.
    """
    backend = spec.method.backend.lower()
    metadata = {
        "backend": backend,
        "algorithm": spec.method.algorithm,
        "n_sites": spec.model.n_sites,
        "time_count": int(len(tlist)),
    }
    if backend not in EXACT_BACKENDS:
        metadata["bonddim"] = spec.method.trunc_params.get("chi_max")
    if spec.collapse_operators:
        metadata["collapse_operators"] = [
            collapse.to_dict() for collapse in spec.collapse_operators
        ]
    return metadata


def _records_from_dataframe(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Convert a pandas DataFrame to a list of records.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame to convert.

    Returns
    -------
    list[dict[str, Any]]
        The list of records.
    """
    return dataframe.to_dict(orient="records")


def _result_from_dataframe(
    dataframe: pd.DataFrame, states: list[Any], metadata: dict[str, Any]
) -> EvolutionResult:
    """
    Create an EvolutionResult from a DataFrame, states, and metadata.

    Parameters
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing evolution records.
    states : list[Any]
        The evolution states.
    metadata : dict[str, Any]
        The simulation metadata.

    Returns
    -------
    EvolutionResult
        The constructed EvolutionResult.
    """
    return EvolutionResult(
        records=_records_from_dataframe(dataframe),
        states=states,
        metadata=metadata,
    )


def _state_records(
    tlist: np.ndarray, bonddim: int | None = None, trajectory_id: str | None = None
) -> list[dict[str, Any]]:
    """
    Generate records for the evolution states.

    Parameters
    ----------
    tlist : np.ndarray
        The time grid.
    bonddim : int, optional
        The bond dimension, by default None.
    trajectory_id : str, optional
        The trajectory ID for MCWF, by default None.

    Returns
    -------
    list[dict[str, Any]]
        The generated records.
    """
    return history_records(
        tlist,
        {"bonddim": bonddim, "trajectory_id": trajectory_id},
        omit_none=True,
    )


def _local_spin_matrix(operator: str) -> np.ndarray:
    """
    Get the local spin matrix for a given operator string.

    Parameters
    ----------
    operator : str
        The operator string (e.g., 'sigmam', 'x', 'z').

    Returns
    -------
    np.ndarray
        The corresponding 2x2 matrix.

    Raises
    ------
    ValueError
        If the operator is unsupported.
    """
    operator = operator.lower()
    if operator == "sigmam":
        return np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    if operator == "sigmap":
        return np.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    if operator == "x":
        return np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    if operator == "y":
        return np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
    if operator == "z":
        return np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    raise ValueError("unsupported collapse operator %r" % (operator,))


def _build_quimb_collapse_ops(
    collapse_specs: tuple[Any, ...], n_sites: int
) -> list[dict[str, Any]]:
    """
    Build Quimb collapse operators from specifications.

    Parameters
    ----------
    collapse_specs : tuple[Any, ...]
        The collapse operator specifications.
    n_sites : int
        The number of sites in the model.

    Returns
    -------
    list[dict[str, Any]]
        The built Quimb collapse operators.
    """
    collapse_ops = []
    for collapse in collapse_specs:
        if collapse.rate == 0.0:
            continue
        collapse_ops.append({
            "operator": np.sqrt(collapse.rate) * _local_spin_matrix(collapse.operator),
            "sites": collapse.expanded_sites(n_sites),
        })
    return collapse_ops


def _qutip_local_spin_operator(operator: str) -> Any:
    """
    Get the QuTiP local spin operator.

    Parameters
    ----------
    operator : str
        The operator string (e.g., 'sigmam', 'x', 'z').

    Returns
    -------
    Any
        The QuTiP operator.

    Raises
    ------
    ValueError
        If the operator is unsupported.
    """
    import qutip

    operator = operator.lower()
    if operator == "sigmam":
        return qutip.sigmam()
    if operator == "sigmap":
        return qutip.sigmap()
    if operator == "x":
        return qutip.sigmax()
    if operator == "y":
        return qutip.sigmay()
    if operator == "z":
        return qutip.sigmaz()
    raise ValueError("unsupported collapse operator %r" % (operator,))


def _qutip_embed_local_operator(n_sites: int, operator: Any, site: int) -> Any:
    """
    Embed a local operator into the full system Hilbert space.

    Parameters
    ----------
    n_sites : int
        The number of sites in the model.
    operator : Any
        The local operator.
    site : int
        The site index.

    Returns
    -------
    Any
        The embedded operator.
    """
    import qutip

    factors = [qutip.qeye(2) for _ in range(n_sites)]
    factors[site] = operator
    return qutip.tensor(factors)


def _build_qutip_collapse_ops(
    collapse_specs: tuple[Any, ...], n_sites: int
) -> list[Any]:
    """
    Build QuTiP collapse operators from specifications.

    Parameters
    ----------
    collapse_specs : tuple[Any, ...],
        The collapse operator specifications.
    n_sites : int
        The number of sites in the model.

    Returns
    -------
    list[Any]
        The built QuTiP collapse operators.
    """
    collapse_ops = []
    for collapse in collapse_specs:
        if collapse.rate == 0.0:
            continue
        local_operator = np.sqrt(collapse.rate) * _qutip_local_spin_operator(
            collapse.operator
        )
        for site in collapse.expanded_sites(n_sites):
            collapse_ops.append(
                _qutip_embed_local_operator(n_sites, local_operator, site)
            )
    return collapse_ops


def _run_tenpy(spec: SimulationSpec, spin_model: SpinHalfPauliModel, tlist: np.ndarray) -> EvolutionResult:
    """
    Run the TenPy time evolution.

    Parameters
    ----------
    spec : SimulationSpec
        The simulation specification.
    spin_model : SpinModel
        The built spin model.
    tlist : np.ndarray
        The time grid.

    Returns
    -------
    EvolutionResult
        The TenPy evolution result.
    """
    from manybody_backends.tenpy.spinmodel import to_tenpy_model
    from manybody_backends.tenpy.timeevolution import (
        manyspin_product_mps,
        solve_mps_history,
    )

    tenpy_model = to_tenpy_model(spin_model, bc_mps="finite", conserve=None)
    initial_mps = manyspin_product_mps(
        tenpy_model.lat.mps_sites(),
        spec.initial_state.theta,
        spec.initial_state.phi,
        bc=tenpy_model.lat.bc_MPS,
        unit_cell_width=tenpy_model.lat.mps_unit_cell_width,
    )
    metadata = _base_metadata(spec, tlist)
    dataframe, states = solve_mps_history(
        tenpy_model,
        initial_mps,
        tlist,
        algorithm=spec.method.algorithm,
        evolution_params=spec.method.evolution_params,
        trunc_params=spec.method.trunc_params,
        metadata=metadata,
    )
    return _result_from_dataframe(dataframe, states, metadata)


def _quimb_evolution_params(spec: SimulationSpec, tlist: np.ndarray) -> dict[str, Any]:
    """
    Build evolution parameters for Quimb.

    Parameters
    ----------
    spec : SimulationSpec
        The simulation specification.
    tlist : np.ndarray
        The time grid.

    Returns
    -------
    dict[str, Any]
        The evolution parameters.
    """
    params = dict(spec.method.evolution_params)
    if len(tlist) > 1 and params.get("dt") is None and params.get("tol") is None:
        params["dt"] = float(tlist[1] - tlist[0])
    return params


def _quimb_mcwf_tebd_params(
    evolution_params: dict[str, Any], tlist: np.ndarray, n_substeps: int
) -> dict[str, Any]:
    """
    Build TEBD parameters for Quimb MCWF.

    Parameters
    ----------
    evolution_params : dict[str, Any]
        The evolution parameters.
    tlist : np.ndarray
        The time grid.
    n_substeps : int
        The number of substeps.

    Returns
    -------
    dict[str, Any]
        The TEBD parameters.
    """
    tebd_params = dict(evolution_params.get("tebd_params", {}))
    for key in ("dt", "tol", "order", "progbar"):
        if key in evolution_params and evolution_params[key] is not None:
            tebd_params[key] = evolution_params[key]
    if (
        len(tlist) > 1
        and tebd_params.get("dt") is None
        and tebd_params.get("tol") is None
    ):
        tebd_params["dt"] = float(tlist[1] - tlist[0]) / n_substeps
    return tebd_params


def _run_quimb(spec: SimulationSpec, spin_model: SpinHalfPauliModel, tlist: np.ndarray) -> EvolutionResult:
    """
    Run the Quimb time evolution.

    Parameters
    ----------
    spec : SimulationSpec
        The simulation specification.
    spin_model : SpinModel
        The built spin model.
    tlist : np.ndarray
        The time grid.

    Returns
    -------
    EvolutionResult
        The Quimb evolution result.
    """
    import quimb.tensor as qtn

    from manybody_backends.quimb.options import split_options
    from manybody_backends.quimb.quimbtebd import TEBDWrapper, spinhalf_state
    from manybody_backends.quimb.quimbmpstrajectory import solve_mps_trajectory
    from manybody_backends.quimb.spinmodel import to_quimb_spinham1d

    initial_mps = qtn.MPS_product_state(
        [spinhalf_state(spec.initial_state.theta, spec.initial_state.phi)]
        * spin_model.n_sites,
        cyclic=(spec.model.bc == "periodic"),
    )
    builder = to_quimb_spinham1d(spin_model)
    metadata = _base_metadata(spec, tlist)
    if spec.method.algorithm == "MCWF":
        evolution_params = dict(spec.method.evolution_params)
        n_substeps = int(evolution_params.get("n_substeps", 1))
        tebd_params = _quimb_mcwf_tebd_params(evolution_params, tlist, n_substeps)

        trajectory = solve_mps_trajectory(
            initial_mps,
            builder.build_local_ham(spin_model.n_sites),
            _build_quimb_collapse_ops(spec.collapse_operators, spin_model.n_sites),
            tlist,
            n_substeps=n_substeps,
            tebd_params=tebd_params,
            split_opts=split_options(spec.method.trunc_params),
            random_numbers=evolution_params.get("random_numbers"),
            seed=evolution_params.get("seed"),
            metadata=metadata,
        )
        metadata = {
            **metadata,
            "trajectory_id": trajectory.str_uuid,
            "tjumps": trajectory.tjumps,
            "whichjumps": trajectory.whichjumps,
            "random_numbers": trajectory.random_numbers,
            "trajectory_parameters": trajectory.parameters,
        }
        return EvolutionResult(
            records=_state_records(
                tlist,
                bonddim=spec.method.trunc_params.get("chi_max"),
                trajectory_id=trajectory.str_uuid,
            ),
            states=trajectory.psi_t,
            metadata=metadata,
        )

    wrapper = TEBDWrapper(
        builder,
        initial_mps,
        tlist,
        _quimb_evolution_params(spec, tlist),
        spec.method.trunc_params,
    )
    wrapper.evolve()
    dataframe, states = wrapper.get_mps_history_df()
    return _result_from_dataframe(dataframe, states, metadata)


def _run_qutip(spec: SimulationSpec, spin_model: SpinHalfPauliModel, tlist: np.ndarray) -> EvolutionResult:
    """
    Run the QuTiP time evolution.

    Parameters
    ----------
    spec : SimulationSpec
        The simulation specification.
    spin_model : SpinModel
        The built spin model.
    tlist : np.ndarray
        The time grid.

    Returns
    -------
    EvolutionResult
        The QuTiP evolution result.
    """
    from manybody_backends.qutip.spinmodel import to_qutip_hamiltonian
    from manybody_backends.qutip.timeevolution import (
        manyspin_product_state,
        solve_master_history,
        solve_monte_carlo_history,
        solve_state_history,
    )

    initial_state = manyspin_product_state(
        spin_model.n_sites,
        spec.initial_state.theta,
        spec.initial_state.phi,
    )
    hamiltonian = to_qutip_hamiltonian(spin_model)
    metadata = _base_metadata(spec, tlist)
    collapse_ops = _build_qutip_collapse_ops(
        spec.collapse_operators,
        spin_model.n_sites,
    )

    if spec.method.algorithm == "mesolve":
        dataframe, states = solve_master_history(
            hamiltonian,
            initial_state,
            tlist,
            collapse_ops=collapse_ops,
            options=spec.method.evolution_params.get("options"),
            metadata=metadata,
        )
    elif spec.method.algorithm == "mcsolve":
        evolution_params = dict(spec.method.evolution_params)
        dataframe, states, solver_metadata = solve_monte_carlo_history(
            hamiltonian,
            initial_state,
            tlist,
            collapse_ops=collapse_ops,
            ntraj=evolution_params.get("ntraj") or 500,
            seeds=evolution_params.get("seeds"),
            target_tol=evolution_params.get("target_tol"),
            timeout=evolution_params.get("timeout"),
            options=evolution_params.get("options"),
            metadata=metadata,
        )
        metadata = {**metadata, "mcsolve": solver_metadata}
    else:
        dataframe, states = solve_state_history(
            hamiltonian,
            initial_state,
            tlist,
            metadata=metadata,
        )

    return _result_from_dataframe(dataframe, states, metadata)


def _run_quspin(spec: SimulationSpec, spin_model: SpinHalfPauliModel, tlist: np.ndarray) -> EvolutionResult:
    """
    Run the QuSpin time evolution.

    Parameters
    ----------
    spec : SimulationSpec
        The simulation specification.
    spin_model : SpinModel
        The built spin model.
    tlist : np.ndarray
        The time grid.

    Returns
    -------
    EvolutionResult
        The QuSpin evolution result.
    """
    from manybody_backends.quspin.spinmodel import (
        to_quspin_basis,
        to_quspin_hamiltonian,
    )
    from manybody_backends.quspin.timeevolution import (
        manyspin_product_state,
        solve_state_history,
    )

    basis = to_quspin_basis(spin_model)
    metadata = _base_metadata(spec, tlist)
    initial_state = manyspin_product_state(
        spin_model.n_sites,
        spec.initial_state.theta,
        spec.initial_state.phi,
        basis=basis,
    )
    dataframe, states = solve_state_history(
        to_quspin_hamiltonian(spin_model, basis=basis),
        initial_state,
        tlist,
        metadata=metadata,
    )
    return _result_from_dataframe(dataframe, states, metadata)
