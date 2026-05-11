from dataclasses import replace
import time
import uuid

import numpy as np

from manybody_util.spinmodel import tilted_field_ising_1d
from time_evolution.persistence import RunStore
from time_evolution.results import EvolutionResult
from time_evolution.specs import EXACT_BACKENDS


def build_spin_model(model_spec):
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


def build_time_grid(time_grid_spec):
    time_grid_spec.validate()
    if time_grid_spec.kind == "linspace":
        return np.linspace(
            time_grid_spec.start,
            time_grid_spec.stop,
            time_grid_spec.num,
        )
    raise ValueError("unsupported time grid kind %r" % (time_grid_spec.kind,))


def run_simulation(spec):
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


def run_and_store(spec):
    store = RunStore(spec.output.base_dir, run_id=spec.output.run_id)
    stored_spec = spec
    if spec.output.run_id != store.run_id:
        stored_spec = replace(spec, output=replace(spec.output, run_id=store.run_id))

    result = run_simulation(stored_spec)
    index = store.save_evolution_result(stored_spec, result)
    return store, index


def _base_metadata(spec, tlist):
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


def _records_from_dataframe(dataframe):
    return dataframe.to_dict(orient="records")


def _state_records(tlist, bonddim=None, trajectory_id=None):
    records = []
    for ix_time, time_value in enumerate(tlist):
        record = {
            "ix_time": ix_time,
            "time": time_value,
            "uuid_str": "%s" % uuid.uuid4(),
            "walltime": time.time(),
        }
        if bonddim is not None:
            record["bonddim"] = bonddim
        if trajectory_id is not None:
            record["trajectory_id"] = trajectory_id
        records.append(record)
    return records


def _local_spin_matrix(operator):
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


def _quimb_split_options(trunc_params):
    params = trunc_params or {}
    options = {}
    if params.get("chi_max") is not None:
        options["max_bond"] = params["chi_max"]
    if params.get("cutoff") is not None:
        options["cutoff"] = params["cutoff"]
    elif params.get("svd_min") is not None:
        options["cutoff"] = params["svd_min"]
    return options


def _build_quimb_collapse_ops(collapse_specs, n_sites):
    collapse_ops = []
    for collapse in collapse_specs:
        if collapse.rate == 0.0:
            continue
        collapse_ops.append({
            "operator": np.sqrt(collapse.rate) * _local_spin_matrix(collapse.operator),
            "sites": collapse.expanded_sites(n_sites),
        })
    return collapse_ops


def _qutip_local_spin_operator(operator):
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


def _qutip_embed_local_operator(n_sites, operator, site):
    import qutip

    factors = [qutip.qeye(2) for _ in range(n_sites)]
    factors[site] = operator
    return qutip.tensor(factors)


def _build_qutip_collapse_ops(collapse_specs, n_sites):
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


def _run_tenpy(spec, spin_model, tlist):
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
    return EvolutionResult(
        records=_records_from_dataframe(dataframe),
        states=states,
        metadata=metadata,
    )


def _quimb_evolution_params(spec, tlist):
    params = dict(spec.method.evolution_params)
    if len(tlist) > 1 and params.get("dt") is None and params.get("tol") is None:
        params["dt"] = float(tlist[1] - tlist[0])
    return params


def _run_quimb(spec, spin_model, tlist):
    import quimb.tensor as qtn

    from manybody_backends.quimb.quimbtebd import TEBDWrapper, spinhalf_state
    from manybody_backends.quimb.quimbmpstrajectory import solve_mps_trajectory
    from manybody_backends.quimb.spinmodel import to_quimb_spinham1d

    initial_mps = qtn.MPS_product_state(
        [spinhalf_state(spec.initial_state.theta, spec.initial_state.phi)]
        * spin_model.n_sites,
        cyclic=(spec.model.bc == "periodic"),
    )
    builder = to_quimb_spinham1d(spin_model)
    if spec.method.algorithm == "MCWF":
        evolution_params = dict(spec.method.evolution_params)
        tebd_params = dict(evolution_params.get("tebd_params", {}))
        n_substeps = int(evolution_params.get("n_substeps", 1))
        for key in ("dt", "tol", "order", "progbar"):
            if key in evolution_params and evolution_params[key] is not None:
                tebd_params[key] = evolution_params[key]
        if (
            len(tlist) > 1
            and tebd_params.get("dt") is None
            and tebd_params.get("tol") is None
        ):
            tebd_params["dt"] = float(tlist[1] - tlist[0]) / n_substeps

        trajectory = solve_mps_trajectory(
            initial_mps,
            builder.build_local_ham(spin_model.n_sites),
            _build_quimb_collapse_ops(spec.collapse_operators, spin_model.n_sites),
            tlist,
            n_substeps=n_substeps,
            tebd_params=tebd_params,
            split_opts=_quimb_split_options(spec.method.trunc_params),
            random_numbers=evolution_params.get("random_numbers"),
            seed=evolution_params.get("seed"),
            metadata=_base_metadata(spec, tlist),
        )
        metadata = {
            **_base_metadata(spec, tlist),
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
    return EvolutionResult(
        records=_records_from_dataframe(dataframe),
        states=states,
        metadata=_base_metadata(spec, tlist),
    )


def _run_qutip(spec, spin_model, tlist):
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

    return EvolutionResult(
        records=_records_from_dataframe(dataframe),
        states=states,
        metadata=metadata,
    )


def _run_quspin(spec, spin_model, tlist):
    from manybody_backends.quspin.spinmodel import (
        to_quspin_basis,
        to_quspin_hamiltonian,
    )
    from manybody_backends.quspin.timeevolution import (
        manyspin_product_state,
        solve_state_history,
    )

    basis = to_quspin_basis(spin_model)
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
        metadata=_base_metadata(spec, tlist),
    )
    return EvolutionResult(
        records=_records_from_dataframe(dataframe),
        states=states,
        metadata=_base_metadata(spec, tlist),
    )
