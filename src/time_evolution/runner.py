from dataclasses import replace

import numpy as np

from manybody_util.spinmodel import tilted_field_ising_1d
from time_evolution.persistence import RunStore
from time_evolution.results import EvolutionResult


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
    return {
        "backend": spec.method.backend.lower(),
        "algorithm": spec.method.algorithm,
        "bonddim": spec.method.trunc_params.get("chi_max"),
        "n_sites": spec.model.n_sites,
        "time_count": int(len(tlist)),
    }


def _records_from_dataframe(dataframe):
    return dataframe.to_dict(orient="records")


def _run_tenpy(spec, spin_model, tlist):
    from wrap_tenpy.spinmodel import to_tenpy_model
    from wrap_tenpy.timeevolution import manyspin_product_mps, solve_mps_history

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

    from wrap_quimb.quimbtebd import TEBDWrapper, spinhalf_state
    from wrap_quimb.spinmodel import to_quimb_spinham1d

    initial_mps = qtn.MPS_product_state(
        [spinhalf_state(spec.initial_state.theta, spec.initial_state.phi)]
        * spin_model.n_sites,
        cyclic=(spec.model.bc == "periodic"),
    )
    wrapper = TEBDWrapper(
        to_quimb_spinham1d(spin_model),
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
    from wrap_qutip.spinmodel import to_qutip_hamiltonian
    from wrap_qutip.timeevolution import manyspin_product_state, solve_state_history

    initial_state = manyspin_product_state(
        spin_model.n_sites,
        spec.initial_state.theta,
        spec.initial_state.phi,
    )
    dataframe, states = solve_state_history(
        to_qutip_hamiltonian(spin_model),
        initial_state,
        tlist,
        metadata=_base_metadata(spec, tlist),
    )
    return EvolutionResult(
        records=_records_from_dataframe(dataframe),
        states=states,
        metadata=_base_metadata(spec, tlist),
    )


def _run_quspin(spec, spin_model, tlist):
    from wrap_quspin.spinmodel import to_quspin_basis, to_quspin_hamiltonian
    from wrap_quspin.timeevolution import manyspin_product_state, solve_state_history

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
