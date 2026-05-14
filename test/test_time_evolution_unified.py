from pathlib import Path

import numpy as np
import pytest

from time_evolution.config_io import dump_simulation_spec, load_json, load_simulation_spec
from time_evolution.persistence import RunStore
from time_evolution.results import EvolutionResult, MarginalResult
from time_evolution.runner import build_spin_model, build_time_grid, run_and_store
from time_evolution.specs import (
    CollapseOperatorSpec,
    InitialStateSpec,
    MethodSpec,
    ModelSpec,
    OutputSpec,
    SimulationSpec,
    TimeGridSpec,
)


def _collapse() -> CollapseOperatorSpec:
    """
    Helper to create a standard local spin collapse operator specification.

    Returns
    -------
    CollapseOperatorSpec
        A specification for a sigma-minus collapse operator.
    """
    return CollapseOperatorSpec.local_spin("sigmam", rate=0.2, sites="all")


def _spec(
    tmp_path: Path,
    backend: str = "qutip",
    algorithm: str = "sesolve",
    run_id: str | None = None,
    collapse_operators: tuple = (),
    evolution_params: dict | None = None,
    trunc_params: dict | None = None,
    j_xx: float = 0.0,
) -> SimulationSpec:
    """
    Helper to create a SimulationSpec for unified evolution tests.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for output.
    backend : str, optional
        The simulation backend to use (default "qutip").
    algorithm : str, optional
        The evolution algorithm to use (default "sesolve").
    run_id : str, optional
        The identifier for the run.
    collapse_operators : tuple, optional
        A tuple of collapse operator specifications (default ()).
    evolution_params : dict, optional
        Algorithm-specific evolution parameters.
    trunc_params : dict, optional
        Algorithm-specific truncation parameters.
    j_xx : float, optional
        Interaction strength for the tilted-field Ising model (default 0.0).

    Returns
    -------
    SimulationSpec
        A fully configured simulation specification.
    """
    return SimulationSpec(
        model=ModelSpec.tilted_field_ising_1d(
            n_sites=2,
            j_xx=j_xx,
            b_z=0.2,
            b_x=0.0,
            bc="open",
        ),
        initial_state=InitialStateSpec.spin_coherent_product(
            theta=np.pi / 2,
            phi=0.0,
        ),
        time_grid=TimeGridSpec.linspace(0.0, 0.05, 2),
        method=MethodSpec(
            backend=backend,
            algorithm=algorithm,
            trunc_params=dict(trunc_params or {}),
            evolution_params=dict(evolution_params or {}),
        ),
        collapse_operators=tuple(collapse_operators),
        output=OutputSpec(base_dir=str(tmp_path), run_id=run_id),
    )


def test_simulation_spec_json_roundtrip(tmp_path: Path) -> None:
    """
    Test that a SimulationSpec can be dumped to and loaded from JSON without loss.

    Verifies that a spec containing dissipative operators is preserved
    through the roundtrip.
    """
    spec = _spec(
        tmp_path,
        backend="qutip",
        algorithm="mesolve",
        run_id="roundtrip-run",
        collapse_operators=(_collapse(),),
    )
    path = tmp_path / "configs" / "spec.json"

    dump_simulation_spec(spec, path)
    loaded = load_simulation_spec(path)

    assert loaded == spec


def test_simulation_spec_validation_rejects_invalid_methods(tmp_path: Path) -> None:
    """
    Test that SimulationSpec validation rejects unsupported backends, algorithms, and parameters.

    Verifies that ValueError is raised when providing invalid method configurations.
    """
    data = _spec(tmp_path).to_dict()
    data["method"]["backend"] = "bad-backend"
    with pytest.raises(ValueError, match="unsupported backend"):
        SimulationSpec.from_dict(data)

    data = _spec(tmp_path).to_dict()
    data["method"]["algorithm"] = "bad-algorithm"
    with pytest.raises(ValueError, match="unsupported algorithm"):
        SimulationSpec.from_dict(data)

    data = _spec(tmp_path).to_dict()
    data["method"]["trunc_params"] = {"chi_max": 4}
    with pytest.raises(ValueError, match="truncation parameters"):
        SimulationSpec.from_dict(data)

    data = _spec(tmp_path).to_dict()
    data["collapse_operators"] = [_collapse().to_dict()]
    with pytest.raises(ValueError, match="collapse operators"):
        SimulationSpec.from_dict(data)


def test_simulation_spec_validation_accepts_dissipative_methods(tmp_path: Path) -> None:
    """
    Test that SimulationSpec validation accepts valid dissipative method configurations.

    Verifies that mesolve, mcsolve, and MCWF specifications are accepted
    when collapse operators are provided.
    """
    collapse_operators = (_collapse(),)

    SimulationSpec.from_dict(
        _spec(
            tmp_path,
            backend="qutip",
            algorithm="mesolve",
            collapse_operators=collapse_operators,
        ).to_dict()
    )
    SimulationSpec.from_dict(
        _spec(
            tmp_path,
            backend="qutip",
            algorithm="mcsolve",
            collapse_operators=collapse_operators,
            evolution_params={"ntraj": 2, "seeds": 123},
        ).to_dict()
    )
    SimulationSpec.from_dict(
        _spec(
            tmp_path,
            backend="quimb",
            algorithm="MCWF",
            collapse_operators=collapse_operators,
            trunc_params={"chi_max": 4},
        ).to_dict()
    )


def test_simulation_spec_normalizes_optional_collapse_fields(tmp_path: Path) -> None:
    """
    Test that SimulationSpec normalizes collapse operator fields during instantiation.

    Verifies that None is converted to an empty tuple and that site indices
    are normalized to tuples.
    """
    data = _spec(tmp_path).to_dict()
    data["collapse_operators"] = None
    loaded = SimulationSpec.from_dict(data)
    assert loaded.collapse_operators == ()

    data = _spec(
        tmp_path,
        backend="qutip",
        algorithm="mesolve",
        collapse_operators=(CollapseOperatorSpec.local_spin("sigmam", 0.2, sites=1),),
    ).to_dict()
    data["collapse_operators"][0]["sites"] = 1
    loaded = SimulationSpec.from_dict(data)
    assert loaded.collapse_operators[0].sites == (1,)


def test_run_store_persists_pickle_objects_and_hdf5_arrays(tmp_path: Path) -> None:
    """
    Test that RunStore correctly persists both pickle and HDF5 payloads.

    Verifies that the storage kind is correctly recorded in the index and
    that payloads can be retrieved and verified.
    """
    spec = _spec(tmp_path, run_id="store-run")
    store = RunStore(tmp_path, run_id="store-run")
    array_payload = np.asarray([1.0 + 2.0j, 3.0 + 4.0j])
    result = EvolutionResult(
        records=[
            {"ix_time": 0, "time": np.float64(0.0), "uuid_str": "object-state"},
            {"ix_time": 1, "time": np.float64(0.1), "uuid_str": "array-state"},
        ],
        states=[{"payload": "object"}, array_payload],
        metadata={"backend": "test"},
    )

    index = store.save_evolution_result(spec, result)
    manifest = load_json(store.run_dir / "manifest.json")

    assert manifest["states"] == "states/index.json"
    assert index["records"][0]["storage"]["kind"] == "pickle"
    assert index["records"][1]["storage"]["kind"] == "hdf5"
    assert store.load_payload(index["records"][0]["storage"]) == {"payload": "object"}
    np.testing.assert_allclose(
        store.load_payload(index["records"][1]["storage"]),
        array_payload,
    )


def test_run_store_persists_marginal_results(tmp_path: Path) -> None:
    """
    Test that RunStore correctly persists marginal result data.

    Verifies that marginal results are saved as HDF5 arrays and that
    the manifest is updated accordingly.
    """
    store = RunStore(tmp_path, run_id="marginal-run")
    marginal = np.eye(2, dtype=np.complex128)
    result = MarginalResult(
        records=[{"sites_sel": (0,), "time": 0.0}],
        marginals=[marginal],
        metadata={"source": "unit-test"},
    )

    index = store.save_marginal_result(result, marginal_run_id="one-site")
    manifest = load_json(store.run_dir / "manifest.json")

    assert manifest["marginal_runs"] == [
        {"marginal_run_id": "one-site", "index": "marginals/one-site/index.json"}
    ]
    assert index["records"][0]["storage"]["kind"] == "hdf5"
    np.testing.assert_allclose(
        store.load_payload(index["records"][0]["storage"]),
        marginal,
    )


def test_runner_builds_shared_model_and_time_grid(tmp_path: Path) -> None:
    """
    Test that the runner correctly builds the shared spin model and time grid.

    Verifies that the built model and time array match the specification.
    """
    spec = _spec(tmp_path)
    model = build_spin_model(spec.model)
    tlist = build_time_grid(spec.time_grid)

    assert model.n_sites == 2
    assert model.metadata["family"] == "tilted_field_ising_1d"
    np.testing.assert_allclose(tlist, [0.0, 0.05])


def test_qutip_runner_smoke_and_pickle_persistence(tmp_path: Path) -> None:
    """
    Smoke test for the QuTiP runner and its pickle persistence.

    Verifies that a simple evolution can be run and stored, and that
    the resulting payload is a valid QuTiP object.
    """
    pytest.importorskip("qutip")

    spec = _spec(tmp_path, backend="qutip", algorithm="sesolve", run_id="qutip-run")
    store, index = run_and_store(spec)
    manifest = load_json(store.run_dir / "manifest.json")

    assert store.run_id == "qutip-run"
    assert len(index["records"]) == 2
    assert "bonddim" not in index["metadata"]
    assert "bonddim" not in manifest["metadata"]
    assert "bonddim" not in index["records"][0]
    assert index["records"][0]["storage"]["kind"] == "pickle"
    assert store.load_payload(index["records"][0]["storage"]).dims == [[2, 2], [1]]


def test_qutip_mesolve_runner_smoke_with_collapse_ops(tmp_path: Path) -> None:
    """
    Smoke test for the QuTiP mesolve runner with dissipative operators.

    Verifies that the master equation evolution correctly stores metadata
    and the resulting density matrix.
    """
    pytest.importorskip("qutip")

    spec = _spec(
        tmp_path,
        backend="qutip",
        algorithm="mesolve",
        run_id="mesolve-run",
        collapse_operators=(_collapse(),),
    )
    store, index = run_and_store(spec)

    assert len(index["records"]) == 2
    assert index["metadata"]["algorithm"] == "mesolve"
    assert index["metadata"]["collapse_operators"] == [_collapse().to_dict()]
    assert index["records"][0]["storage"]["kind"] == "pickle"
    assert store.load_payload(index["records"][0]["storage"]).dims == [[2, 2], [2, 2]]


def test_qutip_mcsolve_runner_smoke_with_collapse_ops(tmp_path: Path) -> None:
    """
    Smoke test for the QuTiP mcsolve runner with dissipative operators.

    Verifies that the Monte Carlo evolution correctly stores trajectory
    metadata and the resulting state.
    """
    pytest.importorskip("qutip")

    spec = _spec(
        tmp_path,
        backend="qutip",
        algorithm="mcsolve",
        run_id="mcsolve-run",
        collapse_operators=(_collapse(),),
        evolution_params={"ntraj": 2, "seeds": 123},
    )
    store, index = run_and_store(spec)

    assert len(index["records"]) == 2
    assert index["metadata"]["algorithm"] == "mcsolve"
    assert index["metadata"]["mcsolve"]["ntraj"] == 2
    assert len(index["metadata"]["mcsolve"]["col_times"]) == 2
    assert index["records"][0]["storage"]["kind"] == "pickle"
    assert store.load_payload(index["records"][0]["storage"]).dims == [[2, 2], [2, 2]]


def test_quspin_runner_smoke_omits_bonddim_labels(tmp_path: Path) -> None:
    """
    Smoke test for the QuSpin runner.

    Verifies that QuSpin evolution (which uses exact states) correctly omits
    bond-dimension metadata and stores the result as an HDF5 array.
    """
    pytest.importorskip("quspin")

    spec = _spec(tmp_path, backend="quspin", algorithm="evolve", run_id="quspin-run")
    store, index = run_and_store(spec)
    manifest = load_json(store.run_dir / "manifest.json")

    assert store.run_id == "quspin-run"
    assert len(index["records"]) == 2
    assert "bonddim" not in index["metadata"]
    assert "bonddim" not in manifest["metadata"]
    assert "bonddim" not in index["records"][0]
    assert index["records"][0]["storage"]["kind"] == "hdf5"
    assert store.load_payload(index["records"][0]["storage"]).shape == (4,)


def test_quimb_mcwf_runner_smoke_with_collapse_ops(tmp_path: Path) -> None:
    """
    Smoke test for the Quimb MCWF runner with dissipative operators.

    Verifies that the MPS-based MCWF evolution correctly records
    jump events and stores the resulting MPS.
    """
    pytest.importorskip("quimb.tensor")

    spec = _spec(
        tmp_path,
        backend="quimb",
        algorithm="MCWF",
        run_id="quimb-mcwf-run",
        collapse_operators=(CollapseOperatorSpec.local_spin("sigmam", 10.0),),
        evolution_params={"random_numbers": [0.99, 0.1]},
        trunc_params={"chi_max": 4},
        j_xx=0.1,
    )
    store, index = run_and_store(spec)

    assert len(index["records"]) == 2
    assert index["metadata"]["algorithm"] == "MCWF"
    assert index["metadata"]["tjumps"]
    assert index["metadata"]["random_numbers"] == {
        "jump_thresholds": [0.99],
        "jump_choices": [0.1],
    }
    assert index["records"][0]["storage"]["kind"] == "pickle"
    assert store.load_payload(index["records"][0]["storage"]).L == 2
