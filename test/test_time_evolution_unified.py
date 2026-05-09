import numpy as np
import pytest

from time_evolution.config_io import dump_simulation_spec, load_json, load_simulation_spec
from time_evolution.persistence import RunStore
from time_evolution.results import EvolutionResult, MarginalResult
from time_evolution.runner import build_spin_model, build_time_grid, run_and_store
from time_evolution.specs import (
    InitialStateSpec,
    MethodSpec,
    ModelSpec,
    OutputSpec,
    SimulationSpec,
    TimeGridSpec,
)


def _spec(tmp_path, backend="qutip", algorithm="sesolve", run_id=None):
    return SimulationSpec(
        model=ModelSpec.tilted_field_ising_1d(
            n_sites=2,
            j_xx=0.0,
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
            trunc_params={},
            evolution_params={},
        ),
        output=OutputSpec(base_dir=str(tmp_path), run_id=run_id),
    )


def test_simulation_spec_json_roundtrip(tmp_path):
    spec = _spec(tmp_path, run_id="roundtrip-run")
    path = tmp_path / "configs" / "spec.json"

    dump_simulation_spec(spec, path)
    loaded = load_simulation_spec(path)

    assert loaded == spec


def test_simulation_spec_validation_rejects_invalid_methods(tmp_path):
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


def test_run_store_persists_pickle_objects_and_hdf5_arrays(tmp_path):
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


def test_run_store_persists_marginal_results(tmp_path):
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


def test_runner_builds_shared_model_and_time_grid(tmp_path):
    spec = _spec(tmp_path)
    model = build_spin_model(spec.model)
    tlist = build_time_grid(spec.time_grid)

    assert model.n_sites == 2
    assert model.metadata["family"] == "tilted_field_ising_1d"
    np.testing.assert_allclose(tlist, [0.0, 0.05])


def test_qutip_runner_smoke_and_pickle_persistence(tmp_path):
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


def test_quspin_runner_smoke_omits_bonddim_labels(tmp_path):
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
