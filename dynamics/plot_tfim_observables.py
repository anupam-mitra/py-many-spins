import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from time_evolution.config_io import load_json
from time_evolution.persistence import RunStore


BACKENDS = ("qutip", "quspin", "quimb", "tenpy")
DEFAULT_ALGORITHMS = {
    "qutip": "sesolve",
    "quspin": "evolve",
    "quimb": "TEBD",
    "tenpy": "TEBD",
}
SIGMA_Z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
SIGMA_ZZ = np.kron(SIGMA_Z, SIGMA_Z)


def _real_scalar(value, label):
    value = complex(np.asarray(value).reshape(()))
    if abs(value.imag) > 1e-8:
        raise ValueError("%s has non-negligible imaginary part %r" % (label, value))
    return float(value.real)


def _asarray(value):
    if hasattr(value, "to_ndarray"):
        return value.to_ndarray()
    if hasattr(value, "full"):
        return value.full()
    return np.asarray(value)


def _density_expectation(rho, operator, label):
    rho_array = _asarray(rho)
    return _real_scalar(np.trace(rho_array @ operator), label)


def _run_records(base_dir):
    runs_dir = base_dir / "runs"
    if not runs_dir.exists():
        raise ValueError("no runs directory found at %s" % (runs_dir,))

    records = []
    for run_dir in sorted(item for item in runs_dir.iterdir() if item.is_dir()):
        manifest_path = run_dir / "manifest.json"
        spec_path = run_dir / "spec.json"
        states_path = run_dir / "states" / "index.json"
        if not manifest_path.exists() or not spec_path.exists() or not states_path.exists():
            continue

        manifest = load_json(manifest_path)
        spec = load_json(spec_path)
        states_index = load_json(states_path)
        metadata = dict(states_index.get("metadata") or manifest.get("metadata") or {})
        method = spec.get("method", {})
        model = spec.get("model", {})
        backend = str(metadata.get("backend") or method.get("backend")).lower()
        algorithm = metadata.get("algorithm") or method.get("algorithm")
        records.append({
            "run_dir": run_dir,
            "run_id": run_dir.name,
            "backend": backend,
            "algorithm": algorithm,
            "n_sites": int(metadata.get("n_sites") or model.get("n_sites")),
            "time_count": int(metadata.get("time_count") or len(states_index["records"])),
            "updated_at": manifest.get("updated_at") or manifest.get("created_at") or "",
            "metadata": metadata,
            "spec": spec,
        })
    return records


def _select_latest_runs(base_dir, backends):
    records = _run_records(base_dir)
    selected = {}
    for backend in backends:
        algorithm = DEFAULT_ALGORITHMS[backend]
        matches = [
            record for record in records
            if record["backend"] == backend and record["algorithm"] == algorithm
        ]
        if not matches:
            raise ValueError(
                "no %s/%s run found under %s" % (backend, algorithm, base_dir)
            )
        selected[backend] = max(
            matches,
            key=lambda record: (record["time_count"], record["updated_at"], record["run_id"]),
        )
    return selected


def _load_states(base_dir, run_record):
    store = RunStore(base_dir, run_id=run_record["run_id"])
    index = store.load_evolution_index()
    times = np.asarray([record["time"] for record in index["records"]], dtype=float)
    states = [store.load_payload(record["storage"]) for record in index["records"]]
    return times, states


def _marginal_observables(states, n_sites, local_marginal):
    sum_z = []
    sum_zz_offdiag = []
    for state in states:
        one_site_total = 0.0
        for site in range(n_sites):
            one_site_total += _density_expectation(
                local_marginal(state, (site,)),
                SIGMA_Z,
                "site %d sigma_z" % site,
            )

        pair_total = 0.0
        for left in range(n_sites):
            for right in range(left + 1, n_sites):
                pair_total += 2.0 * _density_expectation(
                    local_marginal(state, (left, right)),
                    SIGMA_ZZ,
                    "sites %d,%d sigma_z sigma_z" % (left, right),
                )

        sum_z.append(one_site_total)
        sum_zz_offdiag.append(pair_total)
    return np.asarray(sum_z), np.asarray(sum_zz_offdiag)


def _qutip_observables(states, n_sites):
    from manybody_backends.qutip.timeevolution import local_marginal_density_matrix

    return _marginal_observables(states, n_sites, local_marginal_density_matrix)


def _quspin_observables(states, n_sites):
    from manybody_backends.quspin.timeevolution import (
        local_marginal_density_matrix,
        spinhalf_basis,
    )

    basis = spinhalf_basis(n_sites)
    return _marginal_observables(
        states,
        n_sites,
        lambda state, sites: local_marginal_density_matrix(state, sites, basis),
    )


def _quimb_observables(states, n_sites):
    from manybody_backends.quimb.quimbtebd import local_marginal_density_matrix

    return _marginal_observables(states, n_sites, local_marginal_density_matrix)


def _tenpy_observables(states, n_sites):
    sum_z = []
    sum_zz_offdiag = []
    for state in states:
        sum_z.append(
            _real_scalar(np.sum(state.expectation_value("Sigmaz")), "TenPy sum_z")
        )

        pair_total = 0.0
        for left in range(n_sites):
            for right in range(left + 1, n_sites):
                pair_total += 2.0 * _real_scalar(
                    state.expectation_value_term([
                        ("Sigmaz", left),
                        ("Sigmaz", right),
                    ]),
                    "TenPy sites %d,%d sigma_z sigma_z" % (left, right),
                )
        sum_zz_offdiag.append(pair_total)
    return np.asarray(sum_z), np.asarray(sum_zz_offdiag)


def _observable_data(base_dir, selected_runs):
    calculators = {
        "qutip": _qutip_observables,
        "quspin": _quspin_observables,
        "quimb": _quimb_observables,
        "tenpy": _tenpy_observables,
    }
    data = {}
    for backend, run_record in selected_runs.items():
        times, states = _load_states(base_dir, run_record)
        sum_z, sum_zz_offdiag = calculators[backend](states, run_record["n_sites"])
        data[backend] = {
            "time": times,
            "sum_z": sum_z,
            "sum_zz_offdiag": sum_zz_offdiag,
        }
    return data


def _json_attr(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return value


def _write_hdf5(output_path, data, selected_runs):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5file:
        h5file.attrs["model"] = "tilted_field_ising_1d"
        h5file.attrs["sum_z"] = "sum_j <sigma^z_j>"
        h5file.attrs["sum_zz_offdiag"] = "sum_{j != k} <sigma^z_j sigma^z_k>"
        h5file.attrs["pair_counting"] = "ordered off-diagonal pairs"

        for backend, values in data.items():
            group = h5file.create_group(backend)
            group.create_dataset("time", data=values["time"])
            group.create_dataset("sum_z", data=values["sum_z"])
            group.create_dataset("sum_zz_offdiag", data=values["sum_zz_offdiag"])

            run_record = selected_runs[backend]
            metadata = run_record["metadata"]
            group.attrs["run_id"] = run_record["run_id"]
            group.attrs["algorithm"] = run_record["algorithm"]
            group.attrs["n_sites"] = run_record["n_sites"]
            group.attrs["source_run_dir"] = str(run_record["run_dir"])
            if metadata.get("bonddim") is not None:
                group.attrs["bonddim"] = metadata["bonddim"]
            for key, value in metadata.items():
                if key in group.attrs:
                    continue
                group.attrs[key] = _json_attr(value)


def _write_plot(output_path, data):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), sharex=True)

    for backend in BACKENDS:
        if backend not in data:
            continue
        axes[0].plot(data[backend]["time"], data[backend]["sum_z"], label=backend)
        axes[1].plot(
            data[backend]["time"],
            data[backend]["sum_zz_offdiag"],
            label=backend,
        )

    axes[0].set_ylabel(r"$\sum_j \langle \sigma^z_j \rangle$")
    axes[1].set_ylabel(r"$\sum_{j \ne k} \langle \sigma^z_j \sigma^z_k \rangle$")
    axes[1].set_xlabel("time")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot and persist TFIM observables from unified runs.",
    )
    parser.add_argument("--base-dir", default="../pkl/tfim_example")
    parser.add_argument("--output-h5")
    parser.add_argument("--output-plot")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_h5 = Path(args.output_h5) if args.output_h5 else (
        base_dir / "observables" / "tfim_observables.h5"
    )
    output_plot = Path(args.output_plot) if args.output_plot else (
        base_dir / "plots" / "tfim_observables.png"
    )

    selected_runs = _select_latest_runs(base_dir, BACKENDS)
    data = _observable_data(base_dir, selected_runs)
    _write_hdf5(output_h5, data, selected_runs)
    _write_plot(output_plot, data)

    print(output_h5)
    print(output_plot)


if __name__ == "__main__":
    main()
