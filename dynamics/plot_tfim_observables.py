import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from time_evolution.config_io import load_json
from time_evolution.persistence import RunStore


DEMO_RUNS = (
    {"label": "qutip", "backend": "qutip", "algorithm": "sesolve"},
    {"label": "quspin", "backend": "quspin", "algorithm": "evolve"},
    {"label": "quimb", "backend": "quimb", "algorithm": "TEBD"},
    {"label": "tenpy_TEBD", "backend": "tenpy", "algorithm": "TEBD"},
    {"label": "tenpy_TDVP", "backend": "tenpy", "algorithm": "TDVP"},
    {"label": "tenpy_ExpMPO", "backend": "tenpy", "algorithm": "ExpMPO"},
)
SIGMA_Z = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
OBSERVABLE_SPECS = (
    (1, "z1", "sum_i <sigma^z_i> / N"),
    (2, "z2_offdiag", "sum_{i != j} <sigma^z_i sigma^z_j> / N^2"),
    (
        3,
        "z3_offdiag",
        "sum_{distinct i,j,k} <sigma^z_i sigma^z_j sigma^z_k> / N^3",
    ),
    (
        4,
        "z4_offdiag",
        "sum_{distinct i,j,k,l} <sigma^z_i sigma^z_j sigma^z_k sigma^z_l> / N^4",
    ),
)
DESCRIPTION_BY_DATASET = {
    dataset: description for _, dataset, description in OBSERVABLE_SPECS
}
ORDER_BY_DATASET = {dataset: order for order, dataset, _ in OBSERVABLE_SPECS}


def _kron_power(operator, order):
    result = np.asarray([[1.0]], dtype=complex)
    for _ in range(order):
        result = np.kron(result, operator)
    return result


Z_OPERATORS = {
    order: _kron_power(SIGMA_Z, order)
    for order, _, _ in OBSERVABLE_SPECS
}


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


def _select_latest_runs(base_dir, run_specs):
    records = _run_records(base_dir)
    selected = {}
    for run_spec in run_specs:
        label = run_spec["label"]
        backend = run_spec["backend"]
        algorithm = run_spec["algorithm"]
        matches = [
            record for record in records
            if record["backend"] == backend and record["algorithm"] == algorithm
        ]
        if not matches:
            raise ValueError(
                "no %s/%s run found under %s" % (backend, algorithm, base_dir)
            )
        selected[label] = max(
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


def _normalized_marginal_observables(states, n_sites, local_marginal):
    observables = {dataset: [] for _, dataset, _ in OBSERVABLE_SPECS}
    for state in states:
        for order, dataset, _ in OBSERVABLE_SPECS:
            total = 0.0
            for sites in itertools.combinations(range(n_sites), order):
                label = "sites %s order-%d sigma_z" % (
                    ",".join(str(site) for site in sites),
                    order,
                )
                total += math.factorial(order) * _density_expectation(
                    local_marginal(state, sites),
                    Z_OPERATORS[order],
                    label,
                )
            observables[dataset].append(total / (n_sites ** order))

    return {
        dataset: np.asarray(values)
        for dataset, values in observables.items()
    }


def _qutip_observables(states, n_sites):
    from manybody_backends.qutip.timeevolution import local_marginal_density_matrix

    return _normalized_marginal_observables(
        states,
        n_sites,
        local_marginal_density_matrix,
    )


def _quspin_observables(states, n_sites):
    from manybody_backends.quspin.timeevolution import (
        local_marginal_density_matrix,
        spinhalf_basis,
    )

    basis = spinhalf_basis(n_sites)
    return _normalized_marginal_observables(
        states,
        n_sites,
        lambda state, sites: local_marginal_density_matrix(state, sites, basis),
    )


def _quimb_observables(states, n_sites):
    from manybody_backends.quimb.quimbtebd import local_marginal_density_matrix

    return _normalized_marginal_observables(
        states,
        n_sites,
        local_marginal_density_matrix,
    )


def _tenpy_observables(states, n_sites):
    observables = {dataset: [] for _, dataset, _ in OBSERVABLE_SPECS}
    for state in states:
        for order, dataset, _ in OBSERVABLE_SPECS:
            total = 0.0
            for sites in itertools.combinations(range(n_sites), order):
                total += math.factorial(order) * _real_scalar(
                    state.expectation_value_term([
                        ("Sigmaz", site) for site in sites
                    ]),
                    "TenPy sites %s order-%d sigma_z" % (
                        ",".join(str(site) for site in sites),
                        order,
                    ),
                )
            observables[dataset].append(total / (n_sites ** order))

    return {
        dataset: np.asarray(values)
        for dataset, values in observables.items()
    }


def _observable_data(base_dir, selected_runs):
    calculators = {
        "qutip": _qutip_observables,
        "quspin": _quspin_observables,
        "quimb": _quimb_observables,
        "tenpy": _tenpy_observables,
    }
    data = {}
    for label, run_record in selected_runs.items():
        times, states = _load_states(base_dir, run_record)
        backend = run_record["backend"]
        data[label] = {
            "time": times,
            **calculators[backend](states, run_record["n_sites"]),
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
        h5file.attrs["normalization"] = "N^order"
        h5file.attrs["site_counting"] = "ordered distinct-site tuples"
        for dataset, description in DESCRIPTION_BY_DATASET.items():
            h5file.attrs[dataset] = description

        for label, values in data.items():
            group = h5file.create_group(label)
            group.create_dataset("time", data=values["time"])
            for _, dataset, _ in OBSERVABLE_SPECS:
                h5_dataset = group.create_dataset(dataset, data=values[dataset])
                h5_dataset.attrs["description"] = DESCRIPTION_BY_DATASET[dataset]
                h5_dataset.attrs["normalization"] = "N^%d" % ORDER_BY_DATASET[dataset]
                h5_dataset.attrs["site_counting"] = "ordered distinct-site tuples"

            run_record = selected_runs[label]
            metadata = run_record["metadata"]
            group.attrs["run_id"] = run_record["run_id"]
            group.attrs["backend"] = run_record["backend"]
            group.attrs["algorithm"] = run_record["algorithm"]
            group.attrs["n_sites"] = run_record["n_sites"]
            group.attrs["source_run_dir"] = str(run_record["run_dir"])
            if metadata.get("bonddim") is not None:
                group.attrs["bonddim"] = metadata["bonddim"]
            for key, value in metadata.items():
                if key in group.attrs:
                    continue
                group.attrs[key] = _json_attr(value)


def _write_plot(output_path, data, selected_runs):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(8.0, 9.5), sharex=True)

    for run_spec in DEMO_RUNS:
        label = run_spec["label"]
        if label not in data:
            continue
        run_record = selected_runs[label]
        algorithm = run_record["algorithm"]
        bonddim = run_record["metadata"].get("bonddim")
        if bonddim is not None:
            plot_label = f"{algorithm} (χ={bonddim})"
        else:
            plot_label = algorithm
        for axis, (_, dataset, _) in zip(axes, OBSERVABLE_SPECS):
            axis.plot(
                data[label]["time"],
                data[label][dataset],
                label=plot_label,
            )

    axes[0].set_ylabel(r"$\sum_i \langle Z_i \rangle / N$")
    axes[1].set_ylabel(r"$\sum_{i \ne j} \langle Z_i Z_j \rangle / N^2$")
    axes[2].set_ylabel(r"$\sum_{i \ne j \ne k} \langle Z_i Z_j Z_k \rangle / N^3$")
    axes[3].set_ylabel(r"$\sum_{i \ne j \ne k \ne l} \langle Z_i Z_j Z_k Z_l \rangle / N^4$")
    axes[3].set_xlabel("time")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_ylim(-1.0, 1.0)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot and persist TFIM observables from unified runs.",
    )
    parser.add_argument("--base-dir", default="../pkl/tfim_10spin")
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

    selected_runs = _select_latest_runs(base_dir, DEMO_RUNS)
    data = _observable_data(base_dir, selected_runs)
    _write_hdf5(output_h5, data, selected_runs)
    _write_plot(output_plot, data, selected_runs)

    print(output_h5)
    print(output_plot)


if __name__ == "__main__":
    main()
