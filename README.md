# Simulation of many body quantum dynamics
This repository has code to calculate many body quantum non-equilibrium dynamics
using a few python libraries `quimb`, `tenpy`, `qutip`, and `quspin`.

- `manybody_util`: contains utilities for many body calculations.

- `time_evolution`: contains templates for book keeping of time evolution
  and the unified JSON-driven run interface.

- `manybody_backends.quimb`: contains the Quimb backend implementation.
  - `spinmodel`: converts shared spin models to Quimb Hamiltonian builders.
  - `quimbtebd`: provides MPS evolution using Quimb TEBD.
  - `quimbmpstrajectory`: provides MPS MCWF trajectory evolution.

- `manybody_backends.qutip`: contains the QuTiP backend implementation.
  - `spinmodel`: converts shared spin models to exact QuTiP Hamiltonians.
  - `timeevolution`: provides exact state evolution using `qutip.sesolve`, `qutip.mesolve`, and `qutip.mcsolve`.

- `manybody_backends.quspin`: contains the QuSpin backend implementation.
  - `spinmodel`: converts shared spin models to exact QuSpin Hamiltonians.
  - `timeevolution`: provides exact state evolution using `hamiltonian.evolve`.

- `manybody_backends.tenpy`: contains the TenPy backend implementation.
  - `spinmodel`: converts shared spin models to direct TenPy chain models.
  - `timeevolution`: provides MPS evolution using `TEBDEngine`, `TwoSiteTDVPEngine`, and `ExpMPOEvolution`.

## Unified JSON evolution

`dynamics/mainEvolve.py` runs the shared JSON schema through one of the supported
backends: TenPy `TEBD`/`TDVP`/`ExpMPO`, Quimb `TEBD`/`MCWF`, QuTiP
`sesolve`/`mesolve`/`mcsolve`, or QuSpin `evolve`.

Example 8-spin transverse-field Ising config (`dynamics/config_tfim.json`):

```json
{
  "model": {
    "kind": "tilted_field_ising_1d",
    "n_sites": 8,
    "j_xx": -1.0,
    "b_z": -0.7,
    "b_x": 0.0,
    "bc": "open"
  },
  "initial_state": {
    "kind": "spin_coherent_product",
    "theta": 1.5707963267948966,
    "phi": 0.0
  },
  "time_grid": {
    "kind": "linspace",
    "start": 0.0,
    "stop": 0.4,
    "num": 81
  },
  "method": {
    "backend": "qutip",
    "algorithm": "sesolve",
    "trunc_params": {},
    "evolution_params": {
      "dt": 0.005,
      "order": 2,
      "N_steps": 2
    }
  },
  "collapse_operators": [],
  "output": {
    "base_dir": "../pkl/tfim_example"
  }
}
```

This represents `H = Jxx sum_i X_i X_{i+1} + Bz sum_i Z_i` with an initial
product state polarized along `+X`. The default method in the file is QuTiP
`sesolve`; the same config can be reused for the other backends with CLI
overrides.

Dissipative methods accept local spin collapse operators:

```json
"collapse_operators": [
  {
    "kind": "local_spin",
    "operator": "sigmam",
    "rate": 0.1,
    "sites": "all"
  }
]
```

Run all four backends from `dynamics/`:

```bash
../.venv/bin/python mainEvolve.py --config config_tfim.json --backend qutip --algorithm sesolve
../.venv/bin/python mainEvolve.py --config config_tfim.json --backend quspin --algorithm evolve
../.venv/bin/python mainEvolve.py --config config_tfim.json --backend quimb --algorithm TEBD --bonddim 8
../.venv/bin/python mainEvolve.py --config config_tfim.json --backend tenpy --algorithm TEBD --bonddim 8
```

The exact QuTiP and QuSpin runs intentionally omit `--bonddim`; the Quimb and
TenPy MPS runs use `--bonddim 8` so the truncation path is exercised for the
8-spin chain.

Plot and persist TFIM observables from the latest matching run for each backend:

```bash
../.venv/bin/python plot_tfim_observables.py --base-dir ../pkl/tfim_example
```

This writes `../pkl/tfim_example/observables/tfim_observables.h5` with
`sum_z = sum_j <sigma^z_j>` and
`sum_zz_offdiag = sum_{j != k} <sigma^z_j sigma^z_k>`, plus an overlay plot at
`../pkl/tfim_example/plots/tfim_observables.png`.

The unified store writes `runs/<run_id>/spec.json`, `manifest.json`,
`states/index.json`, pickle payloads under `states/objects/`, and NumPy array
payloads in `states/arrays.h5` under `../pkl/tfim_example` for this example.
