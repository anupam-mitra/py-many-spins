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

Example 10-spin transverse-field Ising config (`dynamics/config_tfim.json`):

```json
{
  "model": {
    "kind": "tilted_field_ising_1d",
    "n_sites": 10,
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
    "stop": 16.0,
    "num": 321
  },
  "method": {
    "backend": "qutip",
    "algorithm": "sesolve",
    "trunc_params": {},
    "evolution_params": {
      "dt": 0.005,
      "order": 2,
      "N_steps": 10
    }
  },
  "collapse_operators": [],
  "output": {
    "base_dir": "../pkl/tfim_10spin"
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
../.venv/bin/python mainEvolve.py --config config_tfim.json --backend quimb --algorithm TEBD --bonddim 32
../.venv/bin/python mainEvolve.py --config config_tfim.json --backend tenpy --algorithm TEBD --bonddim 32
```

The exact QuTiP and QuSpin runs intentionally omit `--bonddim`; the Quimb and
TenPy MPS runs use `--bonddim 32`, matching the maximum exact middle-cut rank
for the 10-spin chain.

Plot and persist TFIM observables from the latest matching run for each backend:

```bash
../.venv/bin/python plot_tfim_observables.py --base-dir ../pkl/tfim_10spin
```

This writes `../pkl/tfim_10spin/observables/tfim_observables.h5` with
normalized ordered distinct-site sums `z1`, `z2_offdiag`, `z3_offdiag`, and
`z4_offdiag`, plus an overlay plot at
`../pkl/tfim_10spin/plots/tfim_observables.png`. The datasets divide the raw
order-`p` sums by `N^p`, so each plotted y-axis is between `-1` and `1`.

The unified store writes `runs/<run_id>/spec.json`, `manifest.json`,
`states/index.json`, pickle payloads under `states/objects/`, and NumPy array
payloads in `states/arrays.h5` under `../pkl/tfim_10spin` for this example.
