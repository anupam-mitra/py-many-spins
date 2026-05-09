# Simulation of many body quantum dynamics
This repository has code to calculate many body quantum non-equilibrium dynamics
using a few python libraries `quimb`, `tenpy`, `qutip`, and `quspin`.

- `manybody_util`: contains utilities for many body calculations.

- `time_evolution`: contains templates for book keeping of time evolution
  and the unified JSON-driven run interface.

- `manybody_backends.quimb`: contains the Quimb backend implementation.
  - `time_evolution`: provides time evolution using `Quimb`.
  - `distance_measures`:

- `manybody_backends.qutip`: contains the QuTiP backend implementation.
  - `spinmodel`: converts shared spin models to exact QuTiP Hamiltonians.
  - `timeevolution`: provides exact state evolution using `qutip.sesolve`.

- `manybody_backends.quspin`: contains the QuSpin backend implementation.
  - `spinmodel`: converts shared spin models to exact QuSpin Hamiltonians.
  - `timeevolution`: provides exact state evolution using `hamiltonian.evolve`.

- `manybody_backends.tenpy`: contains the TenPy backend implementation.
  - `spinmodel`: converts shared spin models to direct TenPy chain models.
  - `timeevolution`: provides MPS evolution using `TEBDEngine`, `TwoSiteTDVPEngine`, and `ExpMPOEvolution`.

## Unified JSON evolution

`dynamics/mainEvolve.py` runs the shared JSON schema through one of the supported
backends: TenPy `TEBD`/`TDVP`/`ExpMPO`, Quimb `TEBD`, QuTiP `sesolve`, or QuSpin
`evolve`.

Example config:

```json
{
  "model": {
    "kind": "tilted_field_ising_1d",
    "n_sites": 4,
    "j_xx": 0.1,
    "b_z": 1.0,
    "b_x": 0.15,
    "bc": "open"
  },
  "initial_state": {
    "kind": "spin_coherent_product",
    "theta": 1.5707963267948966,
    "phi": 1.5707963267948966
  },
  "time_grid": {
    "kind": "linspace",
    "start": 0.0,
    "stop": 1.0,
    "num": 3
  },
  "method": {
    "backend": "qutip",
    "algorithm": "sesolve",
    "trunc_params": {},
    "evolution_params": {}
  },
  "output": {
    "base_dir": "../pkl"
  }
}
```

Run from `dynamics/`:

```bash
../.venv/bin/python mainEvolve.py --config config.json
```

The unified store writes `runs/<run_id>/spec.json`, `manifest.json`,
`states/index.json`, pickle payloads under `states/objects/`, and NumPy array
payloads in `states/arrays.h5`.
