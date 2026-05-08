# Simulation of many body quantum dynamics
This repository has code to calculate many body quantum non-equilibrium dynamics
using a few python libraries `quimb`, `tenpy`, `qutip`, and `quspin`.

- `manybody_util`: contains utilities for many body calculations.

- `time_evolution`: contains templates for book keeping of time evolution

- `wrap_quimb`: contains wrappers around `Quimb` classes and functions.
  - `time_evolution`: provides time evolution using `Quimb`.
  - `distance_measures`:

- `wrap_qutip`: contains wrappers around `QuTiP` classes and functions.
  - `spinmodel`: converts shared spin models to exact QuTiP Hamiltonians.
  - `timeevolution`: provides exact state evolution using `qutip.sesolve`.

- `wrap_quspin`: contains wrappers around `QuSpin` classes and functions.
  - `spinmodel`: converts shared spin models to exact QuSpin Hamiltonians.
  - `timeevolution`: provides exact state evolution using `hamiltonian.evolve`.

- `wrap_tenpy`: contains wrappers around `TenPy` classes and functions.
  - `spinmodel`: converts shared spin models to direct TenPy chain models.
  - `timeevolution`: provides MPS evolution using `TEBDEngine` and `TwoSiteTDVPEngine`.
