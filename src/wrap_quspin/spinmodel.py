import numpy as np
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian


def to_quspin_basis(model):
    """Return a full spin-half basis using Pauli operator conventions."""
    return spin_basis_1d(L=model.n_sites, S="1/2", pauli=1)


def to_quspin_static_terms(model):
    """Convert a neutral spin-half Pauli model to QuSpin static terms."""
    terms_by_operator = {}

    for coefficient, operator, site in model.expanded_local_terms():
        terms_by_operator.setdefault(operator, []).append([coefficient, site])

    for coefficient, operators, left, right in model.expanded_two_site_terms():
        opstr = "%s%s" % operators
        terms_by_operator.setdefault(opstr, []).append([coefficient, left, right])

    return [[opstr, couplings] for opstr, couplings in terms_by_operator.items()]


def to_quspin_hamiltonian(model, basis=None, dtype=np.complex128):
    """Convert a neutral spin-half Pauli model to a QuSpin Hamiltonian."""
    if basis is None:
        basis = to_quspin_basis(model)

    return hamiltonian(
        to_quspin_static_terms(model),
        [],
        basis=basis,
        dtype=dtype,
        check_symm=False,
        check_herm=False,
        check_pcon=False,
    )
