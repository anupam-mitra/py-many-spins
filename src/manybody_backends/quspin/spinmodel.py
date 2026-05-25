import numpy as np
from typing import Any, Tuple, List, Union
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian

from manybody_util.spinmodel import SpinHalfPauliModel

 
def to_quspin_basis(model: SpinHalfPauliModel) -> Any:
    """
    Return a full spin-half basis using Pauli operator conventions.

    Parameters
    ----------
    model : SpinHalfPauliModel
        The shared spin-half Pauli model.

    Returns
    -------
    Any
        The QuSpin spin_basis_1d object.
    """
    return spin_basis_1d(L=model.n_sites, S="1/2", pauli=1)

 
def to_quspin_static_terms(model: SpinHalfPauliModel) -> List[List[Union[str, List[Any]]]]:
    """
    Convert a neutral spin-half Pauli model to QuSpin static terms.

    Parameters
    ----------
    model : SpinHalfPauliModel
        The shared spin-half Pauli model.

    Returns
    -------
    List[List[Union[str, List[Any]]]]
        A list of QuSpin static terms.
    """
    terms_by_operator = {}

    for coefficient, operator, site in model.expanded_local_terms():
        terms_by_operator.setdefault(operator, []).append([coefficient, site])

    for coefficient, operators, left, right in model.expanded_two_site_terms():
        opstr = "%s%s" % operators
        terms_by_operator.setdefault(opstr, []).append([coefficient, left, right])

    return [[opstr, couplings] for opstr, couplings in terms_by_operator.items()]

 
def to_quspin_hamiltonian(model: SpinHalfPauliModel, basis: Any = None, dtype: Any = np.complex128) -> Any:
    """
    Convert a neutral spin-half Pauli model to a QuSpin Hamiltonian.

    Parameters
    ----------
    model : SpinHalfPauliModel
        The shared spin-half Pauli model.
    basis : Any, optional
        The QuSpin basis. If None, a default basis is created, by default None.
    dtype : Any, optional
        The data type for the Hamiltonian, by default np.complex128.

    Returns
    -------
    Any
        The QuSpin Hamiltonian object.
    """
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

