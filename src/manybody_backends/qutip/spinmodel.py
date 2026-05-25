import qutip
from typing import Any, List, Tuple

from manybody_util.spinmodel import SpinHalfPauliModel

 
def _pauli_operator(operator: str) -> Any:
    """
    Get the QuTiP Pauli operator for a normalized Pauli operator string.

    Parameters
    ----------
    operator : str
        The normalized Pauli operator ('x', 'y', or 'z').

    Returns
    -------
    Any
        The QuTiP Pauli operator.

    Raises
    ------
    ValueError
        If the operator is unsupported.
    """
    if operator == "x":
        return qutip.sigmax()
    if operator == "y":
        return qutip.sigmay()
    if operator == "z":
        return qutip.sigmaz()
    raise ValueError("unsupported Pauli operator %r" % (operator,))

 
def _embed_spinhalf_operator(n_sites: int, site_operators: List[Tuple[int, Any]]) -> Any:
    """
    Embed single-site operators into the full Hilbert space of a spin-half chain.

    Parameters
    ----------
    n_sites : int
        The number of sites.
    site_operators : List[Tuple[int, Any]]
        A list of (site, operator) tuples.

    Returns
    -------
    Any
        The embedded operator as a QuTiP Qobj.

    Raises
    ------
    ValueError
        If multiple operators are on the same site or a site is out of range.
    """
    operators_by_site = dict(site_operators)
    if len(operators_by_site) != len(site_operators):
        raise ValueError("multiple operators on the same site are not supported")
    if any(site < 0 or site >= n_sites for site in operators_by_site):
        raise ValueError("operator site exceeds n_sites")

    factors = []
    for site in range(n_sites):
        factors.append(operators_by_site.get(site, qutip.qeye(2)))

    return qutip.tensor(factors)

 
def _zero_hamiltonian(n_sites: int) -> Any:
    """
    Create a zero Hamiltonian for a spin-half chain.

    Parameters
    ----------
    n_sites : int
        The number of sites.

    Returns
    -------
    Any
        The zero Hamiltonian as a QuTiP Qobj.
    """
    return 0 * qutip.tensor([qutip.qeye(2) for _ in range(n_sites)])

 
def to_qutip_hamiltonian(model: SpinHalfPauliModel) -> Any:
    """
    Convert a neutral spin-half Pauli model to a QuTiP Hamiltonian.

    Parameters
    ----------
    model : SpinHalfPauliModel
        The shared spin-half Pauli model.

    Returns
    -------
    Any
        The QuTiP Hamiltonian as a Qobj.
    """
    terms = []

    for coefficient, operator, site in model.expanded_local_terms():
        terms.append(
            coefficient
            * _embed_spinhalf_operator(
                model.n_sites,
                [(site, _pauli_operator(operator))],
            )
        )

    for coefficient, operators, left, right in model.expanded_two_site_terms():
        terms.append(
            coefficient
            * _embed_spinhalf_operator(
                model.n_sites,
                [
                    (left, _pauli_operator(operators[0])),
                    (right, _pauli_operator(operators[1])),
                ],
            )
        )

    if not terms:
        return _zero_hamiltonian(model.n_sites)

    return sum(terms[1:], terms[0])

