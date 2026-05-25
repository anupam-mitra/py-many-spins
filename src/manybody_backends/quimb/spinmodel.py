import quimb as qu
import quimb.tensor as qtn
from typing import Any, Tuple

from manybody_util.spinmodel import nearest_neighbor_edges_1d, SpinHalfPauliModel

 
def _pauli_operator(operator: str) -> Any:
    """
    Get the Quimb Pauli operator for a normalized Pauli operator string.

    Parameters
    ----------
    operator : str
        The normalized Pauli operator ('x', 'y', or 'z').

    Returns
    -------
    Any
        The Quimb Pauli operator.

    Raises
    ------
    ValueError
        If the operator is unsupported.
    """
    if operator in ("x", "y", "z"):
        return qu.pauli(operator.upper())
    raise ValueError("unsupported Pauli operator %r" % (operator,))

 
def _edge_signature(edges: tuple[Any, ...]) -> Tuple[Tuple[int, int, float], ...]:
    """
    Get a signature for a set of interaction edges.

    Parameters
    ----------
    edges : tuple[Any, ...]
        The interaction edges.

    Returns
    -------
    Tuple[Tuple[int, int, float], ...]
        The edge signature.
    """
    return tuple((edge.left, edge.right, edge.weight) for edge in edges)

 
def _nearest_neighbor_bc(model: SpinHalfPauliModel) -> str | None:
    """
    Determine the boundary conditions of a model based on its interaction edges.

    Parameters
    ----------
    model : SpinHalfPauliModel
        The spin model.

    Returns
    -------
    str | None
        The boundary conditions ('open' or 'periodic'), or None if no two-site terms exist.

    Raises
    ------
    NotImplementedError
        If the model has multiple edge signatures or unsupported edges.
    """
    if not model.two_site_terms:
        return model.metadata.get("bc", "open")

    signatures = {_edge_signature(term.edges) for term in model.two_site_terms}
    if len(signatures) != 1:
        raise NotImplementedError(
            "Quimb SpinHam1D adapter requires all two-site terms to share edges"
        )

    signature = next(iter(signatures))
    open_edges = _edge_signature(nearest_neighbor_edges_1d(model.n_sites, bc="open"))
    periodic_edges = _edge_signature(
        nearest_neighbor_edges_1d(model.n_sites, bc="periodic")
    )

    if signature == open_edges:
        return "open"
    if signature == periodic_edges:
        return "periodic"
    raise NotImplementedError(
        "Quimb SpinHam1D adapter only supports uniform 1D nearest-neighbor edges"
    )

 
def _uniform_edge_weight(edges: tuple[Any, ...]) -> float:
    """
    Verify that all edges in a set have the same weight and return it.

    Parameters
    ----------
    edges : tuple[Any, ...]
        The interaction edges.

    Returns
    -------
    float
        The uniform edge weight.

    Raises
    ------
    NotImplementedError
        If the edges do not have a uniform weight.
    """
    weights = {edge.weight for edge in edges}
    if len(weights) != 1:
        raise NotImplementedError(
            "Quimb SpinHam1D adapter does not support position-dependent weights"
        )
    return next(iter(weights)) if weights else 1.0

 
def to_quimb_spinham1d(model: SpinHalfPauliModel) -> Any:
    """
    Convert a neutral spin-half Pauli model to ``quimb.tensor.SpinHam1D``.

    Parameters
    ----------
    model : SpinHalfPauliModel
        The shared spin-half Pauli model.

    Returns
    -------
    Any
        The created Quimb SpinHam1D object.

    Raises
    ------
    NotImplementedError
        If the model contains non-uniform local terms.
    """
    bc = _nearest_neighbor_bc(model)
    builder = qtn.SpinHam1D(S=1 / 2, cyclic=(bc == "periodic"))

    for term in model.local_terms:
        if not term.is_uniform:
            raise NotImplementedError(
                "Quimb SpinHam1D adapter only supports uniform local terms"
            )
        if abs(term.coefficient) > 0.0:
            builder.add_term(term.coefficient, _pauli_operator(term.operator))

    for term in model.two_site_terms:
        weight = _uniform_edge_weight(term.edges)
        coefficient = term.coefficient * weight
        if abs(coefficient) > 0.0:
            builder.add_term(
                coefficient,
                _pauli_operator(term.operators[0]),
                _pauli_operator(term.operators[1]),
            )

    return builder

 
def to_quimb_local_hamiltonian(model: SpinHalfPauliModel) -> Any:
    """
    Build the local Hamiltonian object consumed by Quimb TEBD.

    Parameters
    ----------
    model : SpinHalfPauliModel
        The shared spin-half Pauli model.

    Returns
    -------
    Any
        The built local Hamiltonian.
    """
    return to_quimb_spinham1d(model).build_local_ham(model.n_sites)

