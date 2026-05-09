import quimb as qu
import quimb.tensor as qtn

from manybody_util.spinmodel import nearest_neighbor_edges_1d


def _pauli_operator(operator):
    if operator in ("x", "y", "z"):
        return qu.pauli(operator.upper())
    raise ValueError("unsupported Pauli operator %r" % (operator,))


def _edge_signature(edges):
    return tuple((edge.left, edge.right, edge.weight) for edge in edges)


def _nearest_neighbor_bc(model):
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


def _uniform_edge_weight(edges):
    weights = {edge.weight for edge in edges}
    if len(weights) != 1:
        raise NotImplementedError(
            "Quimb SpinHam1D adapter does not support position-dependent weights"
        )
    return next(iter(weights)) if weights else 1.0


def to_quimb_spinham1d(model):
    """Convert a neutral spin-half Pauli model to ``quimb.tensor.SpinHam1D``."""
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


def to_quimb_local_hamiltonian(model):
    """Build the local Hamiltonian object consumed by Quimb TEBD."""
    return to_quimb_spinham1d(model).build_local_ham(model.n_sites)
