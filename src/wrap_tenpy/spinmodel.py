import tenpy.models.tf_ising

from manybody_util.spinmodel import nearest_neighbor_edges_1d


_TENPY_PAULI_OPERATORS = {
    "x": "Sigmax",
    "y": "Sigmay",
    "z": "Sigmaz",
}


def _edge_signature(edges):
    return tuple((edge.left, edge.right) for edge in edges)


def _uniform_edge_weight(edges):
    weights = {edge.weight for edge in edges}
    if len(weights) != 1:
        raise NotImplementedError(
            "TenPy TFIChain adapter does not support position-dependent weights"
        )
    return next(iter(weights)) if weights else 1.0


def _require_open_nearest_neighbors(model, term):
    expected = _edge_signature(nearest_neighbor_edges_1d(model.n_sites, bc="open"))
    actual = _edge_signature(term.edges)
    if actual != expected:
        raise NotImplementedError(
            "TenPy TFIChain adapter only supports open 1D nearest-neighbor edges"
        )


def _extract_tfi_parameters(model):
    j_xx = 0.0
    b_z = 0.0
    b_x = 0.0

    for term in model.local_terms:
        if not term.is_uniform:
            raise NotImplementedError(
                "TenPy TFIChain adapter only supports uniform local terms"
            )
        if term.operator == "z":
            b_z += term.coefficient
        elif term.operator == "x":
            b_x += term.coefficient
        elif abs(term.coefficient) > 0.0:
            raise NotImplementedError(
                "TenPy TFIChain adapter only supports X and Z local terms"
            )

    for term in model.two_site_terms:
        if term.operators != ("x", "x"):
            raise NotImplementedError(
                "TenPy TFIChain adapter only supports XX two-site interactions"
            )
        _require_open_nearest_neighbors(model, term)
        j_xx += term.coefficient * _uniform_edge_weight(term.edges)

    return j_xx, b_z, b_x


def to_tenpy_model(model, bc_mps="finite", conserve=None):
    """Convert a neutral tilted-field Ising model to TenPy's ``TFIChain``."""
    j_xx, b_z, b_x = _extract_tfi_parameters(model)

    tenpy_model = tenpy.models.tf_ising.TFIChain({
        "L": model.n_sites,
        "J": j_xx,
        "g": b_z,
        "bc_MPS": bc_mps,
        "conserve": conserve,
    })

    if abs(b_x) > 0.0:
        tenpy_model.manually_call_init_H = True
        tenpy_model.add_onsite(b_x, 0, _TENPY_PAULI_OPERATORS["x"])
        tenpy_model.init_H_from_terms()

    return tenpy_model
