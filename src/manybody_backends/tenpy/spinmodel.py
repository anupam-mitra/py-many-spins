from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.networks.site import SpinHalfSite


_TENPY_PAULI_OPERATORS = {
    "x": "Sigmax",
    "y": "Sigmay",
    "z": "Sigmaz",
}


def _operator_name(operator):
    try:
        return _TENPY_PAULI_OPERATORS[operator]
    except KeyError as exc:
        raise ValueError("unsupported Pauli operator %r" % (operator,)) from exc


def _chain_index(site):
    return [site, 0]


def _require_nearest_neighbor(left, right):
    if abs(left - right) != 1:
        raise NotImplementedError(
            "TenPy TEBD adapter only supports open 1D nearest-neighbor couplings"
        )


class SpinHalfPauliChain(CouplingMPOModel, NearestNeighborModel):
    """TenPy chain model for shared spin-half Pauli Hamiltonians."""

    default_lattice = Chain
    force_default_lattice = True

    def init_sites(self, model_params):
        conserve = model_params.get("conserve", None)
        sort_charge = model_params.get("sort_charge", True)
        return SpinHalfSite(conserve=conserve, sort_charge=sort_charge)

    def init_terms(self, model_params):
        spin_model = model_params.get("spin_model", None)
        if spin_model is None:
            raise ValueError("model_params must include a 'spin_model'")

        for coefficient, operator, site in spin_model.expanded_local_terms():
            if abs(coefficient) > 0.0:
                self.add_local_term(
                    coefficient,
                    [(_operator_name(operator), _chain_index(site))],
                )

        for coefficient, operators, left, right in spin_model.expanded_two_site_terms():
            if abs(coefficient) > 0.0:
                _require_nearest_neighbor(left, right)
                self.add_local_term(
                    coefficient,
                    [
                        (_operator_name(operators[0]), _chain_index(left)),
                        (_operator_name(operators[1]), _chain_index(right)),
                    ],
                )


def to_tenpy_model(model, bc_mps="finite", conserve=None, sort_charge=True):
    """Convert a shared spin-half Pauli chain to a TenPy model."""
    return SpinHalfPauliChain({
        "L": model.n_sites,
        "spin_model": model,
        "bc_MPS": bc_mps,
        "conserve": conserve,
        "sort_charge": sort_charge,
    })
