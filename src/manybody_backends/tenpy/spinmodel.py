from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.networks.site import SpinHalfSite
from typing import Any, Tuple

 
_TENPY_PAULI_OPERATORS = {
    "x": "Sigmax",
    "y": "Sigmay",
    "z": "Sigmaz",
}

 
def _operator_name(operator: str) -> str:
    """
    Get the TenPy operator name for a normalized Pauli operator.

    Parameters
    ----------
    operator : str
        The normalized Pauli operator ('x', 'y', or 'z').

    Returns
    -------
    str
        The corresponding TenPy operator name.

    Raises
    ------
    ValueError
        If the operator is not supported.
    """
    try:
        return _TENPY_PAULI_OPERATORS[operator]
    except KeyError as exc:
        raise ValueError("unsupported Pauli operator %r" % (operator,)) from exc

 
def _chain_index(site: int) -> Tuple[int, int]:
    """
    Get the TenPy chain index for a site.

    Parameters
    ----------
    site : int
        The site index.

    Returns
    -------
    Tuple[int, int]
        The chain index as (site, 0).
    """
    return [site, 0]

 
def _require_nearest_neighbor(left: int, right: int) -> None:
    """
    Verify that two sites are nearest neighbors.

    Parameters
    ----------
    left : int
        The left site index.
    right : int
        The right site index.

    Raises
    ------
    NotImplementedError
        If the sites are not nearest neighbors.
    """
    if abs(left - right) != 1:
        raise NotImplementedError(
            "TenPy TEBD adapter only supports open 1D nearest-neighbor couplings"
        )

 
class SpinHalfPauliChain(CouplingMPOModel, NearestNeighborModel):
    """
    TenPy chain model for shared spin-half Pauli Hamiltonians.

    Attributes
    ----------
    default_lattice : Type[Chain]
        The default lattice type for the model.
    force_default_lattice : bool
        Whether to force the use of the default lattice.
    """

    default_lattice = Chain
    force_default_lattice = True

    def init_sites(self, model_params: dict[str, Any]) -> SpinHalfSite:
        """
        Initialize the sites for the model.

        Parameters
        ----------
        model_params : dict[str, Any]
            The model parameters.

        Returns
        -------
        SpinHalfSite
            The initialized site.
        """
        conserve = model_params.get("conserve", None)
        sort_charge = model_params.get("sort_charge", True)
        return SpinHalfSite(conserve=conserve, sort_charge=sort_charge)

    def init_terms(self, model_params: dict[str, Any]) -> None:
        """
        Initialize the Hamiltonian terms from a shared spin model.

        Parameters
        ----------
        model_params : dict[str, Any]
            The model parameters, must include 'spin_model'.

        Raises
        ------
        ValueError
            If 'spin_model' is missing from model_params.
        """
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

 
def to_tenpy_model(
    model: Any, bc_mps: str = "finite", conserve: Any = None, sort_charge: bool = True
) -> SpinHalfPauliChain:
    """
    Convert a shared spin-half Pauli chain to a TenPy model.

    Parameters
    ----------
    model : Any
        The shared spin-half Pauli model.
    bc_mps : str, optional
        Boundary conditions for the MPS, by default "finite".
    conserve : Any, optional
        Conservation law, by default None.
    sort_charge : bool, optional
        Whether to sort charges, by default True.

    Returns
    -------
    SpinHalfPauliChain
        The created TenPy model.
    """
    return SpinHalfPauliChain({
        "L": model.n_sites,
        "spin_model": model,
        "bc_MPS": bc_mps,
        "conserve": conserve,
        "sort_charge": sort_charge,
    })

