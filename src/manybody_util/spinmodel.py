from dataclasses import dataclass, field
from typing import Mapping, Iterator, Tuple

PAULI_OPERATORS = frozenset(("x", "y", "z"))


def _normalize_pauli(operator: str) -> str:
    """
    Normalize a Pauli operator string to a standard single-character representation.

    Parameters
    ----------
    operator : str
        The Pauli operator string (e.g., 'SigmaX', 'Pauliz').

    Returns
    -------
    str
        The normalized Pauli operator ('x', 'y', or 'z').

    Raises
    ------
    ValueError
        If the operator is not a recognized Pauli operator.
    """
    op = operator.strip().lower()
    aliases = {
        "sigmax": "x",
        "sigmay": "y",
        "sigmaz": "z",
        "paulix": "x",
        "pauliy": "y",
        "pauliz": "z",
    }
    op = aliases.get(op, op)
    if op not in PAULI_OPERATORS:
        raise ValueError("unsupported Pauli operator %r" % (operator,))
    return op


def _is_nonzero(value: complex) -> bool:
    """
    Check if a complex value is numerically non-zero.

    Parameters
    ----------
    value : complex
        The value to check.

    Returns
    -------
    bool
        True if the absolute value is greater than 0.0, False otherwise.
    """
    return abs(value) > 0.0


@dataclass(frozen=True)
class WeightedEdge:
    """
    A weighted two-site interaction edge.

    Attributes
    ----------
    left : int
        Index of the left site.
    right : int
        Index of the right site.
    weight : float
        The weight of the interaction.
    """

    left: int
    right: int
    weight: float = 1.0

    def __post_init__(self) -> None:
        """
        Validate the edge sites.

        Raises
        ------
        ValueError
            If sites are negative or if it is a self-edge.
        """
        if self.left < 0 or self.right < 0:
            raise ValueError("edge sites must be non-negative")
        if self.left == self.right:
            raise ValueError("self-edges are not valid two-site interactions")


@dataclass(frozen=True)
class LocalTerm:
    """
    A local Pauli term, uniform when ``sites`` is ``None``.

    Attributes
    ----------
    coefficient : complex
        The coefficient of the term.
    operator : str
        The normalized Pauli operator.
    sites : tuple[int, ...] | None
        The sites to which the term is applied. If None, it is applied to all sites.
    """

    coefficient: complex
    operator: str
    sites: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """
        Normalize operator and validate sites.

        Raises
        ------
        ValueError
            If any site index is negative.
        """
        object.__setattr__(self, "operator", _normalize_pauli(self.operator))
        if self.sites is not None:
            object.__setattr__(self, "sites", tuple(self.sites))
            if any(site < 0 for site in self.sites):
                raise ValueError("local term sites must be non-negative")

    @property
    def is_uniform(self) -> bool:
        """
        Check if the term is applied uniformly across all sites.

        Returns
        -------
        bool
            True if sites is None, False otherwise.
        """
        return self.sites is None


@dataclass(frozen=True)
class TwoSiteTerm:
    """
    A two-site Pauli interaction over explicit weighted edges.

    Attributes
    ----------
    coefficient : complex
        The coefficient of the interaction.
    operators : tuple[str, str]
        The pair of Pauli operators.
    edges : tuple[WeightedEdge, ...]
        The weighted edges over which the interaction is applied.
    """

    coefficient: complex
    operators: tuple[str, str]
    edges: tuple[WeightedEdge, ...]

    def __post_init__(self) -> None:
        """
        Validate and normalize the two-site term.

        Raises
        ------
        ValueError
            If the number of operators is not exactly two.
        """
        if len(self.operators) != 2:
            raise ValueError("two-site terms require exactly two operators")
        operators = tuple(_normalize_pauli(operator) for operator in self.operators)
        object.__setattr__(self, "operators", operators)
        object.__setattr__(self, "edges", tuple(self.edges))


@dataclass(frozen=True)
class SpinHalfPauliModel:
    """
    Backend-independent spin-half Hamiltonian using Pauli X/Y/Z labels.

    Attributes
    ----------
    n_sites : int
        The number of sites in the model.
    local_terms : tuple[LocalTerm, ...]
        A tuple of local Pauli terms.
    two_site_terms : tuple[TwoSiteTerm, ...]
        A tuple of two-site Pauli terms.
    metadata : Mapping[str, object]
        Additional metadata for the model.
    """

    n_sites: int
    local_terms: tuple[LocalTerm, ...] = ()
    two_site_terms: tuple[TwoSiteTerm, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Validate model consistency.

        Raises
        ------
        ValueError
            If n_sites is non-positive or if any term refers to a site index
            outside the range [0, n_sites - 1].
        """
        if self.n_sites <= 0:
            raise ValueError("n_sites must be positive")

        local_terms = tuple(self.local_terms)
        two_site_terms = tuple(self.two_site_terms)

        for term in local_terms:
            sites = range(self.n_sites) if term.sites is None else term.sites
            for site in sites:
                if site >= self.n_sites:
                    raise ValueError("local term site %d exceeds n_sites" % (site,))

        for term in two_site_terms:
            for edge in term.edges:
                if edge.left >= self.n_sites or edge.right >= self.n_sites:
                    raise ValueError(
                        "edge (%d, %d) exceeds n_sites" % (edge.left, edge.right)
                    )

        object.__setattr__(self, "local_terms", local_terms)
        object.__setattr__(self, "two_site_terms", two_site_terms)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def expanded_local_terms(self) -> Iterator[Tuple[complex, str, int]]:
        """
        Yield coefficients and operators for every local term.

        Yields
        ------
        Tuple[complex, str, int]
            A tuple of (coefficient, operator, site).
        """
        for term in self.local_terms:
            sites = range(self.n_sites) if term.sites is None else term.sites
            for site in sites:
                yield term.coefficient, term.operator, site

    def expanded_two_site_terms(self) -> Iterator[Tuple[complex, Tuple[str, str], int, int]]:
        """
        Yield expanded coefficients and operators for each edge.

        Yields
        ------
        Tuple[complex, Tuple[str, str], int, int]
            A tuple of (coefficient * weight, operators, left_site, right_site).
        """
        for term in self.two_site_terms:
            for edge in term.edges:
                yield (
                    term.coefficient * edge.weight,
                    term.operators,
                    edge.left,
                    edge.right,
                )


def nearest_neighbor_edges_1d(
    n_sites: int,
    bc: str = "open",
    weight: float = 1.0,
) -> tuple[WeightedEdge, ...]:
    """
    Return nearest-neighbor edges for a one-dimensional chain.

    Parameters
    ----------
    n_sites : int
        The number of sites in the chain.
    bc : str, optional
        Boundary conditions ('open' or 'periodic'), by default "open".
    weight : float, optional
        The weight of the edges, by default 1.0.

    Returns
    -------
    tuple[WeightedEdge, ...]
        A tuple of WeightedEdge objects.

    Raises
    ------
    ValueError
        If n_sites is non-positive or bc is unsupported.
    """
    if n_sites <= 0:
        raise ValueError("n_sites must be positive")
    if bc not in ("open", "periodic"):
        raise ValueError("bc must be 'open' or 'periodic'")

    edges = [WeightedEdge(site, site + 1, weight) for site in range(n_sites - 1)]
    if bc == "periodic" and n_sites > 2:
        edges.append(WeightedEdge(n_sites - 1, 0, weight))
    return tuple(edges)


def tilted_field_ising_1d(
    n_sites: int,
    j_xx: float,
    b_z: float,
    b_x: float = 0.0,
    bc: str = "open",
) -> SpinHalfPauliModel:
    """
    Build a 1D tilted-field Ising model Hamiltonian.

    H = Jxx sum_{<i,j>} XX + Bz sum_i Z + Bx sum_i X

    Parameters
    ----------
    n_sites : int
        The number of sites.
    j_xx : float
        Interaction strength J_xx.
    b_z : float
        Transverse field B_z.
    b_x : float, optional
        Transverse field B_x, by default 0.0.
    bc : str, optional
        Boundary conditions ('open' or 'periodic'), by default "open".

    Returns
    -------
    SpinHalfPauliModel
        The built spin-half Pauli model.
    """
    local_terms = []
    if _is_nonzero(b_z):
        local_terms.append(LocalTerm(b_z, "z"))
    if _is_nonzero(b_x):
        local_terms.append(LocalTerm(b_x, "x"))

    two_site_terms = []
    if _is_nonzero(j_xx):
        two_site_terms.append(
            TwoSiteTerm(
                j_xx,
                ("x", "x"),
                nearest_neighbor_edges_1d(n_sites, bc=bc),
            )
        )

    return SpinHalfPauliModel(
        n_sites=n_sites,
        local_terms=tuple(local_terms),
        two_site_terms=tuple(two_site_terms),
        metadata={
            "family": "tilted_field_ising_1d",
            "bc": bc,
            "j_xx": j_xx,
            "b_z": b_z,
            "b_x": b_x,
        },
    )
