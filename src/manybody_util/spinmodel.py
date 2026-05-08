from dataclasses import dataclass, field
from typing import Mapping


PAULI_OPERATORS = frozenset(("x", "y", "z"))


def _normalize_pauli(operator: str) -> str:
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
    return abs(value) > 0.0


@dataclass(frozen=True)
class WeightedEdge:
    """A weighted two-site interaction edge."""

    left: int
    right: int
    weight: float = 1.0

    def __post_init__(self):
        if self.left < 0 or self.right < 0:
            raise ValueError("edge sites must be non-negative")
        if self.left == self.right:
            raise ValueError("self-edges are not valid two-site interactions")


@dataclass(frozen=True)
class LocalTerm:
    """A local Pauli term, uniform when ``sites`` is ``None``."""

    coefficient: complex
    operator: str
    sites: tuple[int, ...] | None = None

    def __post_init__(self):
        object.__setattr__(self, "operator", _normalize_pauli(self.operator))
        if self.sites is not None:
            object.__setattr__(self, "sites", tuple(self.sites))
            if any(site < 0 for site in self.sites):
                raise ValueError("local term sites must be non-negative")

    @property
    def is_uniform(self) -> bool:
        return self.sites is None


@dataclass(frozen=True)
class TwoSiteTerm:
    """A two-site Pauli interaction over explicit weighted edges."""

    coefficient: complex
    operators: tuple[str, str]
    edges: tuple[WeightedEdge, ...]

    def __post_init__(self):
        if len(self.operators) != 2:
            raise ValueError("two-site terms require exactly two operators")
        operators = tuple(_normalize_pauli(operator) for operator in self.operators)
        object.__setattr__(self, "operators", operators)
        object.__setattr__(self, "edges", tuple(self.edges))


@dataclass(frozen=True)
class SpinHalfPauliModel:
    """Backend-independent spin-half Hamiltonian using Pauli X/Y/Z labels."""

    n_sites: int
    local_terms: tuple[LocalTerm, ...] = ()
    two_site_terms: tuple[TwoSiteTerm, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
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

    def expanded_local_terms(self):
        """Yield ``(coefficient, operator, site)`` for every local term."""
        for term in self.local_terms:
            sites = range(self.n_sites) if term.sites is None else term.sites
            for site in sites:
                yield term.coefficient, term.operator, site

    def expanded_two_site_terms(self):
        """Yield ``(coefficient, operators, left, right)`` for each edge."""
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
    """Return nearest-neighbor edges for a one-dimensional chain."""
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
    """Build ``H = Jxx sum XX + Bz sum Z + Bx sum X`` on a 1D chain."""
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
