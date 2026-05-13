from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvolutionResult:
    """
    The result of a time evolution simulation.

    Attributes
    ----------
    records : list[dict[str, Any]]
        The record of each time step.
    states : list[Any]
        The states at each time step.
    metadata : dict[str, Any]
        Additional metadata for the evolution run.
    """
    records: list[dict[str, Any]]
    states: list[Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MarginalResult:
    """
    The result of a marginal simulation run.

    Attributes
    ----------
    records : list[dict[str, Any]]
        The record of each time step.
    marginals : list[Any]
        The marginals at each time step.
    metadata : dict[str, Any]
        Additional metadata for the marginal run.
    """
    records: list[dict[str, Any]]
    marginals: list[Any]
    metadata: dict[str, Any] = field(default_factory=dict)

