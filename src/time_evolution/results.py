from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvolutionResult:
    records: list[dict[str, Any]]
    states: list[Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MarginalResult:
    records: list[dict[str, Any]]
    marginals: list[Any]
    metadata: dict[str, Any] = field(default_factory=dict)
