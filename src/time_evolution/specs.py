from dataclasses import asdict, dataclass, field
from typing import Any


SUPPORTED_METHODS = {
    "tenpy": {"TEBD", "TDVP", "ExpMPO"},
    "quimb": {"TEBD", "MCWF"},
    "qutip": {"sesolve", "mesolve", "mcsolve"},
    "quspin": {"evolve"},
}

EXACT_BACKENDS = frozenset(("qutip", "quspin"))
DISSIPATIVE_METHODS = frozenset((
    ("quimb", "MCWF"),
    ("qutip", "mesolve"),
    ("qutip", "mcsolve"),
))
COLLAPSE_OPERATORS = frozenset(("sigmam", "sigmap", "x", "y", "z"))


def _clean_dict(value):
    return dict(value or {})


def _has_nonempty_truncation(trunc_params):
    return any(value is not None for value in _clean_dict(trunc_params).values())


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    n_sites: int
    j_xx: float
    b_z: float
    b_x: float = 0.0
    bc: str = "open"

    @classmethod
    def tilted_field_ising_1d(
        cls,
        n_sites,
        j_xx,
        b_z,
        b_x=0.0,
        bc="open",
    ):
        return cls(
            kind="tilted_field_ising_1d",
            n_sites=n_sites,
            j_xx=j_xx,
            b_z=b_z,
            b_x=b_x,
            bc=bc,
        )

    @classmethod
    def from_dict(cls, data):
        return cls(**dict(data))

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if self.kind != "tilted_field_ising_1d":
            raise ValueError("unsupported model kind %r" % (self.kind,))
        if self.n_sites <= 0:
            raise ValueError("n_sites must be positive")
        if self.bc not in ("open", "periodic"):
            raise ValueError("bc must be 'open' or 'periodic'")


@dataclass(frozen=True)
class InitialStateSpec:
    kind: str = "spin_coherent_product"
    theta: float = 0.0
    phi: float = 0.0

    @classmethod
    def spin_coherent_product(cls, theta, phi):
        return cls(kind="spin_coherent_product", theta=theta, phi=phi)

    @classmethod
    def from_dict(cls, data):
        return cls(**dict(data))

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if self.kind != "spin_coherent_product":
            raise ValueError("unsupported initial state kind %r" % (self.kind,))


@dataclass(frozen=True)
class TimeGridSpec:
    kind: str = "linspace"
    start: float = 0.0
    stop: float = 1.0
    num: int = 2

    @classmethod
    def linspace(cls, start, stop, num):
        return cls(kind="linspace", start=start, stop=stop, num=num)

    @classmethod
    def from_dict(cls, data):
        return cls(**dict(data))

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if self.kind != "linspace":
            raise ValueError("unsupported time grid kind %r" % (self.kind,))
        if self.num <= 0:
            raise ValueError("time grid num must be positive")


@dataclass(frozen=True)
class CollapseOperatorSpec:
    kind: str = "local_spin"
    operator: str = "sigmam"
    rate: float = 1.0
    sites: Any = "all"

    def __post_init__(self):
        object.__setattr__(self, "operator", self.operator.lower())
        if isinstance(self.sites, int):
            object.__setattr__(self, "sites", (self.sites,))
        elif self.sites != "all" and not isinstance(self.sites, tuple):
            object.__setattr__(self, "sites", tuple(self.sites))

    @classmethod
    def local_spin(cls, operator, rate, sites="all"):
        return cls(kind="local_spin", operator=operator, rate=rate, sites=sites)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data.setdefault("kind", "local_spin")
        data.setdefault("sites", "all")
        if data["sites"] != "all" and not isinstance(data["sites"], int):
            data["sites"] = tuple(data["sites"])
        return cls(**data)

    def to_dict(self):
        return {
            "kind": self.kind,
            "operator": self.operator,
            "rate": self.rate,
            "sites": list(self.sites) if self.sites != "all" else "all",
        }

    def validate(self, n_sites=None):
        if self.kind != "local_spin":
            raise ValueError("unsupported collapse operator kind %r" % (self.kind,))
        if self.operator not in COLLAPSE_OPERATORS:
            raise ValueError("unsupported collapse operator %r" % (self.operator,))
        if self.rate < 0.0:
            raise ValueError("collapse operator rate must be non-negative")
        if self.sites == "all":
            return
        if not isinstance(self.sites, (list, tuple)):
            raise ValueError("collapse operator sites must be 'all' or a sequence")
        if len(self.sites) == 0:
            raise ValueError("collapse operator sites must be non-empty")
        for site in self.sites:
            if not isinstance(site, int):
                raise ValueError("collapse operator sites must be integers")
            if site < 0:
                raise ValueError("collapse operator sites must be non-negative")
            if n_sites is not None and site >= n_sites:
                raise ValueError("collapse operator site %d exceeds n_sites" % (site,))

    def expanded_sites(self, n_sites):
        self.validate(n_sites=n_sites)
        if self.sites == "all":
            return tuple(range(n_sites))
        return tuple(self.sites)


@dataclass(frozen=True)
class MethodSpec:
    backend: str
    algorithm: str
    trunc_params: dict[str, Any] = field(default_factory=dict)
    evolution_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        data["trunc_params"] = _clean_dict(data.get("trunc_params"))
        data["evolution_params"] = _clean_dict(data.get("evolution_params"))
        return cls(**data)

    def to_dict(self):
        return asdict(self)

    def validate(self):
        backend = self.backend.lower()
        if backend not in SUPPORTED_METHODS:
            raise ValueError("unsupported backend %r" % (self.backend,))
        if self.algorithm not in SUPPORTED_METHODS[backend]:
            raise ValueError(
                "unsupported algorithm %r for backend %r"
                % (self.algorithm, self.backend)
            )
        if backend in EXACT_BACKENDS and _has_nonempty_truncation(self.trunc_params):
            raise ValueError(
                "truncation parameters are not supported for exact backend %r"
                % (self.backend,)
            )


@dataclass(frozen=True)
class OutputSpec:
    base_dir: str = "../pkl"
    run_id: str | None = None

    @classmethod
    def from_dict(cls, data):
        return cls(**dict(data or {}))

    def to_dict(self):
        return asdict(self)

    def validate(self):
        if not self.base_dir:
            raise ValueError("output base_dir must be non-empty")


@dataclass(frozen=True)
class SimulationSpec:
    model: ModelSpec
    initial_state: InitialStateSpec
    time_grid: TimeGridSpec
    method: MethodSpec
    collapse_operators: tuple[CollapseOperatorSpec, ...] = ()
    output: OutputSpec = field(default_factory=OutputSpec)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        spec = cls(
            model=ModelSpec.from_dict(data["model"]),
            initial_state=InitialStateSpec.from_dict(data["initial_state"]),
            time_grid=TimeGridSpec.from_dict(data["time_grid"]),
            method=MethodSpec.from_dict(data["method"]),
            collapse_operators=tuple(
                CollapseOperatorSpec.from_dict(item)
                for item in data.get("collapse_operators") or ()
            ),
            output=OutputSpec.from_dict(data.get("output")),
        )
        spec.validate()
        return spec

    def to_dict(self):
        return {
            "model": self.model.to_dict(),
            "initial_state": self.initial_state.to_dict(),
            "time_grid": self.time_grid.to_dict(),
            "method": self.method.to_dict(),
            "collapse_operators": [
                collapse.to_dict() for collapse in self.collapse_operators
            ],
            "output": self.output.to_dict(),
        }

    def validate(self):
        self.model.validate()
        self.initial_state.validate()
        self.time_grid.validate()
        self.method.validate()
        for collapse in self.collapse_operators:
            collapse.validate(n_sites=self.model.n_sites)
        method_key = (self.method.backend.lower(), self.method.algorithm)
        if self.collapse_operators and method_key not in DISSIPATIVE_METHODS:
            raise ValueError(
                "collapse operators are not supported for backend %r algorithm %r"
                % (self.method.backend, self.method.algorithm)
            )
        self.output.validate()
