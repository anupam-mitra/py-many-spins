from dataclasses import asdict, dataclass, field
from typing import Any, Type, TypeVar

T = TypeVar("T")


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


def _clean_dict(value: Any) -> dict[str, Any]:
    """
    Ensure the value is a dictionary.

    Parameters
    ----------
    value : Any
        The value to clean.

    Returns
    -------
    dict[str, Any]
        A dictionary representation of the value.
    """
    return dict(value or {})


def _has_nonempty_truncation(trunc_params: Any) -> bool:
    """
    Check if truncation parameters contain any non-None values.

    Parameters
    ----------
    trunc_params : Any
        The truncation parameters to check.

    Returns
    -------
    bool
        True if at least one parameter is not None, False otherwise.
    """
    return any(value is not None for value in _clean_dict(trunc_params).values())


@dataclass(frozen=True)
class ModelSpec:
    """
    Specification for the physical model.

    Attributes
    ----------
    kind : str
        The kind of model.
    n_sites : int
        Number of sites in the model.
    j_xx : float
        Interaction strength J_xx.
    b_z : float
        Transverse field B_z.
    b_x : float
        Transverse field B_x.
    bc : str
        Boundary conditions ('open' or 'periodic').
    """
    kind: str
    n_sites: int
    j_xx: float
    b_z: float
    b_x: float = 0.0
    bc: str = "open"

    @classmethod
    def tilted_field_ising_1d(
        cls: Type["ModelSpec"],
        n_sites: int,
        j_xx: float,
        b_z: float,
        b_x: float = 0.0,
        bc: str = "open",
    ) -> "ModelSpec":
        """
        Create a specification for a 1D tilted-field Ising model.

        Parameters
        ----------
        n_sites : int
            Number of sites.
        j_xx : float
            Interaction strength J_xx.
        b_z : float
            Transverse field B_z.
        b_x : float, optional
            Transverse field B_x, by default 0.0.
        bc : str, optional
            Boundary conditions, by default "open".

        Returns
        -------
        ModelSpec
            The created ModelSpec.
        """
        return cls(
            kind="tilted_field_ising_1d",
            n_sites=n_sites,
            j_xx=j_xx,
            b_z=b_z,
            b_x=b_x,
            bc=bc,
        )

    @classmethod
    def from_dict(cls: Type["ModelSpec"], data: dict[str, Any]) -> "ModelSpec":
        """
        Create a ModelSpec from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The data dictionary.

        Returns
        -------
        ModelSpec
            The created ModelSpec.
        """
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the ModelSpec to a dictionary.

        Returns
        -------
        dict[str, Any]
            The dictionary representation.
        """
        return asdict(self)

    def validate(self) -> None:
        """
        Validate the ModelSpec parameters.

        Raises
        ------
        ValueError
            If the model kind is unsupported, n_sites is non-positive,
            or boundary conditions are invalid.
        """
        if self.kind != "tilted_field_ising_1d":
            raise ValueError("unsupported model kind %r" % (self.kind,))
        if self.n_sites <= 0:
            raise ValueError("n_sites must be positive")
        if self.bc not in ("open", "periodic"):
            raise ValueError("bc must be 'open' or 'periodic'")


@dataclass(frozen=True)
class InitialStateSpec:
    """
    Specification for the initial state of the simulation.

    Attributes
    ----------
    kind : str
        The kind of initial state.
    theta : float
        Theta angle for the spin coherent state.
    phi : float
        Phi angle for the spin coherent state.
    """
    kind: str = "spin_coherent_product"
    theta: float = 0.0
    phi: float = 0.0

    @classmethod
    def spin_coherent_product(
        cls: Type["InitialStateSpec"], theta: float, phi: float
    ) -> "InitialStateSpec":
        """
        Create a specification for a spin-coherent product state.

        Parameters
        ----------
        theta : float
            Theta angle.
        phi : float
            Phi angle.

        Returns
        -------
        InitialStateSpec
            The created InitialStateSpec.
        """
        return cls(kind="spin_coherent_product", theta=theta, phi=phi)

    @classmethod
    def from_dict(cls: Type["InitialStateSpec"], data: dict[str, Any]) -> "InitialStateSpec":
        """
        Create an InitialStateSpec from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The data dictionary.

        Returns
        -------
        InitialStateSpec
            The created InitialStateSpec.
        """
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the InitialStateSpec to a dictionary.

        Returns
        -------
        dict[str, Any]
            The dictionary representation.
        """
        return asdict(self)

    def validate(self) -> None:
        """
        Validate the InitialStateSpec parameters.

        Raises
        ------
        ValueError
            If the initial state kind is unsupported.
        """
        if self.kind != "spin_coherent_product":
            raise ValueError("unsupported initial state kind %r" % (self.kind,))


@dataclass(frozen=True)
class TimeGridSpec:
    """
    Specification for the time grid.

    Attributes
    ----------
    kind : str
        The kind of time grid.
    start : float
        The start time.
    stop : float
        The stop time.
    num : int
        The number of points in the grid.
    """
    kind: str = "linspace"
    start: float = 0.0
    stop: float = 1.0
    num: int = 2

    @classmethod
    def linspace(
        cls: Type["TimeGridSpec"], start: float, stop: float, num: int
    ) -> "TimeGridSpec":
        """
        Create a specification for a linear time grid.

        Parameters
        ----------
        start : float
            Start time.
        stop : float
            Stop time.
        num : int
            Number of points.

        Returns
        -------
        TimeGridSpec
            The created TimeGridSpec.
        """
        return cls(kind="linspace", start=start, stop=stop, num=num)

    @classmethod
    def from_dict(cls: Type["TimeGridSpec"], data: dict[str, Any]) -> "TimeGridSpec":
        """
        Create a TimeGridSpec from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The data dictionary.

        Returns
        -------
        TimeGridSpec
            The created TimeGridSpec.
        """
        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the TimeGridSpec to a dictionary.

        Returns
        -------
        dict[str, Any]
            The dictionary representation.
        """
        return asdict(self)

    def validate(self) -> None:
        """
        Validate the TimeGridSpec parameters.

        Raises
        ------
        ValueError
            If the time grid kind is unsupported or num is non-positive.
        """
        if self.kind != "linspace":
            raise ValueError("unsupported time grid kind %r" % (self.kind,))
        if self.num <= 0:
            raise ValueError("time grid num must be positive")


@dataclass(frozen=True)
class CollapseOperatorSpec:
    """
    Specification for a collapse operator.

    Attributes
    ----------
    kind : str
        The kind of collapse operator.
    operator : str
        The operator type (e.g., 'sigmam').
    rate : float
        The collapse rate.
    sites : Any
        The sites to which the operator is applied.
    """
    kind: str = "local_spin"
    operator: str = "sigmam"
    rate: float = 1.0
    sites: Any = "all"

    def __post_init__(self) -> None:
        """
        Normalize collapse-operator fields after initialization.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        object.__setattr__(self, "operator", self.operator.lower())
        if isinstance(self.sites, int):
            object.__setattr__(self, "sites", (self.sites,))
        elif self.sites != "all" and not isinstance(self.sites, tuple):
            object.__setattr__(self, "sites", tuple(self.sites))

    @classmethod
    def local_spin(
        cls: Type["CollapseOperatorSpec"], operator: str, rate: float, sites: Any = "all"
    ) -> "CollapseOperatorSpec":
        """
        Create a specification for a local spin collapse operator.

        Parameters
        ----------
        operator : str
            The operator type.
        rate : float
            The collapse rate.
        sites : Any, optional
            The sites to which the operator is applied, by default "all".

        Returns
        -------
        CollapseOperatorSpec
            The created CollapseOperatorSpec.
        """
        return cls(kind="local_spin", operator=operator, rate=rate, sites=sites)

    @classmethod
    def from_dict(cls: Type["CollapseOperatorSpec"], data: dict[str, Any]) -> "CollapseOperatorSpec":
        """
        Create a CollapseOperatorSpec from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The data dictionary.

        Returns
        -------
        CollapseOperatorSpec
            The created CollapseOperatorSpec.
        """
        data = dict(data)
        data.setdefault("kind", "local_spin")
        data.setdefault("sites", "all")
        if data["sites"] != "all" and not isinstance(data["sites"], int):
            data["sites"] = tuple(data["sites"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the CollapseOperatorSpec to a dictionary.

        Returns
        -------
        dict[str, Any]
            The dictionary representation.
        """
        return {
            "kind": self.kind,
            "operator": self.operator,
            "rate": self.rate,
            "sites": list(self.sites) if self.sites != "all" else "all",
        }

    def validate(self, n_sites: int | None = None) -> None:
        """
        Validate the CollapseOperatorSpec parameters.

        Parameters
        ----------
        n_sites : int, optional
            The total number of sites in the model, by default None.

        Raises
        ------
        ValueError
            If the kind, operator, rate, or sites are invalid.
        """
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

    def expanded_sites(self, n_sites: int) -> tuple[int, ...]:
        """
        Expand the sites specification to a tuple of indices.

        Parameters
        ----------
        n_sites : int
            The total number of sites in the model.

        Returns
        -------
        tuple[int, ...]
            The expanded tuple of site indices.
        """
        self.validate(n_sites=n_sites)
        if self.sites == "all":
            return tuple(range(n_sites))
        return tuple(self.sites)


@dataclass(frozen=True)
class MethodSpec:
    """
    Specification for the evolution method.

    Attributes
    ----------
    backend : str
        The backend to use.
    algorithm : str
        The algorithm to use.
    trunc_params : dict[str, Any]
        Truncation parameters for MPS methods.
    evolution_params : dict[str, Any]
        Additional evolution parameters.
    """
    backend: str
    algorithm: str
    trunc_params: dict[str, Any] = field(default_factory=dict)
    evolution_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls: Type["MethodSpec"], data: dict[str, Any]) -> "MethodSpec":
        """
        Create a MethodSpec from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The data dictionary.

        Returns
        -------
        MethodSpec
            The created MethodSpec.
        """
        data = dict(data)
        data["trunc_params"] = _clean_dict(data.get("trunc_params"))
        data["evolution_params"] = _clean_dict(data.get("evolution_params"))
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the MethodSpec to a dictionary.

        Returns
        -------
        dict[str, Any]
            The dictionary representation.
        """
        return asdict(self)

    def validate(self) -> None:
        """
        Validate the MethodSpec parameters.

        Raises
        ------
        ValueError
            If the backend or algorithm is unsupported, or if truncation
            parameters are provided for an exact backend.
        """
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
    """
    Specification for the output of the simulation.

    Attributes
    ----------
    base_dir : str
        The base directory for outputs.
    run_id : str | None
        The unique ID for the run.
    """
    base_dir: str = "../pkl"
    run_id: str | None = None

    @classmethod
    def from_dict(cls: Type["OutputSpec"], data: dict[str, Any]) -> "OutputSpec":
        """
        Create an OutputSpec from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The data dictionary.

        Returns
        -------
        OutputSpec
            The created OutputSpec.
        """
        return cls(**dict(data or {}))

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the OutputSpec to a dictionary.

        Returns
        -------
        dict[str, Any]
            The dictionary representation.
        """
        return asdict(self)

    def validate(self) -> None:
        """
        Validate the OutputSpec parameters.

        Raises
        ------
        ValueError
            If the base_dir is empty.
        """
        if not self.base_dir:
            raise ValueError("output base_dir must be non-empty")


@dataclass(frozen=True)
class SimulationSpec:
    """
    Full specification for a simulation run.

    Attributes
    ----------
    model : ModelSpec
        The physical model specification.
    initial_state : InitialStateSpec
        The initial state specification.
    time_grid : TimeGridSpec
        The time grid specification.
    method : MethodSpec
        The evolution method specification.
    collapse_operators : tuple[CollapseOperatorSpec, ...]
        A tuple of collapse operators for dissipative simulations.
    output : OutputSpec
        The output specification.
    """
    model: ModelSpec
    initial_state: InitialStateSpec
    time_grid: TimeGridSpec
    method: MethodSpec
    collapse_operators: tuple[CollapseOperatorSpec, ...] = ()
    output: OutputSpec = field(default_factory=OutputSpec)

    @classmethod
    def from_dict(cls: Type["SimulationSpec"], data: dict[str, Any]) -> "SimulationSpec":
        """
        Create a SimulationSpec from a dictionary.

        Parameters
        ----------
        data : dict[str, Any]
            The data dictionary.

        Returns
        -------
        SimulationSpec
            The created SimulationSpec.
        """
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

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the SimulationSpec to a dictionary.

        Returns
        -------
        dict[str, Any]
            The dictionary representation.
        """
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

    def validate(self) -> None:
        """
        Validate the SimulationSpec and its components.

        Raises
        ------
        ValueError
            If any component is invalid, or if collapse operators are
            provided for a non-dissipative method.
        """
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
