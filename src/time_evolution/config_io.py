import json
from pathlib import Path
from typing import Any

from time_evolution.specs import SimulationSpec


def load_json(path: str | Path) -> Any:
    """
    Load data from a JSON file.

    Parameters
    ----------
    path : str | Path
        The path to the JSON file.

    Returns
    -------
    Any
        The loaded JSON data.
    """
    with Path(path).open("r", encoding="utf-8") as infile:
        return json.load(infile)


def dump_json(data: Any, path: str | Path) -> None:
    """
    Save data to a JSON file.

    Parameters
    ----------
    data : Any
        The data to dump.
    path : str | Path
        The path to the JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=2, sort_keys=True)
        outfile.write("\n")


def load_simulation_spec(path: str | Path) -> SimulationSpec:
    """
    Load a SimulationSpec from a JSON file.

    Parameters
    ----------
    path : str | Path
        The path to the JSON file.

    Returns
    -------
    SimulationSpec
        The loaded SimulationSpec.
    """
    return SimulationSpec.from_dict(load_json(path))


def dump_simulation_spec(spec: SimulationSpec, path: str | Path) -> None:
    """
    Save a SimulationSpec to a JSON file.

    Parameters
    ----------
    spec : SimulationSpec
        The SimulationSpec to dump.
    path : str | Path
        The path to the JSON file.
    """
    dump_json(spec.to_dict(), path)

