import json
from pathlib import Path

from time_evolution.specs import SimulationSpec


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as infile:
        return json.load(infile)


def dump_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as outfile:
        json.dump(data, outfile, indent=2, sort_keys=True)
        outfile.write("\n")


def load_simulation_spec(path):
    return SimulationSpec.from_dict(load_json(path))


def dump_simulation_spec(spec, path):
    dump_json(spec.to_dict(), path)
