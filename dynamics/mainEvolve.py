import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from time_evolution.config_io import load_simulation_spec
from time_evolution.runner import run_and_store


logging.basicConfig(
    format="%(asctime)s: %(levelname)s: %(message)s",
    level=logging.INFO,
)


def _apply_overrides(spec, args):
    model = spec.model
    if args.systemsize is not None:
        model = replace(model, n_sites=args.systemsize)

    method = spec.method
    backend = args.backend if args.backend is not None else method.backend
    algorithm = args.algorithm if args.algorithm is not None else method.algorithm
    trunc_params = dict(method.trunc_params)
    if args.bonddim is not None:
        trunc_params["chi_max"] = args.bonddim

    method = replace(
        method,
        backend=backend,
        algorithm=algorithm,
        trunc_params=trunc_params,
    )
    return replace(spec, model=model, method=method)


def main():
    argument_parser = argparse.ArgumentParser(
        prog="manyspin_evolve",
        description="Run a JSON-configured many-spin evolution.",
    )
    argument_parser.add_argument("--config", required=True)
    argument_parser.add_argument("--backend")
    argument_parser.add_argument("--algorithm")
    argument_parser.add_argument("--systemsize", type=int)
    argument_parser.add_argument("--bonddim", type=int)

    args = argument_parser.parse_args()
    logging.info("Input arguments = %s", vars(args))

    try:
        spec = _apply_overrides(load_simulation_spec(args.config), args)
        spec.validate()
    except ValueError as exc:
        argument_parser.error(str(exc))

    store, index = run_and_store(spec)
    logging.info("Saved run %s with %d states", store.run_dir, len(index["records"]))
    print(store.run_dir)


if __name__ == "__main__":
    main()
