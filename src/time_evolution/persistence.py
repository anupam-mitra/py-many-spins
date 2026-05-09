import pickle
import uuid
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from time_evolution.config_io import dump_json, load_json


SCHEMA_VERSION = 1


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value):
    if isinstance(value, (complex, np.complexfloating)):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _payload_id(record, label, used_ids):
    for key in ("state_id", "marginal_id", "uuid_str"):
        if record.get(key) is not None:
            base_id = str(record[key])
            break
    else:
        base_id = "%s-%s" % (label, uuid.uuid4())

    base_id = base_id.replace("/", "_")
    payload_id = base_id
    suffix = 1
    while payload_id in used_ids:
        payload_id = "%s-%d" % (base_id, suffix)
        suffix += 1
    used_ids.add(payload_id)
    return payload_id


def _is_hdf5_payload(payload):
    return isinstance(payload, np.ndarray)


class RunStore:
    """Filesystem store for one JSON-indexed simulation run."""

    def __init__(self, base_dir, run_id=None):
        self.base_dir = Path(base_dir)
        self.run_id = run_id or "%s" % uuid.uuid4()
        self.run_dir = self.base_dir / "runs" / self.run_id

    @property
    def states_dir(self):
        return self.run_dir / "states"

    @property
    def manifest_path(self):
        return self.run_dir / "manifest.json"

    @property
    def spec_path(self):
        return self.run_dir / "spec.json"

    def ensure_dirs(self):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.states_dir.mkdir(parents=True, exist_ok=True)

    def save_spec(self, spec):
        self.ensure_dirs()
        dump_json(spec.to_dict(), self.spec_path)

    def load_manifest(self):
        if self.manifest_path.exists():
            return load_json(self.manifest_path)
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at": _utc_now(),
            "updated_at": None,
            "spec": "spec.json",
            "states": None,
            "marginal_runs": [],
        }

    def save_manifest(self, manifest):
        manifest = dict(manifest)
        manifest["schema_version"] = SCHEMA_VERSION
        manifest["run_id"] = self.run_id
        manifest["updated_at"] = _utc_now()
        dump_json(_json_ready(manifest), self.manifest_path)

    def save_evolution_result(self, spec, result):
        self.ensure_dirs()
        self.save_spec(spec)
        records = self._save_payload_records(
            directory=self.states_dir,
            label="state",
            records=result.records,
            payloads=result.states,
        )
        index = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "kind": "states",
            "metadata": _json_ready(result.metadata),
            "records": records,
        }
        dump_json(index, self.states_dir / "index.json")

        manifest = self.load_manifest()
        manifest["states"] = "states/index.json"
        manifest["metadata"] = _json_ready(result.metadata)
        self.save_manifest(manifest)
        return index

    def load_evolution_index(self):
        return load_json(self.states_dir / "index.json")

    def save_marginal_result(self, result, marginal_run_id=None):
        self.ensure_dirs()
        marginal_run_id = marginal_run_id or "%s" % uuid.uuid4()
        marginal_dir = self.run_dir / "marginals" / marginal_run_id
        marginal_dir.mkdir(parents=True, exist_ok=True)

        records = self._save_payload_records(
            directory=marginal_dir,
            label="marginal",
            records=result.records,
            payloads=result.marginals,
        )
        index = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "marginal_run_id": marginal_run_id,
            "kind": "marginals",
            "metadata": _json_ready(result.metadata),
            "records": records,
        }
        index_path = marginal_dir / "index.json"
        dump_json(index, index_path)

        manifest = self.load_manifest()
        marginal_record = {
            "marginal_run_id": marginal_run_id,
            "index": str(index_path.relative_to(self.run_dir)),
        }
        existing = manifest.setdefault("marginal_runs", [])
        existing[:] = [
            item for item in existing
            if item.get("marginal_run_id") != marginal_run_id
        ]
        existing.append(marginal_record)
        self.save_manifest(manifest)
        return index

    def load_marginal_index(self, marginal_run_id):
        return load_json(self.run_dir / "marginals" / marginal_run_id / "index.json")

    def load_payload(self, storage):
        kind = storage["kind"]
        payload_path = self.run_dir / storage["path"]
        if kind == "pickle":
            with payload_path.open("rb") as infile:
                return pickle.load(infile)
        if kind == "hdf5":
            with h5py.File(payload_path, "r") as h5file:
                return h5file[storage["dataset"]][()]
        raise ValueError("unsupported payload storage kind %r" % (kind,))

    def _save_payload_records(self, directory, label, records, payloads):
        if len(records) != len(payloads):
            raise ValueError(
                "record count %d does not match payload count %d"
                % (len(records), len(payloads))
            )

        records_out = []
        used_ids = set()
        objects_dir = directory / "objects"
        arrays_path = directory / "arrays.h5"

        for record, payload in zip(records, payloads):
            record = dict(record)
            payload_id = _payload_id(record, label, used_ids)
            record["%s_id" % label] = payload_id

            if _is_hdf5_payload(payload):
                directory.mkdir(parents=True, exist_ok=True)
                with h5py.File(arrays_path, "a") as h5file:
                    if payload_id in h5file:
                        del h5file[payload_id]
                    h5file.create_dataset(payload_id, data=payload)
                storage = {
                    "kind": "hdf5",
                    "path": str(arrays_path.relative_to(self.run_dir)),
                    "dataset": payload_id,
                }
            else:
                objects_dir.mkdir(parents=True, exist_ok=True)
                object_path = objects_dir / ("%s.pkl" % payload_id)
                with object_path.open("wb") as outfile:
                    pickle.dump(payload, outfile)
                storage = {
                    "kind": "pickle",
                    "path": str(object_path.relative_to(self.run_dir)),
                }

            record["storage"] = storage
            records_out.append(_json_ready(record))

        return records_out
