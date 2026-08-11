"""Rebuildable mmap cache for Step2 V6 shards.

The cache is derived data only.  Every cache manifest binds each ``.npy`` array to
the source V6 shard SHA, shape, dtype, and this cache contract.  It never changes
the V6 source shard or any SWMM artifact.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .step2_shards import load_shard_manifest, sha256_file
from .step2_stability import STABILITY_CACHE_CONTRACT

CACHE_ARRAY_FIELDS = (
    "initial_state",
    "rainfall",
    "settings",
    "previous_actuator_flow",
    "target_states",
    "target_actuator_flows",
    "elapsed_seconds",
    "exact_node_flood_volume_m3",
    "node_ids",
    "actuator_ids",
    "action_or_sequence_sha256",
    "event_id",
    "rainfall_group",
    "scientific_split",
    "development_fold",
    "data_role",
    "checkpoint_id",
    "base_action_sha256",
    "source_kind",
)


def materialize_shard_arrays(raw: Any) -> dict[str, np.ndarray]:
    """Decompress each required NPZ field once and make it slice-safe."""

    missing = [name for name in CACHE_ARRAY_FIELDS if name not in raw.files]
    if missing:
        raise ValueError(f"Step2 shard lacks required fields: {missing}")
    return {
        name: np.ascontiguousarray(raw[name])
        for name in CACHE_ARRAY_FIELDS
    }


def _cache_manifest_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "CACHE_MANIFEST.json"


def _cache_is_current(
    cache_manifest: dict[str, Any],
    *,
    source_manifest_sha256: str,
    source_manifest: dict[str, Any],
    out_dir: Path,
) -> bool:
    if cache_manifest.get("contract") != STABILITY_CACHE_CONTRACT:
        return False
    if cache_manifest.get("source_manifest_sha256") != source_manifest_sha256:
        return False
    entries = cache_manifest.get("shards")
    source_entries = source_manifest.get("shards")
    if not isinstance(entries, list) or not isinstance(source_entries, list):
        return False
    if len(entries) != len(source_entries):
        return False
    for cached, source in zip(entries, source_entries, strict=True):
        if cached.get("source_sha256") != source.get("sha256"):
            return False
        shard_dir = out_dir / str(cached.get("directory", ""))
        arrays = cached.get("arrays")
        if not isinstance(arrays, dict):
            return False
        for name, metadata in arrays.items():
            path = shard_dir / str(metadata.get("file", ""))
            if not path.is_file():
                return False
            try:
                arr = np.load(path, mmap_mode="r", allow_pickle=False)
                shape = list(arr.shape)
                dtype = str(arr.dtype)
                arr._mmap.close() if hasattr(arr, "_mmap") and arr._mmap is not None else None
            except Exception:
                return False
            if shape != metadata.get("shape") or dtype != metadata.get("dtype"):
                return False
    return True


def build_step2_training_cache(
    manifest_path: str | Path,
    out_dir: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Build or reuse a content-bound mmap cache for every V6 shard."""

    source_manifest = load_shard_manifest(manifest_path)
    source_manifest_sha256 = sha256_file(manifest_path)
    out = Path(out_dir)
    cache_path = _cache_manifest_path(out)
    if not force and cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if _cache_is_current(
                cached,
                source_manifest_sha256=source_manifest_sha256,
                source_manifest=source_manifest,
                out_dir=out,
            ):
                return cache_path
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for shard_index, item in enumerate(source_manifest["shards"]):
        source_path = Path(str(item["path"]))
        shard_dir = tmp / f"shard_{shard_index:05d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        with np.load(source_path, allow_pickle=False) as raw:
            arrays = materialize_shard_arrays(raw)
            metadata: dict[str, Any] = {}
            for name, value in arrays.items():
                filename = f"{name}.npy"
                np.save(shard_dir / filename, value, allow_pickle=False)
                metadata[name] = {
                    "file": filename,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
        entries.append(
            {
                "source_path": str(source_path.resolve()),
                "source_sha256": str(item["sha256"]),
                "directory": shard_dir.name,
                "rows": int(item["rows"]),
                "arrays": metadata,
            }
        )
    cache_payload = {
        "contract": STABILITY_CACHE_CONTRACT,
        "source_manifest_path": str(Path(manifest_path).resolve()),
        "source_manifest_sha256": source_manifest_sha256,
        "source_shard_count": len(entries),
        "source_row_count": int(source_manifest["row_count"]),
        "model_step_seconds": int(source_manifest["model_step_seconds"]),
        "horizon_steps": int(source_manifest["horizon_steps"]),
        "swmm_engine_version": str(source_manifest["swmm_engine_version"]),
        "shards": entries,
    }
    (tmp / "CACHE_MANIFEST.json").write_text(
        json.dumps(cache_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    return cache_path


def load_step2_training_cache(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("contract") != STABILITY_CACHE_CONTRACT:
        raise ValueError("not a Step2 rebuildable training cache")
    base = Path(path).parent
    loaded = dict(payload)
    loaded_shards: list[dict[str, Any]] = []
    for entry in payload.get("shards", []):
        arrays: dict[str, np.ndarray] = {}
        directory = base / str(entry["directory"])
        for name, metadata in entry["arrays"].items():
            array_path = directory / str(metadata["file"])
            array = np.load(array_path, mmap_mode="r", allow_pickle=False)
            if list(array.shape) != metadata["shape"] or str(array.dtype) != metadata["dtype"]:
                raise ValueError(f"cache array changed: {array_path}")
            arrays[name] = array
        loaded_shards.append({**entry, "path": str(directory), "arrays": arrays})
    loaded["shards"] = loaded_shards
    return loaded
