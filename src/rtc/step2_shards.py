from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from .dataset_compile import compile_branches_to_npz


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compile_step2_shards(
    run_index: pd.DataFrame,
    *,
    output_dir: str | Path,
    shard_size: int = 128,
) -> Path:
    """Compile D2/D3 branches in bounded-memory shards instead of one monolithic NPZ."""

    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if "metadata_path" not in run_index.columns or run_index.empty:
        raise ValueError("non-empty run index with metadata_path is required")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, object]] = []
    frame = run_index.reset_index(drop=True)
    for start in range(0, len(frame), shard_size):
        chunk = frame.iloc[start : start + shard_size].reset_index(drop=True)
        shard = out / f"step2_{start//shard_size:05d}.npz"
        compile_branches_to_npz(
            chunk["metadata_path"].astype(str).tolist(),
            shard,
            provenance=chunk,
        )
        shards.append({
            "path": str(shard.resolve()),
            "sha256": sha256_file(shard),
            "rows": int(len(chunk)),
            "start_row": int(start),
        })
    manifest = {
        "contract": "STEP2_SHARDED_DATASET_V2",
        "row_count": int(len(frame)),
        "shard_size": int(shard_size),
        "shards": shards,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def load_shard_manifest(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("contract") != "STEP2_SHARDED_DATASET_V2":
        raise ValueError("not a STEP2_SHARDED_DATASET_V2 manifest")
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Step2 shard manifest is empty")
    for item in shards:
        if not isinstance(item, dict):
            raise ValueError("invalid shard entry")
        p = Path(str(item["path"]))
        if not p.is_file() or sha256_file(p) != str(item["sha256"]):
            raise ValueError(f"Step2 shard missing/changed: {p}")
    return payload
