from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .code_contract import rtc_source_tree_sha256
from .dataset_compile import compile_branches_to_npz


SHARD_CONTRACT = "STEP2_SHARDED_DATASET_V6_COUNTERFACTUAL_GROUP_PRESERVING"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataframe_sha256(frame: pd.DataFrame) -> str:
    text = frame.to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def simulation_identity_set_sha256(frame: pd.DataFrame) -> str:
    if "simulation_identity_sha256" not in frame.columns:
        return ""
    values = frame["simulation_identity_sha256"].fillna("").astype(str)
    if (values == "").all():
        return ""
    if (values == "").any():
        raise ValueError("Step2 run index mixes simulation-identity-aware and legacy rows")
    if values.duplicated().any():
        raise ValueError("Step2 run index duplicates simulation identities")
    canonical = "\n".join(sorted(values.tolist())) + "\n"
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _group_columns(frame: pd.DataFrame) -> list[str]:
    keys = [
        name
        for name in ("source_kind", "rainfall_group", "event_id", "checkpoint_id")
        if name in frame.columns
    ]
    if "checkpoint_id" not in keys:
        return []
    return keys


def _action_sort_column(frame: pd.DataFrame) -> str | None:
    for name in ("candidate_action_sha256", "sequence_sha256", "simulation_identity_sha256"):
        if name in frame.columns:
            return name
    return None


def _group_preserving_chunks(
    frame: pd.DataFrame,
    *,
    shard_size: int,
) -> tuple[pd.DataFrame, list[pd.DataFrame], list[str]]:
    """Sort by counterfactual group and never split a checkpoint group across shards."""

    group_keys = _group_columns(frame)
    if not group_keys:
        ordered = frame.reset_index(drop=True)
        chunks = [
            ordered.iloc[start : start + shard_size].reset_index(drop=True)
            for start in range(0, len(ordered), shard_size)
        ]
        return ordered, chunks, []

    sort_cols = list(group_keys)
    action_col = _action_sort_column(frame)
    if action_col:
        sort_cols.append(action_col)
    ordered = frame.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    groups = [
        group.reset_index(drop=True)
        for _, group in ordered.groupby(group_keys, sort=False, dropna=False)
    ]
    if any(len(group) > shard_size for group in groups):
        largest = max(len(group) for group in groups)
        raise ValueError(
            f"counterfactual checkpoint group ({largest} rows) exceeds shard_size={shard_size}"
        )

    chunks: list[pd.DataFrame] = []
    pending: list[pd.DataFrame] = []
    pending_rows = 0
    for group in groups:
        if pending and pending_rows + len(group) > shard_size:
            chunks.append(pd.concat(pending, ignore_index=True))
            pending = []
            pending_rows = 0
        pending.append(group)
        pending_rows += len(group)
    if pending:
        chunks.append(pd.concat(pending, ignore_index=True))
    return ordered, chunks, group_keys


def compile_step2_shards(
    run_index: pd.DataFrame,
    *,
    output_dir: str | Path,
    shard_size: int = 128,
    expected_model_step_seconds: int | None = None,
    expected_horizon_steps: int | None = None,
) -> Path:
    """Compile bounded-memory D2/D3 shards while keeping same-prefix groups intact."""

    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    if "metadata_path" not in run_index.columns or run_index.empty:
        raise ValueError("non-empty run index with metadata_path is required")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frame, chunks, group_keys = _group_preserving_chunks(
        run_index.reset_index(drop=True),
        shard_size=shard_size,
    )
    identity_set_sha = simulation_identity_set_sha256(frame)
    shards: list[dict[str, object]] = []
    global_step: int | None = None
    global_horizon: int | None = None
    global_engine: str | None = None
    start_row = 0

    for shard_index, chunk in enumerate(chunks):
        shard = out / f"step2_{shard_index:05d}.npz"
        compile_branches_to_npz(
            chunk["metadata_path"].astype(str).tolist(),
            shard,
            provenance=chunk,
        )
        with np.load(shard, allow_pickle=False) as ds:
            step = int(ds["model_step_seconds"].item())
            horizon = int(ds["horizon_steps"].item())
            engine = str(ds["swmm_engine_version"].item())
        if global_step is None:
            global_step, global_horizon, global_engine = step, horizon, engine
        elif step != global_step or horizon != global_horizon or engine != global_engine:
            raise ValueError(
                "Step2 shards do not share one frozen model-step/horizon/SWMM-engine contract"
            )
        shards.append(
            {
                "path": str(shard.resolve()),
                "sha256": sha256_file(shard),
                "rows": int(len(chunk)),
                "start_row": int(start_row),
                "model_step_seconds": step,
                "horizon_steps": horizon,
                "swmm_engine_version": engine,
                "simulation_identity_set_sha256": simulation_identity_set_sha256(chunk),
            }
        )
        start_row += len(chunk)

    assert global_step is not None and global_horizon is not None and global_engine is not None
    if expected_model_step_seconds is not None and global_step != int(expected_model_step_seconds):
        raise ValueError(
            f"Step2 data step {global_step}s differs from frozen production step "
            f"{expected_model_step_seconds}s"
        )
    if expected_horizon_steps is not None and global_horizon != int(expected_horizon_steps):
        raise ValueError(
            f"Step2 data horizon {global_horizon} differs from frozen production horizon "
            f"{expected_horizon_steps}"
        )

    manifest = {
        "contract": SHARD_CONTRACT,
        "row_count": int(len(frame)),
        "shard_size": int(shard_size),
        "model_step_seconds": global_step,
        "horizon_steps": global_horizon,
        "swmm_engine_version": global_engine,
        "source_run_index_sha256": dataframe_sha256(frame),
        "source_simulation_identity_set_sha256": identity_set_sha,
        "simulation_identity_bound": bool(identity_set_sha),
        "rtc_source_tree_sha256": rtc_source_tree_sha256(),
        "counterfactual_groups_preserved": bool(group_keys),
        "counterfactual_group_columns": group_keys,
        "shards": shards,
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def load_shard_manifest(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("contract") != SHARD_CONTRACT:
        raise ValueError(f"not a {SHARD_CONTRACT} manifest")
    if payload.get("rtc_source_tree_sha256") != rtc_source_tree_sha256():
        raise ValueError(
            "Step2 shard manifest was compiled by a different RTC scientific implementation"
        )
    step = int(payload.get("model_step_seconds", 0))
    horizon = int(payload.get("horizon_steps", 0))
    engine = str(payload.get("swmm_engine_version", "")).strip()
    identity_bound = bool(payload.get("simulation_identity_bound", False))
    identity_set_sha = str(payload.get("source_simulation_identity_set_sha256", ""))
    if identity_bound != bool(identity_set_sha):
        raise ValueError("Step2 shard manifest simulation-identity binding is inconsistent")
    if step <= 0 or horizon <= 0 or not engine:
        raise ValueError("Step2 shard manifest lacks a valid time/engine contract")
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("Step2 shard manifest is empty")
    for item in shards:
        if not isinstance(item, dict):
            raise ValueError("invalid shard entry")
        p = Path(str(item["path"]))
        if not p.is_file() or sha256_file(p) != str(item["sha256"]):
            raise ValueError(f"Step2 shard missing/changed: {p}")
        if int(item.get("model_step_seconds", -1)) != step:
            raise ValueError("Step2 shard step differs from manifest")
        if int(item.get("horizon_steps", -1)) != horizon:
            raise ValueError("Step2 shard horizon differs from manifest")
        if str(item.get("swmm_engine_version", "")) != engine:
            raise ValueError("Step2 shard SWMM engine differs from manifest")
        if identity_bound and not str(item.get("simulation_identity_set_sha256", "")):
            raise ValueError("identity-bound Step2 shard entry lacks an identity-set hash")
        with np.load(p, allow_pickle=False) as ds:
            if int(ds["model_step_seconds"].item()) != step:
                raise ValueError(f"Step2 shard embedded time step differs: {p}")
            if int(ds["horizon_steps"].item()) != horizon:
                raise ValueError(f"Step2 shard embedded horizon differs: {p}")
            if str(ds["swmm_engine_version"].item()) != engine:
                raise ValueError(f"Step2 shard embedded SWMM engine differs: {p}")
    return payload
