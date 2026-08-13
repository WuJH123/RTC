"""V6 guard around the stable group-preserving Step2 shard/cache infrastructure."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .step2_d3_design_v60 import D3_V60_ACTIVE_ROLE, D3_V60_CANDIDATE_ROLE, D3_V60_HOLD_ROLE
from .step2_shards import compile_step2_shards, sha256_file
from .step2_v60_contract import V60_CONTRACT, V60_D3_DATA_CONTRACT

_ALLOWED_D3 = {D3_V60_HOLD_ROLE, D3_V60_CANDIDATE_ROLE, D3_V60_ACTIVE_ROLE}


def _one_nonempty(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns:
        raise ValueError(f"V6 run index lacks {column}")
    values = set(frame[column].fillna("").astype(str)) - {""}
    if len(values) != 1:
        raise ValueError(f"V6 run index requires exactly one non-empty {column}: {sorted(values)}")
    return next(iter(values))


def validate_v60_run_index_for_shards(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty or "source_kind" not in frame.columns:
        raise ValueError("V6 shard compile requires a non-empty source-aware run index")
    if set(frame["scientific_split"].astype(str).str.lower()) != {"development"}:
        raise ValueError("V6 shard compile is development-only")
    if set(frame["development_fold"].astype(str).str.lower()) != {"train"}:
        raise ValueError("V6 targeted shard compile is Train-only")
    d3 = frame[frame["source_kind"].astype(str).str.upper() == "D3"]
    d2 = frame[frame["source_kind"].astype(str).str.upper() == "D2"]
    if d2.empty or d3.empty:
        raise ValueError("V6 requires both D2 and targeted D3-v2")
    roles = set(d3["data_role"].astype(str))
    if not roles <= _ALLOWED_D3 or D3_V60_HOLD_ROLE not in roles or D3_V60_CANDIDATE_ROLE not in roles:
        raise ValueError("V6 shard compile refuses legacy/incomplete D3 roles")
    if set(d3["v60_data_contract"].astype(str)) != {V60_D3_DATA_CONTRACT}:
        raise ValueError("V6 D3 rows are not bound to the targeted data contract")
    basis_sha = _one_nonempty(d3, "v60_control_basis_sha256")
    design_sha = _one_nonempty(d3, "v60_design_contract_sha256")
    for key, group in d3.groupby(["rainfall_group", "event_id", "checkpoint_id"], dropna=False, sort=False):
        if int((group["data_role"].astype(str) == D3_V60_HOLD_ROLE).sum()) != 1:
            raise ValueError(f"V6 D3 group {key} must contain exactly one HOLD")
    return {
        "contract": "PROJECT7_STEP2_V60_SHARD_INPUT_GUARD_V1",
        "rows": int(len(frame)),
        "d2_rows": int(len(d2)),
        "d3_rows": int(len(d3)),
        "v60_control_basis_sha256": basis_sha,
        "v60_design_contract_sha256": design_sha,
        "legacy_dense_d3_present": False,
    }


def compile_step2_shards_v60(
    run_index: pd.DataFrame,
    *,
    output_dir: str | Path,
    shard_size: int = 128,
    expected_model_step_seconds: int = 300,
    expected_horizon_steps: int = 72,
) -> Path:
    guard = validate_v60_run_index_for_shards(run_index)
    manifest_path = compile_step2_shards(
        run_index,
        output_dir=output_dir,
        shard_size=shard_size,
        expected_model_step_seconds=expected_model_step_seconds,
        expected_horizon_steps=expected_horizon_steps,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.update(
        {
            "v60_contract": V60_CONTRACT,
            "v60_data_contract": V60_D3_DATA_CONTRACT,
            "v60_control_basis_sha256": guard["v60_control_basis_sha256"],
            "v60_design_contract_sha256": guard["v60_design_contract_sha256"],
            "v60_legacy_dense_d3_present": False,
            "v60_source_guard": guard,
        }
    )
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def validate_v60_cache_lineage(cache_manifest_path: str | Path) -> dict[str, Any]:
    cache_path = Path(cache_manifest_path).resolve()
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    source_raw = str(cache.get("source_manifest_path", ""))
    expected_sha = str(cache.get("source_manifest_sha256", ""))
    if not source_raw or not expected_sha:
        raise ValueError("V6 cache manifest lacks source shard lineage")
    source = Path(source_raw)
    if not source.is_file() or sha256_file(source) != expected_sha:
        raise ValueError("V6 source shard manifest is missing or changed")
    manifest = json.loads(source.read_text(encoding="utf-8"))
    if manifest.get("v60_contract") != V60_CONTRACT:
        raise ValueError("cache source is not a V6 shard manifest")
    if manifest.get("v60_data_contract") != V60_D3_DATA_CONTRACT:
        raise ValueError("V6 cache source has the wrong D3 data contract")
    if manifest.get("v60_legacy_dense_d3_present") is not False:
        raise ValueError("V6 cache source admits legacy dense D3")
    basis_sha = str(manifest.get("v60_control_basis_sha256", ""))
    design_sha = str(manifest.get("v60_design_contract_sha256", ""))
    if not basis_sha or not design_sha:
        raise ValueError("V6 cache source lacks basis/design hashes")
    return {
        "contract": "PROJECT7_STEP2_V60_CACHE_LINEAGE_GUARD_V1",
        "cache_manifest": str(cache_path),
        "source_manifest": str(source.resolve()),
        "source_manifest_sha256": expected_sha,
        "v60_control_basis_sha256": basis_sha,
        "v60_design_contract_sha256": design_sha,
        "legacy_dense_d3_present": False,
    }


__all__ = [
    "compile_step2_shards_v60",
    "validate_v60_cache_lineage",
    "validate_v60_run_index_for_shards",
]
