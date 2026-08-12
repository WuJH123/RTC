"""Dedicated V6 run index: existing D2 plus targeted D3-v2 only.

Legacy dense D3 is intentionally excluded so old/new HOLD rows cannot collide.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .data_index import build_d2_run_index, standardize_d3_run_index
from .step2_d3_design_v60 import D3_V60_ACTIVE_ROLE, D3_V60_CANDIDATE_ROLE, D3_V60_HOLD_ROLE
from .step2_v60_contract import V60_D3_DATA_CONTRACT

_ALLOWED = {D3_V60_HOLD_ROLE, D3_V60_CANDIDATE_ROLE, D3_V60_ACTIVE_ROLE}


def build_step2_v60_run_index(*, d2_manifest: pd.DataFrame, d2_run_summary: pd.DataFrame, d3_v60_run_summary: pd.DataFrame) -> pd.DataFrame:
    d2 = build_d2_run_index(d2_manifest, d2_run_summary)
    d3 = standardize_d3_run_index(d3_v60_run_summary)
    roles = set(d3["data_role"].astype(str))
    if not roles <= _ALLOWED:
        raise ValueError("V6 run index refuses legacy/non-manifold D3 roles: " + ", ".join(sorted(roles - _ALLOWED)))
    if D3_V60_HOLD_ROLE not in roles or D3_V60_CANDIDATE_ROLE not in roles:
        raise ValueError("V6 run index requires HOLD and targeted manifold candidates")
    if set(d3["scientific_split"].astype(str).str.lower()) != {"development"}:
        raise ValueError("V6 targeted D3 is development-only")
    if set(d3["development_fold"].astype(str).str.lower()) != {"train"}:
        raise ValueError("V6 targeted D3 generation is Train-only")
    for key, group in d3.groupby(["rainfall_group", "event_id", "checkpoint_id"], sort=False, dropna=False):
        holds = int((group["data_role"].astype(str) == D3_V60_HOLD_ROLE).sum())
        if holds != 1:
            raise ValueError(f"V6 D3 group {key} requires exactly one HOLD, got {holds}")
        if not group["data_role"].astype(str).isin([D3_V60_CANDIDATE_ROLE, D3_V60_ACTIVE_ROLE]).any():
            raise ValueError(f"V6 D3 group {key} has no targeted candidate")
    result = pd.concat([d2, d3], ignore_index=True)
    if result["metadata_path"].duplicated().any():
        raise ValueError("V6 run index duplicates a simulation metadata branch")
    if (result["scientific_split"].astype(str).str.lower() == "final").any():
        raise ValueError("V6 run index refuses Final rows")
    for raw in result["metadata_path"].astype(str):
        if not Path(raw).is_file():
            raise ValueError(f"V6 branch metadata is missing: {raw}")
    return result.sort_values(["scientific_split","development_fold","rainfall_group","event_id","checkpoint_id","source_kind","data_role","action_or_sequence_sha256"]).reset_index(drop=True)


def v60_run_index_summary(frame: pd.DataFrame) -> dict[str, object]:
    d3 = frame[frame["source_kind"].astype(str) == "D3"]
    return {
        "contract":"PROJECT7_STEP2_V60_RUN_INDEX_D2_PLUS_TARGETED_D3_V1",
        "d3_data_contract":V60_D3_DATA_CONTRACT,
        "rows":int(len(frame)),
        "d2_rows":int((frame["source_kind"].astype(str)=="D2").sum()),
        "d3_rows":int(len(d3)),
        "d3_roles":sorted(d3["data_role"].astype(str).unique().tolist()),
        "legacy_dense_d3_present":bool(d3["data_role"].astype(str).isin(["D3_MULTI_ACTUATOR_ROLLOUT","D3_MULTI_ACTUATOR_SEQUENCE"]).any()),
        "rainfall_groups":int(frame["rainfall_group"].nunique()),
        "scientific_splits":sorted(frame["scientific_split"].astype(str).unique().tolist()),
        "development_folds":sorted(frame["development_fold"].astype(str).unique().tolist()),
    }


__all__=["build_step2_v60_run_index","v60_run_index_summary"]
