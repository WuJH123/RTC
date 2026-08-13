"""Dedicated V6 Step2 run index: D2 + targeted D3-v2 only.

Legacy dense D3 is intentionally excluded so old and new HOLD rows can never collide
inside the group-preserving shard compiler. Targeted D3 design/basis lineage is
rejoined after execution because the generic D3 run summary intentionally carries
only execution identity fields.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data_index import build_d2_run_index, standardize_d3_run_index
from .step2_d3_design_v60 import D3_V60_ACTIVE_ROLE, D3_V60_CANDIDATE_ROLE, D3_V60_HOLD_ROLE
from .step2_v60_contract import V60_D3_DATA_CONTRACT

_ALLOWED_D3_ROLES = {D3_V60_HOLD_ROLE, D3_V60_CANDIDATE_ROLE, D3_V60_ACTIVE_ROLE}
_PROVENANCE_COLUMNS = (
    "v60_data_contract",
    "v60_control_basis_contract",
    "v60_control_basis_sha256",
    "v60_design_contract_sha256",
    "candidate_family",
    "v60_coefficients_json",
    "active_control_groups",
    "active_actuators",
)


def _d3_manifest_provenance(manifest: pd.DataFrame) -> pd.DataFrame:
    required = {
        "sequence_sha256",
        "checkpoint_id",
        "event_id",
        "rainfall_group",
        "scientific_split",
        "development_fold",
        "data_role",
        *_PROVENANCE_COLUMNS,
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"V6 D3 manifest lacks provenance columns: {missing}")
    if set(manifest["v60_data_contract"].astype(str)) != {V60_D3_DATA_CONTRACT}:
        raise ValueError("V6 D3 manifest has an incompatible data contract")
    for column in ("v60_control_basis_sha256", "v60_design_contract_sha256"):
        values = set(manifest[column].fillna("").astype(str))
        if len(values) != 1 or "" in values:
            raise ValueError(f"V6 D3 manifest must have exactly one non-empty {column}")
    if manifest.duplicated(["checkpoint_id", "sequence_sha256"]).any():
        raise ValueError("V6 D3 manifest duplicates checkpoint/sequence identity")
    result = manifest[
        [
            "checkpoint_id",
            "sequence_sha256",
            "event_id",
            "rainfall_group",
            "scientific_split",
            "development_fold",
            "data_role",
            *_PROVENANCE_COLUMNS,
        ]
    ].copy()
    return result.rename(columns={"sequence_sha256": "action_or_sequence_sha256"})


def build_step2_v60_run_index(
    *,
    d2_manifest: pd.DataFrame,
    d2_run_summary: pd.DataFrame,
    d3_v60_manifest: pd.DataFrame,
    d3_v60_run_summary: pd.DataFrame,
) -> pd.DataFrame:
    d2 = build_d2_run_index(d2_manifest, d2_run_summary)
    d3 = standardize_d3_run_index(d3_v60_run_summary)
    provenance = _d3_manifest_provenance(d3_v60_manifest)
    keys = ["checkpoint_id", "action_or_sequence_sha256"]
    joined = d3.merge(
        provenance,
        on=keys,
        how="inner",
        suffixes=("_run", "_manifest"),
        validate="one_to_one",
    )
    if len(joined) != len(d3) or len(joined) != len(provenance):
        raise ValueError(
            "V6 D3 run summary and design manifest are not one-to-one; "
            "do not continue with incomplete or stale targeted D3 lineage"
        )
    for column in ("event_id", "rainfall_group", "scientific_split", "development_fold", "data_role"):
        run = joined[f"{column}_run"].fillna("").astype(str)
        designed = joined[f"{column}_manifest"].fillna("").astype(str)
        mismatch = (run != "") & (designed != "") & (run != designed)
        if mismatch.any():
            raise ValueError(f"V6 D3 manifest/run-summary lineage mismatch in {column}")
        joined[column] = designed.where(designed != "", run)

    d3_enriched = pd.DataFrame()
    for column in d3.columns:
        if column in joined.columns:
            d3_enriched[column] = joined[column]
        elif f"{column}_run" in joined.columns:
            d3_enriched[column] = joined[f"{column}_run"]
        else:
            raise ValueError(f"V6 enriched D3 lost standardized column: {column}")
    for column in ("event_id", "rainfall_group", "scientific_split", "development_fold", "data_role"):
        d3_enriched[column] = joined[column]
    for column in _PROVENANCE_COLUMNS:
        d3_enriched[column] = joined[column]

    roles = set(d3_enriched["data_role"].astype(str))
    if not roles <= _ALLOWED_D3_ROLES:
        raise ValueError(
            "V6 run index refuses legacy/non-manifold D3 roles: "
            + ", ".join(sorted(roles - _ALLOWED_D3_ROLES))
        )
    if D3_V60_HOLD_ROLE not in roles or D3_V60_CANDIDATE_ROLE not in roles:
        raise ValueError("V6 run index requires HOLD and targeted manifold D3 roles")
    if set(d3_enriched["scientific_split"].astype(str).str.lower()) != {"development"}:
        raise ValueError("V6 targeted D3 is development-only")
    if set(d3_enriched["development_fold"].astype(str).str.lower()) != {"train"}:
        raise ValueError("V6 targeted D3 generation is Train-only")

    group_cols = ["rainfall_group", "event_id", "checkpoint_id"]
    for key, group in d3_enriched.groupby(group_cols, sort=False, dropna=False):
        hold_count = int((group["data_role"].astype(str) == D3_V60_HOLD_ROLE).sum())
        if hold_count != 1:
            raise ValueError(f"V6 D3 group {key} requires exactly one HOLD, got {hold_count}")
        if not group["data_role"].astype(str).isin(
            [D3_V60_CANDIDATE_ROLE, D3_V60_ACTIVE_ROLE]
        ).any():
            raise ValueError(f"V6 D3 group {key} has no targeted candidate")

    result = pd.concat([d2, d3_enriched], ignore_index=True, sort=False)
    if result["metadata_path"].duplicated().any():
        raise ValueError("V6 run index duplicates a simulation metadata branch")
    if (result["scientific_split"].astype(str).str.lower() == "final").any():
        raise ValueError("V6 run index refuses Final rows")
    for raw in result["metadata_path"].astype(str):
        if not Path(raw).is_file():
            raise ValueError(f"V6 branch metadata is missing: {raw}")
    return result.sort_values(
        [
            "scientific_split",
            "development_fold",
            "rainfall_group",
            "event_id",
            "checkpoint_id",
            "source_kind",
            "data_role",
            "action_or_sequence_sha256",
        ]
    ).reset_index(drop=True)


def v60_run_index_summary(frame: pd.DataFrame) -> dict[str, object]:
    d3 = frame[frame["source_kind"].astype(str) == "D3"]

    def unique_value(column: str) -> str:
        values = set(d3[column].dropna().astype(str)) if column in d3.columns else set()
        return next(iter(values)) if len(values) == 1 else ""

    return {
        "contract": "PROJECT7_STEP2_V60_RUN_INDEX_D2_PLUS_TARGETED_D3_V1",
        "d3_data_contract": V60_D3_DATA_CONTRACT,
        "v60_control_basis_sha256": unique_value("v60_control_basis_sha256"),
        "v60_design_contract_sha256": unique_value("v60_design_contract_sha256"),
        "rows": int(len(frame)),
        "d2_rows": int((frame["source_kind"].astype(str) == "D2").sum()),
        "d3_rows": int(len(d3)),
        "d3_roles": sorted(d3["data_role"].astype(str).unique().tolist()),
        "legacy_dense_d3_present": bool(
            d3["data_role"].astype(str).isin(
                ["D3_MULTI_ACTUATOR_ROLLOUT", "D3_MULTI_ACTUATOR_SEQUENCE"]
            ).any()
        ),
        "rainfall_groups": int(frame["rainfall_group"].nunique()),
        "scientific_splits": sorted(frame["scientific_split"].astype(str).unique().tolist()),
        "development_folds": sorted(frame["development_fold"].astype(str).unique().tolist()),
    }


__all__ = ["build_step2_v60_run_index", "v60_run_index_summary"]
