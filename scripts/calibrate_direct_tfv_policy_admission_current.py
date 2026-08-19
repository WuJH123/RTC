"""Calibrate Direct-TFV V7 admission on authoritative V6 raw optimizer queries.

This is a Development-only refinement of an already-valid fresh-D3 V1 admission artifact.  The
policy panel must contain one or more authoritative SWMM-labelled V6 raw optimizer plans for every
fresh admission rainfall group.  T10/T20 and untouched evaluation events are not accepted here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rtc.direct_tfv_admission import DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT
from rtc.direct_tfv_policy_admission import (
    DIRECT_TFV_POLICY_PANEL_CONTRACT,
    DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT,
    derive_policy_matched_admission,
)
from rtc.step2_train_response_v60 import V60TrainCache


CURRENT_POLICY_ADMISSION_RUN_CONTRACT = (
    "PROJECT7_CURRENT_DIRECT_TFV_POLICY_MATCHED_ADMISSION_CALIBRATION_V1"
)
POLICY_CANDIDATE_ROLE = "D3_V6_POLICY_CALIBRATION_CANDIDATE"


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _true_delta(entry, candidate_index: int) -> float:
    arrays = entry.arrays
    reference = int(entry.reference_index)
    ref = float(np.asarray(arrays["exact_node_flood_volume_m3"][reference], dtype=np.float64).sum())
    cand = float(np.asarray(arrays["exact_node_flood_volume_m3"][candidate_index], dtype=np.float64).sum())
    return cand - ref


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-admission", required=True)
    p.add_argument("--policy-cache-manifest", required=True)
    p.add_argument("--policy-design-manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--coverage", type=float, default=0.90)
    args = p.parse_args()

    base = json.loads(Path(args.base_admission).read_text(encoding="utf-8"))
    if not isinstance(base, dict) or str(base.get("contract", "")) != DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT:
        raise ValueError("policy calibration requires the accepted V1 admission artifact")
    partition = base.get("partition")
    if not isinstance(partition, dict) or partition.get("ready_for_admission_calibration") is not True:
        raise ValueError("base admission artifact lacks a ready fresh-calibration partition")
    expected_groups = [str(x) for x in partition.get("fresh_calibration_rainfall_groups", ())]
    if len(set(expected_groups)) < 9:
        raise ValueError("base admission partition has insufficient fresh rainfall groups")

    design = pd.read_csv(args.policy_design_manifest)
    required = {
        "rainfall_group",
        "event_id",
        "checkpoint_id",
        "data_role",
        "sequence_sha256",
        "predicted_delta_tfv_m3",
        "active_facility_count",
        "policy_panel_contract",
        "policy_query_step3_contract",
    }
    missing = sorted(required - set(design.columns))
    if missing:
        raise ValueError(f"policy design manifest missing columns: {missing}")
    candidates = design[design["data_role"].astype(str) == POLICY_CANDIDATE_ROLE].copy()
    if candidates.empty:
        raise ValueError("policy design manifest contains no V6 policy calibration candidates")
    if set(candidates["policy_panel_contract"].astype(str)) != {DIRECT_TFV_POLICY_PANEL_CONTRACT}:
        raise ValueError("policy design manifest has the wrong panel contract")
    if set(candidates["policy_query_step3_contract"].astype(str)) != {DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT}:
        raise ValueError("policy design manifest was not produced by the current V6 raw optimizer")
    if {str(x) for x in candidates["rainfall_group"]} != set(expected_groups):
        raise ValueError("policy design rainfall groups differ from the fresh admission calibration role")
    forbidden = ("T10_D180", "T20_D300", "validation", "final", "formal", "policy_lock", "policylock")
    for value in candidates["event_id"].astype(str):
        lowered = value.lower()
        if any(token.lower() in lowered for token in forbidden):
            raise ValueError(f"policy calibration uses reserved/untouched event: {value}")

    by_sha = {
        str(row.sequence_sha256): row
        for row in candidates.itertuples(index=False)
    }
    if len(by_sha) != len(candidates):
        raise ValueError("policy design manifest repeats a candidate sequence SHA")

    cache = V60TrainCache(args.policy_cache_manifest)
    records: list[dict[str, object]] = []
    seen_groups: set[str] = set()
    for name in cache.targeted_d3_names():
        entry = cache.entry(name)
        group = str(entry.rainfall_group)
        if group not in set(expected_groups):
            raise ValueError(f"policy cache contains non-calibration rainfall group: {group}")
        candidate_indices = [int(i) for i in entry.indices if int(i) != int(entry.reference_index)]
        if len(candidate_indices) != 1:
            raise ValueError(f"{name}: policy calibration cache requires exactly one candidate")
        index = candidate_indices[0]
        arrays = entry.arrays
        sha_values = np.asarray(arrays["action_or_sequence_sha256"]).astype(str)
        sequence_sha = str(sha_values[index])
        row = by_sha.get(sequence_sha)
        if row is None:
            raise ValueError(f"{name}: policy cache candidate SHA missing from design manifest")
        if str(row.rainfall_group) != group or str(row.event_id) != str(entry.event_id):
            raise ValueError(f"{name}: policy design/cache identity mismatch")
        records.append(
            {
                "rainfall_group": group,
                "event_id": str(entry.event_id),
                "checkpoint_id": str(entry.checkpoint_id),
                "plan_sha256": sequence_sha,
                "predicted_delta_tfv_m3": float(row.predicted_delta_tfv_m3),
                "true_delta_tfv_m3": _true_delta(entry, index),
                "active_facility_count": int(row.active_facility_count),
            }
        )
        seen_groups.add(group)

    if seen_groups != set(expected_groups):
        raise ValueError(
            "policy cache does not cover every fresh calibration rainfall group; "
            f"missing={sorted(set(expected_groups) - seen_groups)}"
        )

    calibrated = derive_policy_matched_admission(
        base_admission=base,
        panel_contract=DIRECT_TFV_POLICY_PANEL_CONTRACT,
        panel_step3_contract=DIRECT_TFV_POLICY_QUERY_STEP3_CONTRACT,
        panel_records=records,
        expected_rainfall_groups=expected_groups,
        coverage=float(args.coverage),
    )
    payload = {
        **calibrated,
        "run_contract": CURRENT_POLICY_ADMISSION_RUN_CONTRACT,
        "base_admission_contract": str(base["contract"]),
        "base_admission_sha256": _sha(args.base_admission),
        "policy_panel_contract": DIRECT_TFV_POLICY_PANEL_CONTRACT,
        "policy_panel_records": records,
        "lineage": {
            "base_admission_sha256": _sha(args.base_admission),
            "policy_cache_sha256": _sha(args.policy_cache_manifest),
            "policy_design_manifest_sha256": _sha(args.policy_design_manifest),
            "step2_checkpoint_sha256": str(base.get("lineage", {}).get("step2_checkpoint_sha256", "")),
            "sequence_support_sha256": str(base.get("lineage", {}).get("sequence_support_sha256", "")),
        },
        "online_swmm_called": False,
        "coverage_claim_scope": (
            "90% one-sided finite-sample claim applies to rainfall-group maxima from the current "
            "V6 raw optimizer calibration panel; legacy pre-V6 optimizer replay is diagnostic only"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
