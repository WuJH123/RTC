"""Calibrate V8 executable-prefix admission from authoritative Development SWMM labels."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rtc.direct_tfv_policy_admission import DIRECT_TFV_POLICY_ADMISSION_CONTRACT
from rtc.direct_tfv_receding_prefix import (
    DIRECT_TFV_RECEDING_PREFIX_PANEL_CONTRACT,
    DIRECT_TFV_RECEDING_PREFIX_QUERY_STEP3_CONTRACT,
    derive_receding_prefix_admission,
)
from rtc.step2_train_response_v60 import V60TrainCache


CURRENT_RECEDING_PREFIX_CALIBRATION_RUN_CONTRACT = (
    "PROJECT7_CURRENT_DIRECT_TFV_RECEDING_PREFIX_ADMISSION_CALIBRATION_V1"
)
PREFIX_CANDIDATE_ROLE = "D3_V8_RECEDING_PREFIX_CALIBRATION_CANDIDATE"


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
    p.add_argument("--base-policy-admission", required=True)
    p.add_argument("--prefix-cache-manifest", required=True)
    p.add_argument("--prefix-design-manifest", required=True)
    p.add_argument("--coverage", type=float, default=0.90)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    base = json.loads(Path(args.base_policy_admission).read_text(encoding="utf-8"))
    if not isinstance(base, dict) or str(base.get("contract", "")) != DIRECT_TFV_POLICY_ADMISSION_CONTRACT:
        raise ValueError("receding-prefix calibration requires the accepted V2 policy admission")
    expected_groups = [str(x) for x in base.get("policy_calibration_rainfall_groups", ())]
    if len(set(expected_groups)) < 9:
        raise ValueError("V2 policy admission has insufficient rainfall groups for prefix calibration")

    design = pd.read_csv(args.prefix_design_manifest)
    required = {
        "rainfall_group",
        "event_id",
        "checkpoint_id",
        "data_role",
        "sequence_sha256",
        "predicted_prefix_delta_tfv_m3",
        "active_facility_count",
        "receding_prefix_panel_contract",
        "receding_prefix_query_step3_contract",
    }
    missing = sorted(required - set(design.columns))
    if missing:
        raise ValueError(f"receding-prefix design manifest missing columns: {missing}")
    candidates = design[design["data_role"].astype(str) == PREFIX_CANDIDATE_ROLE].copy()
    if candidates.empty:
        raise ValueError("receding-prefix design contains no prefix calibration candidates")
    if set(candidates["receding_prefix_panel_contract"].astype(str)) != {
        DIRECT_TFV_RECEDING_PREFIX_PANEL_CONTRACT
    }:
        raise ValueError("receding-prefix design has the wrong panel contract")
    if set(candidates["receding_prefix_query_step3_contract"].astype(str)) != {
        DIRECT_TFV_RECEDING_PREFIX_QUERY_STEP3_CONTRACT
    }:
        raise ValueError("receding-prefix design was not generated from the frozen raw optimizer")
    if {str(x) for x in candidates["rainfall_group"]} != set(expected_groups):
        raise ValueError("receding-prefix design rainfall groups differ from V2 policy calibration")
    forbidden = (
        "T10_D180",
        "T20_D300",
        "P15_D150",
        "P35_D270",
        "P75_D90",
        "validation",
        "final",
        "formal",
        "policy_lock",
        "policylock",
    )
    for value in candidates["event_id"].astype(str):
        lowered = value.lower()
        if any(token.lower() in lowered for token in forbidden):
            raise ValueError(f"receding-prefix calibration uses observed/reserved event: {value}")

    by_sha = {str(row.sequence_sha256): row for row in candidates.itertuples(index=False)}
    if len(by_sha) != len(candidates):
        raise ValueError("receding-prefix design repeats a candidate sequence SHA")

    cache = V60TrainCache(args.prefix_cache_manifest)
    records: list[dict[str, object]] = []
    seen_groups: set[str] = set()
    for name in cache.names("D3"):
        entry = cache.entry(name)
        roles = {str(role).strip().upper() for role in entry.candidate_roles}
        if roles != {PREFIX_CANDIDATE_ROLE}:
            raise ValueError(f"{name}: receding-prefix cache has wrong candidate roles: {sorted(roles)}")
        group = str(entry.rainfall_group)
        if group not in set(expected_groups):
            raise ValueError(f"receding-prefix cache contains non-calibration rainfall group: {group}")
        candidate_indices = [int(i) for i in entry.indices if int(i) != int(entry.reference_index)]
        if len(candidate_indices) != 1:
            raise ValueError(f"{name}: receding-prefix cache requires exactly one candidate")
        index = candidate_indices[0]
        arrays = entry.arrays
        sha_values = np.asarray(arrays["action_or_sequence_sha256"]).astype(str)
        sequence_sha = str(sha_values[index])
        row = by_sha.get(sequence_sha)
        if row is None:
            raise ValueError(f"{name}: receding-prefix candidate SHA missing from design manifest")
        if str(row.rainfall_group) != group or str(row.event_id) != str(entry.event_id):
            raise ValueError(f"{name}: receding-prefix design/cache identity mismatch")
        records.append(
            {
                "rainfall_group": group,
                "event_id": str(entry.event_id),
                "checkpoint_id": str(entry.checkpoint_id),
                "plan_sha256": sequence_sha,
                "predicted_prefix_delta_tfv_m3": float(row.predicted_prefix_delta_tfv_m3),
                "true_prefix_delta_tfv_m3": _true_delta(entry, index),
                "active_facility_count": int(row.active_facility_count),
            }
        )
        seen_groups.add(group)

    if seen_groups != set(expected_groups):
        raise ValueError(
            "receding-prefix cache does not cover every policy-calibration rainfall group; "
            f"missing={sorted(set(expected_groups) - seen_groups)}"
        )

    calibrated = derive_receding_prefix_admission(
        base_policy_admission=base,
        panel_contract=DIRECT_TFV_RECEDING_PREFIX_PANEL_CONTRACT,
        panel_step3_contract=DIRECT_TFV_RECEDING_PREFIX_QUERY_STEP3_CONTRACT,
        panel_records=records,
        expected_rainfall_groups=expected_groups,
        coverage=float(args.coverage),
    )
    payload = {
        **calibrated,
        "run_contract": CURRENT_RECEDING_PREFIX_CALIBRATION_RUN_CONTRACT,
        "base_policy_admission_sha256": _sha(args.base_policy_admission),
        "prefix_panel_records": records,
        "lineage": {
            "base_policy_admission_sha256": _sha(args.base_policy_admission),
            "prefix_cache_sha256": _sha(args.prefix_cache_manifest),
            "prefix_design_manifest_sha256": _sha(args.prefix_design_manifest),
            "step2_checkpoint_sha256": str(base.get("lineage", {}).get("step2_checkpoint_sha256", "")),
            "sequence_support_sha256": str(base.get("lineage", {}).get("sequence_support_sha256", "")),
        },
        "online_swmm_called": False,
        "coverage_claim_scope": (
            "90% one-sided finite-sample claim applies to rainfall-group residual maxima for the "
            "EXECUTE_H10_THEN_HOLD_H350 estimand on the frozen V6 raw optimizer first move"
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
