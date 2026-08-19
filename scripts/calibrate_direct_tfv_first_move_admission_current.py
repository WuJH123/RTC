"""Calibrate optimizer-matched admission directly from exact first-move SWMM branches.

The calibration does not rebuild or read a V60/generic-D3 cache. Identity is keyed by
(rainfall_group, sequence_sha256), because HOLD sequences may be byte-identical across rainfall
groups while their hydraulic truth is not.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

from rtc.checkpoint_direct_tfv import (
    direct_tfv_first_move_behavioral_source_sha256,
    direct_tfv_first_move_source_sha256,
)
from rtc.direct_tfv_first_move_admission import (
    DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT,
    DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT,
    derive_first_move_admission,
)


CURRENT_FIRST_MOVE_ADMISSION_RUN_CONTRACT = (
    "PROJECT7_CURRENT_DIRECT_TFV_REFINED_FIRST_MOVE_ADMISSION_CALIBRATION_V2_DIRECT_SWMM"
)
HOLD_ROLE = "D3_V60_HOLD_REFERENCE"
FIRST_MOVE_CANDIDATE_ROLE = "D3_V9_REFINED_FIRST_MOVE_CALIBRATION_CANDIDATE"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tfv(node_statistics_path: Path) -> float:
    total = 0.0
    with gzip.open(node_statistics_path, "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            total += float(row["delta_flooding_volume_m3"])
    return float(total)


def _truth_index(run_dir: Path) -> dict[tuple[str, str], dict[str, object]]:
    summary = run_dir / "D3_RUN_SUMMARY.csv"
    if not summary.is_file():
        raise FileNotFoundError(f"first-move run directory lacks D3_RUN_SUMMARY.csv: {run_dir}")
    frame = pd.read_csv(summary)
    required = {"rainfall_group", "sequence_sha256", "metadata_path", "status"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"D3 run summary missing columns: {missing}")
    result: dict[tuple[str, str], dict[str, object]] = {}
    for row in frame.itertuples(index=False):
        group = str(row.rainfall_group)
        sequence_sha = str(row.sequence_sha256)
        key = (group, sequence_sha)
        metadata_path = Path(str(row.metadata_path)).resolve()
        if not metadata_path.is_file():
            raise FileNotFoundError(f"first-move branch metadata missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(metadata.get("sequence_sha256", "")) != sequence_sha:
            raise ValueError("first-move run summary/metadata sequence SHA mismatch")
        verification = metadata.get("same_prefix_verification")
        if not isinstance(verification, dict) or verification.get("passed") is not True:
            raise ValueError("first-move SWMM branch lacks exact same-prefix verification")
        stats = metadata_path.parent / str(metadata.get("node_statistics_file", ""))
        if not stats.is_file():
            raise FileNotFoundError(f"first-move node statistics missing: {stats}")
        value = {
            "metadata_path": str(metadata_path),
            "metadata_sha256": _sha(metadata_path),
            "node_statistics_path": str(stats),
            "node_statistics_sha256": _sha(stats),
            "tfv_m3": _tfv(stats),
            "flow_routing_error_pct": float(metadata.get("flow_routing_error_pct", float("nan"))),
        }
        previous = result.get(key)
        if previous is not None and previous != value:
            raise ValueError(
                "multiple non-identical authoritative branches share "
                f"(rainfall_group, sequence_sha256)={key}"
            )
        result[key] = value
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first-move-run-dir", required=True)
    p.add_argument("--first-move-design-manifest", required=True)
    p.add_argument("--coverage", type=float, default=0.90)
    p.add_argument("--step2-checkpoint", required=True)
    p.add_argument("--sequence-support", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    design = pd.read_csv(args.first_move_design_manifest)
    required = {
        "rainfall_group", "event_id", "checkpoint_id", "data_role", "sequence_sha256",
        "predicted_refined_delta_tfv_m3", "first_move_changed_facility_count",
        "first_move_panel_contract", "first_move_query_step3_contract",
        "first_move_source_sha256", "first_move_behavioral_source_sha256",
        "candidate_rows_used", "generic_d3_candidate_dependency",
    }
    if missing := sorted(required - set(design.columns)):
        raise ValueError(f"first-move design manifest missing columns: {missing}")
    if design["candidate_rows_used"].astype(bool).any() or design[
        "generic_d3_candidate_dependency"
    ].astype(bool).any():
        raise ValueError("current first-move calibration refuses generic D3 candidate dependency")
    if set(design["first_move_panel_contract"].astype(str)) != {DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT}:
        raise ValueError("first-move design has the wrong panel contract")
    if set(design["first_move_query_step3_contract"].astype(str)) != {
        DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT
    }:
        raise ValueError("first-move design was generated by a different optimizer contract")
    current_full = direct_tfv_first_move_source_sha256()
    current_behavior = direct_tfv_first_move_behavioral_source_sha256()
    if {str(x).lower() for x in design["first_move_behavioral_source_sha256"]} != {
        current_behavior.lower()
    }:
        raise ValueError("first-move design behavioral source differs from calibration runtime")

    forbidden = (
        "T5", "T10", "T20", "P15", "P35", "P75", "V10_RP12", "V10_RP40",
        "V10_RP90", "validation", "final", "formal", "policy_lock", "policylock",
    )
    for event in design["event_id"].astype(str):
        if any(token.lower() in event.lower() for token in forbidden):
            raise ValueError(f"first-move calibration uses observed/reserved event: {event}")

    truth = _truth_index(Path(args.first_move_run_dir))
    groups = sorted(set(design["rainfall_group"].astype(str)))
    records: list[dict[str, object]] = []
    for group in groups:
        rows = design[design["rainfall_group"].astype(str) == group]
        hold = rows[rows["data_role"].astype(str) == HOLD_ROLE]
        candidate = rows[rows["data_role"].astype(str) == FIRST_MOVE_CANDIDATE_ROLE]
        if len(hold) != 1 or len(candidate) != 1:
            raise ValueError(f"{group}: expected exactly one HOLD and one candidate")
        hold_row = hold.iloc[0]
        cand_row = candidate.iloc[0]
        hold_key = (group, str(hold_row["sequence_sha256"]))
        cand_key = (group, str(cand_row["sequence_sha256"]))
        if hold_key not in truth or cand_key not in truth:
            raise ValueError(f"{group}: authoritative SWMM truth missing HOLD/candidate branch")
        true_delta = float(truth[cand_key]["tfv_m3"]) - float(truth[hold_key]["tfv_m3"])
        records.append(
            {
                "rainfall_group": group,
                "event_id": str(cand_row["event_id"]),
                "checkpoint_id": str(cand_row["checkpoint_id"]),
                "plan_sha256": str(cand_row["sequence_sha256"]),
                "predicted_refined_delta_tfv_m3": float(cand_row["predicted_refined_delta_tfv_m3"]),
                "true_refined_delta_tfv_m3": true_delta,
                "first_move_changed_facility_count": int(cand_row["first_move_changed_facility_count"]),
                "hold_tfv_m3": float(truth[hold_key]["tfv_m3"]),
                "candidate_tfv_m3": float(truth[cand_key]["tfv_m3"]),
                "hold_metadata_sha256": truth[hold_key]["metadata_sha256"],
                "candidate_metadata_sha256": truth[cand_key]["metadata_sha256"],
            }
        )

    calibrated = derive_first_move_admission(
        panel_contract=DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT,
        panel_step3_contract=DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT,
        panel_records=records,
        expected_rainfall_groups=groups,
        coverage=float(args.coverage),
    )
    payload = {
        **calibrated,
        "run_contract": CURRENT_FIRST_MOVE_ADMISSION_RUN_CONTRACT,
        "panel_records": records,
        "generic_d3_labels_used": False,
        "lineage": {
            "first_move_source_sha256": current_full,
            "first_move_behavioral_source_sha256": current_behavior,
            "first_move_run_summary_sha256": _sha(
                Path(args.first_move_run_dir) / "D3_RUN_SUMMARY.csv"
            ),
            "first_move_design_manifest_sha256": _sha(args.first_move_design_manifest),
            "step2_checkpoint_sha256": _sha(args.step2_checkpoint),
            "sequence_support_sha256": _sha(args.sequence_support),
        },
        "coverage_claim_scope": (
            "90% rainfall-group split-conformal evidence for exact candidate-free, same-prefix, "
            "target-latched refined first-move SWMM branches."
        ),
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
