"""Calibrate V12 scenario-mean first-move admission from exact same-prefix SWMM branches."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import pandas as pd

from rtc.direct_tfv_first_move_admission import (
    DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT,
    derive_first_move_admission,
)
from rtc.direct_tfv_v12_lineage import direct_tfv_v12_behavioral_sha256
from rtc.step3_tfv_value_mpc_v10 import (
    DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT,
    DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT,
)


HOLD_ROLE = "D3_V60_HOLD_REFERENCE"
CANDIDATE_ROLE = "D3_V12_SCENARIO_MEAN_REFINED_FIRST_MOVE_CALIBRATION_CANDIDATE"
RUN_CONTRACT = "PROJECT7_CURRENT_DIRECT_TFV_V12_SCENARIO_MEAN_FIRST_MOVE_ADMISSION_V1"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tfv(path: Path) -> float:
    total = 0.0
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            total += float(row["delta_flooding_volume_m3"])
    return float(total)


def _truth(run_dir: Path) -> dict[tuple[str, str], dict[str, object]]:
    summary = run_dir / "D3_RUN_SUMMARY.csv"
    frame = pd.read_csv(summary)
    required = {"rainfall_group", "sequence_sha256", "metadata_path"}
    if missing := sorted(required - set(frame.columns)):
        raise ValueError(f"V12 SWMM summary missing columns: {missing}")
    result: dict[tuple[str, str], dict[str, object]] = {}
    for row in frame.itertuples(index=False):
        key = (str(row.rainfall_group), str(row.sequence_sha256))
        meta_path = Path(str(row.metadata_path)).resolve()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if str(meta.get("sequence_sha256", "")) != key[1]:
            raise ValueError("V12 SWMM summary/metadata sequence mismatch")
        verification = meta.get("same_prefix_verification")
        if not isinstance(verification, dict) or verification.get("passed") is not True:
            raise ValueError("V12 SWMM branch lacks exact same-prefix verification")
        stats = meta_path.parent / str(meta.get("node_statistics_file", ""))
        value = {
            "tfv_m3": _tfv(stats),
            "metadata_sha256": _sha(meta_path),
            "node_statistics_sha256": _sha(stats),
        }
        previous = result.get(key)
        if previous is not None and previous != value:
            raise ValueError(f"duplicate non-identical V12 SWMM truth for {key}")
        result[key] = value
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first-move-run-dir", required=True)
    p.add_argument("--first-move-design-manifest", required=True)
    p.add_argument("--step2-checkpoint", required=True)
    p.add_argument("--sequence-support", required=True)
    p.add_argument("--coverage", type=float, default=0.90)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    design = pd.read_csv(args.first_move_design_manifest)
    required = {
        "rainfall_group", "event_id", "checkpoint_id", "data_role", "sequence_sha256",
        "predicted_refined_delta_tfv_m3", "first_move_changed_facility_count",
        "first_move_panel_contract", "first_move_query_step3_contract",
        "rainfall_scenario_contract", "v12_behavioral_source_sha256",
        "candidate_rows_used", "generic_d3_candidate_dependency",
    }
    if missing := sorted(required - set(design.columns)):
        raise ValueError(f"V12 first-move design missing columns: {missing}")
    bool_true = {"true", "1", "yes"}
    if any(str(value).strip().lower() in bool_true for value in design["candidate_rows_used"]):
        raise ValueError("V12 admission refuses contexts built from candidate rows")
    if any(
        str(value).strip().lower() in bool_true
        for value in design["generic_d3_candidate_dependency"]
    ):
        raise ValueError("V12 admission refuses generic D3 candidate dependency")
    if set(design["first_move_panel_contract"].astype(str)) != {DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT}:
        raise ValueError("V12 panel contract mismatch")
    if set(design["first_move_query_step3_contract"].astype(str)) != {
        DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT
    }:
        raise ValueError("V12 panel has the wrong scenario-mean query contract")
    if set(design["rainfall_scenario_contract"].astype(str)) != {
        DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT
    }:
        raise ValueError("V12 panel has the wrong rainfall scenario contract")
    current_behavior = direct_tfv_v12_behavioral_sha256()
    if {str(x).lower() for x in design["v12_behavioral_source_sha256"]} != {
        current_behavior.lower()
    }:
        raise ValueError("V12 panel behavioral fingerprint differs from calibrator")
    truth = _truth(Path(args.first_move_run_dir))
    groups = sorted(set(design["rainfall_group"].astype(str)))
    if len(groups) < 24:
        raise ValueError("V12 admission requires >=24 rainfall groups")
    records = []
    for group in groups:
        rows = design[design["rainfall_group"].astype(str) == group]
        hold = rows[rows["data_role"].astype(str) == HOLD_ROLE]
        candidate = rows[rows["data_role"].astype(str) == CANDIDATE_ROLE]
        if len(hold) != 1 or len(candidate) != 1:
            raise ValueError(f"{group}: V12 panel requires exactly HOLD+candidate")
        h = hold.iloc[0]; c = candidate.iloc[0]
        hkey = (group, str(h["sequence_sha256"])); ckey = (group, str(c["sequence_sha256"]))
        if hkey not in truth or ckey not in truth:
            raise ValueError(f"{group}: V12 SWMM truth missing HOLD/candidate")
        records.append(
            {
                "rainfall_group": group,
                "event_id": str(c["event_id"]),
                "checkpoint_id": str(c["checkpoint_id"]),
                "plan_sha256": str(c["sequence_sha256"]),
                "predicted_refined_delta_tfv_m3": float(c["predicted_refined_delta_tfv_m3"]),
                "true_refined_delta_tfv_m3": float(truth[ckey]["tfv_m3"]) - float(truth[hkey]["tfv_m3"]),
                "first_move_changed_facility_count": int(c["first_move_changed_facility_count"]),
            }
        )
    payload = derive_first_move_admission(
        panel_contract=DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT,
        panel_step3_contract=DIRECT_TFV_SCENARIO_MEAN_STEP3_CONTRACT,
        panel_records=records,
        expected_rainfall_groups=groups,
        coverage=float(args.coverage),
        rainfall_scenario_contract=DIRECT_TFV_CAUSAL_RAINFALL_SCENARIO_CONTRACT,
    )
    payload.update(
        {
            "run_contract": RUN_CONTRACT,
            "generic_d3_labels_used": False,
            "v12_behavioral_source_sha256": current_behavior,
            "lineage": {
                "v12_behavioral_source_sha256": current_behavior,
                "step2_checkpoint_sha256": _sha(args.step2_checkpoint),
                "sequence_support_sha256": _sha(args.sequence_support),
                "first_move_design_manifest_sha256": _sha(args.first_move_design_manifest),
                "first_move_run_summary_sha256": _sha(
                    Path(args.first_move_run_dir) / "D3_RUN_SUMMARY.csv"
                ),
            },
        }
    )
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
