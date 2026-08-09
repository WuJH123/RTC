from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import SafetyCalibration
from .contracts import load_priority_nodes
from .safety_audit import audit_selected_action_safety


def _bool_array(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }
    values: list[bool] = []
    for value in series:
        key = str(value).strip().lower()
        if key not in mapping:
            raise ValueError(f"cannot parse boolean audit value: {value!r}")
        values.append(mapping[key])
    return np.asarray(values, dtype=bool)


def _matrix(frame: pd.DataFrame, prefix: str, nodes: tuple[str, ...]) -> np.ndarray:
    columns = [f"{prefix}:{node}" for node in nodes]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"audit cases missing {prefix} columns: {missing}")
    return frame[columns].to_numpy(dtype=float)


def run_formal_safety_audit(
    *,
    cases_path: str | Path,
    priority_path: str | Path,
    calibration_path: str | Path,
    budget_config_path: str | Path,
    output_path: str | Path,
) -> dict[str, object]:
    frame = pd.read_csv(cases_path)
    required = {"event_id", "rainfall_group", "scientific_split", "admitted", "fallback_used"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"formal safety-audit cases missing columns: {missing}")
    if not (frame["scientific_split"].astype(str) == "safety_audit").all():
        raise ValueError("formal safety audit received non-safety_audit rows")
    nodes = load_priority_nodes(priority_path)
    calibration = SafetyCalibration.from_json(calibration_path)
    if calibration.priority_nodes != nodes:
        raise ValueError("calibration priority-node order differs from audit contract")
    audit_groups = set(frame["rainfall_group"].astype(str))
    overlap = sorted(audit_groups & set(calibration.calibration_rainfall_groups))
    if overlap:
        raise ValueError(f"calibration/audit rainfall-group leakage: {overlap[:20]}")
    if not audit_groups:
        raise ValueError("formal safety audit has no rainfall groups")

    cfg = json.loads(Path(budget_config_path).read_text(encoding="utf-8"))
    flood_raw = cfg.get("flood_budget_m3")
    depth_raw = cfg.get("depth_budget_m")
    if not isinstance(flood_raw, dict) or not isinstance(depth_raw, dict):
        raise ValueError("budget config requires flood_budget_m3 and depth_budget_m objects")
    flood_budget = np.asarray([float(flood_raw[node]) for node in nodes], dtype=float)
    depth_budget = np.asarray([float(depth_raw[node]) for node in nodes], dtype=float)
    report = audit_selected_action_safety(
        event_ids=frame["event_id"].astype(str).to_numpy(),
        admitted=_bool_array(frame["admitted"]),
        fallback_used=_bool_array(frame["fallback_used"]),
        predicted_flood_ucb_m3=_matrix(frame, "pred_flood_ucb_m3", nodes),
        true_flood_deterioration_m3=_matrix(frame, "true_flood_delta_m3", nodes),
        flood_budget_m3=flood_budget,
        predicted_depth_ucb_m=_matrix(frame, "pred_depth_ucb_m", nodes),
        true_depth_deterioration_m=_matrix(frame, "true_depth_delta_m", nodes),
        depth_budget_m=depth_budget,
        maximum_false_safe_rate=float(cfg.get("maximum_false_safe_rate", 0.01)),
        minimum_sitewise_coverage=float(cfg.get("minimum_sitewise_coverage", 0.95)),
    )
    payload = {
        "contract": "INDEPENDENT_SITEWISE_SELECTED_ACTION_SAFETY_AUDIT_V2",
        "passed": report.passed,
        "decisions": report.decisions,
        "admitted_decisions": report.admitted_decisions,
        "fallback_decisions": report.fallback_decisions,
        "false_safe_decisions": report.false_safe_decisions,
        "false_safe_rate_given_admitted": report.false_safe_rate_given_admitted,
        "event_balanced_false_safe_rate": report.event_balanced_false_safe_rate,
        "sitewise_flood_coverage": list(report.sitewise_flood_coverage),
        "sitewise_depth_coverage": list(report.sitewise_depth_coverage),
        "calibration_rainfall_groups": list(calibration.calibration_rainfall_groups),
        "audit_rainfall_groups": sorted(audit_groups),
        "rainfall_group_overlap": overlap,
        "cases_path": str(Path(cases_path)),
        "calibration_path": str(Path(calibration_path)),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Independent rainfall-group-disjoint safety audit gate")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--budget-config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = run_formal_safety_audit(
        cases_path=args.cases,
        priority_path=args.priority,
        calibration_path=args.calibration,
        budget_config_path=args.budget_config,
        output_path=args.out,
    )
    print(json.dumps(payload, indent=2))
    if payload["passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
