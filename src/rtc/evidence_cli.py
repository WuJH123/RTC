from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import fit_sitewise_safety_calibration
from .contracts import load_priority_nodes
from .pipeline import PipelineLedger, create_policy_lock, evidence_from_files
from .safety_audit import audit_selected_action_safety


def _matrix(frame: pd.DataFrame, prefix: str, nodes: tuple[str, ...]) -> np.ndarray:
    columns = [f"{prefix}:{node}" for node in nodes]
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise ValueError(f"missing columns for {prefix}: {missing}")
    return frame[columns].to_numpy(dtype=float)


def calibrate_safety_main() -> None:
    parser = argparse.ArgumentParser(description="Fit calibration-only one-sided site-wise safety error bounds")
    parser.add_argument("--input", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--coverage", type=float, default=0.95)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    if "scientific_split" in frame.columns:
        frame = frame[frame["scientific_split"].astype(str) == "calibration"]
    if frame.empty:
        raise ValueError("no calibration rows available")
    if "rainfall_group" not in frame.columns:
        raise ValueError("calibration input requires rainfall_group")
    nodes = load_priority_nodes(args.priority)
    calibration = fit_sitewise_safety_calibration(
        priority_nodes=nodes,
        predicted_flood_deterioration_m3=_matrix(frame, "pred_flood_delta_m3", nodes),
        true_flood_deterioration_m3=_matrix(frame, "true_flood_delta_m3", nodes),
        predicted_depth_deterioration_m=_matrix(frame, "pred_depth_delta_m", nodes),
        true_depth_deterioration_m=_matrix(frame, "true_depth_delta_m", nodes),
        rainfall_groups=frame["rainfall_group"].astype(str).to_numpy(),
        coverage=args.coverage,
    )
    calibration.to_json(args.out)
    print(json.dumps({
        "out": args.out,
        "samples": calibration.calibration_sample_count,
        "rainfall_groups": len(calibration.calibration_rainfall_groups),
        "coverage": calibration.coverage,
    }, indent=2))


def audit_safety_main() -> None:
    parser = argparse.ArgumentParser(description="Independent pre-lock selected-action SWMM safety audit")
    parser.add_argument("--input", required=True)
    parser.add_argument("--priority", required=True)
    parser.add_argument("--budget-config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    if "scientific_split" in frame.columns:
        frame = frame[frame["scientific_split"].astype(str) == "safety_audit"]
    if frame.empty:
        raise ValueError("no safety_audit rows available")
    required = {"event_id", "admitted", "fallback_used"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"safety audit input missing columns: {missing}")
    nodes = load_priority_nodes(args.priority)
    cfg = json.loads(Path(args.budget_config).read_text(encoding="utf-8"))
    flood_cfg = cfg.get("flood_budget_m3", {})
    depth_cfg = cfg.get("depth_budget_m", {})
    flood_budget = np.array([float(flood_cfg[node]) for node in nodes], dtype=float)
    depth_budget = np.array([float(depth_cfg[node]) for node in nodes], dtype=float)
    report = audit_selected_action_safety(
        event_ids=frame["event_id"].astype(str).to_numpy(),
        admitted=frame["admitted"].astype(bool).to_numpy(),
        fallback_used=frame["fallback_used"].astype(bool).to_numpy(),
        predicted_flood_ucb_m3=_matrix(frame, "pred_flood_ucb_m3", nodes),
        true_flood_deterioration_m3=_matrix(frame, "true_flood_delta_m3", nodes),
        flood_budget_m3=flood_budget,
        predicted_depth_ucb_m=_matrix(frame, "pred_depth_ucb_m", nodes),
        true_depth_deterioration_m=_matrix(frame, "true_depth_delta_m", nodes),
        depth_budget_m=depth_budget,
        maximum_false_safe_rate=float(cfg.get("maximum_false_safe_rate", 0.01)),
        minimum_sitewise_coverage=float(cfg.get("minimum_sitewise_coverage", 0.95)),
    )
    report.to_json(args.out)
    print(Path(args.out).read_text(encoding="utf-8"))
    if not report.passed:
        raise SystemExit(2)


def record_stage_main() -> None:
    parser = argparse.ArgumentParser(description="Record hashed pass/fail evidence in the fail-closed pipeline ledger")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--evidence", nargs="+", required=True)
    parser.add_argument("--passed", action="store_true")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    ledger_path = Path(args.ledger)
    ledger = PipelineLedger.from_json(ledger_path) if ledger_path.exists() else PipelineLedger()
    evidence = evidence_from_files(args.stage, args.evidence, passed=args.passed, notes=args.notes)
    ledger.record(evidence)
    ledger.to_json(ledger_path)
    print(json.dumps({"ledger": str(ledger_path), "stage": args.stage, "passed": args.passed}, indent=2))


def policy_lock_main() -> None:
    parser = argparse.ArgumentParser(description="Freeze production policy lineage before untouched Final")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--artifacts", required=True, help="JSON object mapping required artifact names to paths")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    ledger = PipelineLedger.from_json(args.ledger)
    artifacts = json.loads(Path(args.artifacts).read_text(encoding="utf-8"))
    if not isinstance(artifacts, dict):
        raise ValueError("artifacts JSON must be an object mapping names to paths")
    payload = create_policy_lock(ledger=ledger, artefacts=artifacts, output_path=args.out)
    ledger.record(evidence_from_files("policy_lock", [args.out], passed=True, notes=str(payload["policy_sha256"])))
    ledger.to_json(args.ledger)
    print(json.dumps(payload, indent=2))
