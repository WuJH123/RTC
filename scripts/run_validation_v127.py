"""Run the frozen V127 model once across a predeclared Validation event manifest.

Validation is evaluation only: this script never trains, selects thresholds, changes the model,
or accesses Final/Formal. Events run serially to keep a 16-GB workstation below RAM pressure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

V127_VALIDATION_CONTRACT = "PROJECT7_V127_READ_ONCE_VALIDATION_V1_FROZEN_MODEL"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"event_id", "inp_path", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V127 Validation manifest lacks columns: {missing}")
    if frame.empty:
        raise ValueError("V127 Validation manifest is empty")
    if set(frame["split"].astype(str).str.lower()) != {"validation"}:
        raise ValueError("V127 Validation manifest must contain only split=validation")
    if frame["event_id"].astype(str).duplicated().any():
        raise ValueError("V127 Validation manifest duplicates event_id")
    if "rainfall_group" in frame.columns and frame["rainfall_group"].astype(str).duplicated().any():
        raise ValueError("V127 Validation requires unique predeclared rainfall groups")
    for raw in frame["inp_path"].astype(str):
        if not Path(raw).is_file():
            raise FileNotFoundError(raw)
    return frame


def _mean(values: pd.Series) -> float:
    array = values.to_numpy(dtype=float)
    array = array[np.isfinite(array)]
    return float(array.mean()) if array.size else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation-manifest", required=True)
    p.add_argument("--sensors", required=True)
    p.add_argument("--priority-nodes", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--native-controls-template", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--step1", required=True)
    p.add_argument("--step2", required=True)
    p.add_argument("--continuous-gate", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--lbfgsb-maxiter", type=int, default=30)
    p.add_argument("--optimizer-deadline-seconds", type=float, default=480.0)
    p.add_argument("--decision-runtime-budget-seconds", type=float, default=540.0)
    p.add_argument("--pfv-soft-margin-m3", type=float, default=100.0)
    p.add_argument("--pfv-penalty-weight", type=float, default=1.0)
    args = p.parse_args()

    manifest_path = Path(args.validation_manifest).resolve()
    events = _load_manifest(manifest_path)
    common_paths = [
        Path(args.sensors),
        Path(args.priority_nodes),
        Path(args.config),
        Path(args.native_controls_template),
        Path(args.graph),
        Path(args.step1),
        Path(args.step2),
        Path(args.continuous_gate),
    ]
    for path in common_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).resolve().parent / "run_seven_strategies_v127.py"

    all_rows: list[dict[str, object]] = []
    event_reports: list[dict[str, object]] = []
    for _, event in events.iterrows():
        event_id = str(event["event_id"])
        event_out = root / event_id
        event_out.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(runner),
            "--inp",
            str(Path(str(event["inp_path"])).resolve()),
            "--event-id",
            event_id,
            "--sensors",
            str(Path(args.sensors).resolve()),
            "--priority-nodes",
            str(Path(args.priority_nodes).resolve()),
            "--config",
            str(Path(args.config).resolve()),
            "--native-controls-template",
            str(Path(args.native_controls_template).resolve()),
            "--graph",
            str(Path(args.graph).resolve()),
            "--step1",
            str(Path(args.step1).resolve()),
            "--step2",
            str(Path(args.step2).resolve()),
            "--continuous-gate",
            str(Path(args.continuous_gate).resolve()),
            "--out-dir",
            str(event_out),
            "--device",
            str(args.device),
            "--lbfgsb-maxiter",
            str(args.lbfgsb_maxiter),
            "--optimizer-deadline-seconds",
            str(args.optimizer_deadline_seconds),
            "--decision-runtime-budget-seconds",
            str(args.decision_runtime_budget_seconds),
            "--pfv-soft-margin-m3",
            str(args.pfv_soft_margin_m3),
            "--pfv-penalty-weight",
            str(args.pfv_penalty_weight),
        ]
        subprocess.run(command, check=True)
        report_path = event_out / "PROJECT7_V127_SEVEN_STRATEGY_COMPARISON.json"
        if not report_path.is_file():
            raise RuntimeError(f"Validation event {event_id} did not produce seven-strategy report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        rows = report.get("rows")
        if not isinstance(rows, list) or len(rows) != 7:
            raise RuntimeError(f"Validation event {event_id} does not contain seven strategies")
        for row in rows:
            record = dict(row)
            record["validation_event_id"] = event_id
            if "rainfall_group" in events.columns:
                record["rainfall_group"] = str(event["rainfall_group"])
            all_rows.append(record)
        event_reports.append(
            {
                "event_id": event_id,
                "report_path": str(report_path.resolve()),
                "source_inp_sha256": _sha(str(event["inp_path"])),
            }
        )

    detail = pd.DataFrame.from_records(all_rows)
    if detail.empty or detail["validation_event_id"].nunique() != len(events):
        raise RuntimeError("V127 Validation detail is incomplete")
    summary_rows: list[dict[str, object]] = []
    metrics = [
        "tfv_m3",
        "priority8_pfv_m3",
        "global_peak_flood_rate_m3s",
        "flow_routing_error_pct",
        "tfv_reduction_vs_no_control_pct",
        "pfv_change_vs_no_control_pct",
    ]
    for strategy, group in detail.groupby("strategy", sort=True):
        row: dict[str, object] = {
            "strategy": str(strategy),
            "events": int(group["validation_event_id"].nunique()),
        }
        for metric in metrics:
            if metric in group.columns:
                row[f"event_balanced_mean_{metric}"] = _mean(group[metric])
        summary_rows.append(row)
    summary = pd.DataFrame.from_records(summary_rows)

    detail_path = root / "PROJECT7_V127_VALIDATION_EVENT_DETAIL.csv"
    summary_path = root / "PROJECT7_V127_VALIDATION_EVENT_BALANCED.csv"
    detail.to_csv(detail_path, index=False, quoting=csv.QUOTE_MINIMAL)
    summary.to_csv(summary_path, index=False, quoting=csv.QUOTE_MINIMAL)
    payload = {
        "contract": V127_VALIDATION_CONTRACT,
        "validation_events": int(len(events)),
        "validation_manifest": str(manifest_path),
        "validation_manifest_sha256": _sha(manifest_path),
        "step2_sha256": _sha(args.step2),
        "continuous_gate_sha256": _sha(args.continuous_gate),
        "event_reports": event_reports,
        "event_detail_csv": str(detail_path.resolve()),
        "event_balanced_csv": str(summary_path.resolve()),
        "rows": summary_rows,
        "boundary": {
            "model_frozen_before_validation": True,
            "validation_used_for_training": False,
            "validation_used_for_threshold_selection": False,
            "validation_used_for_model_selection": False,
            "final_accessed": False,
            "formal_accessed": False,
            "policy_lock_after_validation": False,
        },
    }
    (root / "PROJECT7_V127_VALIDATION.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
