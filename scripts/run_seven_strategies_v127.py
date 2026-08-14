"""Run Project7 V127 Proposed plus the six frozen authoritative SWMM baselines."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys

try:
    from run_six_baselines_v122 import BASELINES, _run_one
except ModuleNotFoundError:  # pragma: no cover
    from scripts.run_six_baselines_v122 import BASELINES, _run_one

V127_SEVEN_STRATEGY_CONTRACT = "PROJECT7_V127_SEVEN_STRATEGY_AUTHORITATIVE_SWMM_COMPARISON_V1"


def _priority_nodes(path: Path) -> set[str]:
    nodes = {
        line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if len(nodes) != 8:
        raise ValueError("V127 seven-strategy comparison requires frozen Priority8")
    return nodes


def _node_volumes(path: Path, priority: set[str]) -> tuple[float, float]:
    tfv = pfv = 0.0
    seen: set[str] = set()
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"node_id", "delta_flooding_volume_m3"}.issubset(reader.fieldnames):
            raise ValueError(f"invalid node statistics: {path}")
        for row in reader:
            node = str(row["node_id"])
            if node in seen:
                raise ValueError(f"duplicate node statistics: {node}")
            seen.add(node)
            value = max(float(row["delta_flooding_volume_m3"]), 0.0)
            tfv += value
            if node in priority:
                pfv += value
    return float(tfv), float(pfv)


def _augment(row: dict[str, object], priority: set[str]) -> dict[str, object]:
    stats = Path(str(row["node_statistics_path"]))
    tfv, pfv = _node_volumes(stats, priority)
    row = dict(row)
    row["tfv_m3"] = tfv
    row["priority8_pfv_m3"] = pfv
    return row


def _run_proposed(args, root: Path) -> dict[str, object]:
    out = root / "proposed_v127"
    out.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.event_id}__proposed_v127"
    script = Path(__file__).resolve().parent / "run_policy_v127.py"
    command = [
        sys.executable, str(script),
        "--inp", str(Path(args.inp).resolve()),
        "--out-dir", str(out),
        "--run-id", run_id,
        "--sensors", str(Path(args.sensors).resolve()),
        "--priority-nodes", str(Path(args.priority_nodes).resolve()),
        "--config", str(Path(args.config).resolve()),
        "--graph", str(Path(args.graph).resolve()),
        "--step1", str(Path(args.step1).resolve()),
        "--step2", str(Path(args.step2).resolve()),
        "--continuous-gate", str(Path(args.continuous_gate).resolve()),
        "--device", str(args.device),
        "--lbfgsb-maxiter", str(args.lbfgsb_maxiter),
        "--decision-runtime-budget-seconds", str(args.decision_runtime_budget_seconds),
        "--pfv-soft-margin-m3", str(args.pfv_soft_margin_m3),
        "--pfv-penalty-weight", str(args.pfv_penalty_weight),
    ]
    subprocess.run(command, check=True)
    meta_path = out / f"{run_id}.json"
    stats_path = out / f"{run_id}.node_statistics.csv.gz"
    decisions_path = out / f"{run_id}.decisions.jsonl"
    if not meta_path.is_file() or not stats_path.is_file() or not decisions_path.is_file():
        raise RuntimeError("V127 Proposed authoritative outputs are incomplete")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    decisions = [
        json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    continuous = sum(1 for row in decisions if str(row.get("source")) == "MPC_V127_CONTINUOUS")
    rbc = sum(1 for row in decisions if str(row.get("source")) == "RBC_SAFETY_V127")
    return {
        "event_id": str(args.event_id),
        "strategy": "proposed_v127",
        "tfv_m3": 0.0,
        "global_peak_flood_rate_m3s": float(meta["global_peak_flood_rate_m3s"]),
        "flow_routing_error_pct": float(meta["flow_routing_error_pct"]),
        "decisions": int(meta["decisions"]),
        "continuous_decisions": continuous,
        "rbc_safety_fallback_decisions": rbc,
        "metadata_path": str(meta_path.resolve()),
        "node_statistics_path": str(stats_path.resolve()),
        "decision_path": str(decisions_path.resolve()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inp", required=True)
    p.add_argument("--event-id", required=True)
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
    p.add_argument("--decision-runtime-budget-seconds", type=float, default=540.0)
    p.add_argument("--pfv-soft-margin-m3", type=float, default=100.0)
    p.add_argument("--pfv-penalty-weight", type=float, default=1.0)
    args = p.parse_args()

    required = [
        Path(args.inp), Path(args.sensors), Path(args.priority_nodes), Path(args.config),
        Path(args.native_controls_template), Path(args.graph), Path(args.step1),
        Path(args.step2), Path(args.continuous_gate),
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    priority = _priority_nodes(Path(args.priority_nodes))
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    runtime_cache = root / "_runtime_inp"
    rows = []
    for strategy in BASELINES:
        row = _run_one(
            strategy=strategy,
            inp=Path(args.inp).resolve(),
            sensors=Path(args.sensors).resolve(),
            config=Path(args.config).resolve(),
            native_controls_template=Path(args.native_controls_template).resolve(),
            root=root,
            runtime_cache=runtime_cache,
            event_id=str(args.event_id),
        )
        rows.append(_augment(row, priority))
    rows.append(_augment(_run_proposed(args, root), priority))
    no_control = next(row for row in rows if row["strategy"] == "no_control")
    nc_tfv = float(no_control["tfv_m3"])
    nc_pfv = float(no_control["priority8_pfv_m3"])
    for row in rows:
        tfv = float(row["tfv_m3"])
        pfv = float(row["priority8_pfv_m3"])
        row["tfv_reduction_vs_no_control_pct"] = 100.0 * (nc_tfv - tfv) / nc_tfv if nc_tfv > 0 else 0.0
        row["pfv_change_vs_no_control_pct"] = 100.0 * (pfv - nc_pfv) / nc_pfv if nc_pfv > 0 else 0.0
    csv_path = root / "PROJECT7_V127_SEVEN_STRATEGY_COMPARISON.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "contract": V127_SEVEN_STRATEGY_CONTRACT,
        "event_id": str(args.event_id),
        "strategies": [str(row["strategy"]) for row in rows],
        "tfv_primary": True,
        "priority8_pfv_soft_secondary": True,
        "global_peak_report_only": True,
        "rows": rows,
        "comparison_csv": str(csv_path.resolve()),
    }
    (root / "PROJECT7_V127_SEVEN_STRATEGY_COMPARISON.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
