"""Run Project7 V127 Proposed plus six authoritative SWMM baselines on one event."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from rtc.execution_audit_v127 import audit_target_write_readback_v127
from rtc.replay_peak import replay_exact_global_peak

try:
    from run_six_baselines_v122 import BASELINES, _run_one
except ModuleNotFoundError:  # pragma: no cover
    from scripts.run_six_baselines_v122 import BASELINES, _run_one

V127_SEVEN_STRATEGY_CONTRACT = "PROJECT7_V127_SEVEN_STRATEGY_AUTHORITATIVE_SWMM_COMPARISON_V6_SEMANTIC_EVENT_WRITE_AUDIT"


def _priority_nodes(path: Path) -> set[str]:
    nodes = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
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
        if reader.fieldnames is None or not {
            "node_id",
            "delta_flooding_volume_m3",
        }.issubset(reader.fieldnames):
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
    result = dict(row)
    result["tfv_m3"] = tfv
    result["priority8_pfv_m3"] = pfv
    return result


def _exact_peak(row: dict[str, object]) -> dict[str, object]:
    result = dict(row)
    metadata_path = Path(str(result["metadata_path"]))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    write_audit = audit_target_write_readback_v127(metadata_path=metadata_path)
    if write_audit.get("passed") is not True:
        raise RuntimeError(
            f"{result.get('strategy')}: target write/readback audit failed: "
            + json.dumps(write_audit, sort_keys=True)
        )
    runtime_inp = Path(str(metadata["inp_path"]))
    decision_name = str(metadata.get("decision_file", ""))
    decision_path = metadata_path.parent / decision_name if decision_name else None
    if decision_path is not None and not decision_path.is_file():
        raise RuntimeError(f"missing frozen decision log for exact peak replay: {decision_path}")
    replay_path = metadata_path.with_suffix(".routing_step_peak.json")
    replay = replay_exact_global_peak(
        inp_path=runtime_inp,
        decision_log_path=decision_path,
        output_path=replay_path,
        source_main_metadata_path=metadata_path,
    )
    if str(replay.get("swmm_engine_version", "")) != str(metadata.get("swmm_engine_version", "")):
        raise RuntimeError("exact-peak replay SWMM engine differs from authoritative main run")
    result["target_write_readback_passed"] = True
    result["target_write_readback_max_error"] = float(
        write_audit["max_target_write_readback_error"]
    )
    result["sampled_300s_global_peak_flood_rate_m3s"] = float(
        result.get("global_peak_flood_rate_m3s", 0.0)
    )
    result["global_peak_flood_rate_m3s"] = float(
        replay["routing_step_global_peak_flood_rate_m3s"]
    )
    result["global_peak_semantics"] = "routing-step frozen-decision replay"
    result["global_peak_replay_path"] = str(replay_path.resolve())
    result["runtime_inp_sha256"] = str(metadata.get("inp_sha256", ""))
    result["source_inp_sha256_provenance"] = str(metadata.get("source_inp_sha256", ""))
    result["swmm_engine_version"] = str(metadata.get("swmm_engine_version", ""))
    result["control_start_minutes"] = int(metadata["control_start_minutes"])
    result["control_update_seconds"] = int(metadata["control_update_seconds"])
    result["observation_update_seconds"] = int(metadata["observation_update_seconds"])
    return result


def _run_proposed(args, root: Path) -> dict[str, object]:
    out = root / "proposed_v127"
    out.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.event_id}__proposed_v127"
    script = Path(__file__).resolve().parent / "run_policy_v127.py"
    command = [
        sys.executable,
        str(script),
        "--inp",
        str(Path(args.inp).resolve()),
        "--out-dir",
        str(out),
        "--run-id",
        run_id,
        "--sensors",
        str(Path(args.sensors).resolve()),
        "--priority-nodes",
        str(Path(args.priority_nodes).resolve()),
        "--config",
        str(Path(args.config).resolve()),
        "--graph",
        str(Path(args.graph).resolve()),
        "--step1",
        str(Path(args.step1).resolve()),
        "--step2",
        str(Path(args.step2).resolve()),
        "--continuous-gate",
        str(Path(args.continuous_gate).resolve()),
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
    meta_path = out / f"{run_id}.json"
    stats_path = out / f"{run_id}.node_statistics.csv.gz"
    decisions_path = out / f"{run_id}.decisions.jsonl"
    if not meta_path.is_file() or not stats_path.is_file() or not decisions_path.is_file():
        raise RuntimeError("V127 Proposed authoritative outputs are incomplete")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    decisions = [
        json.loads(line)
        for line in decisions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    continuous = sum(
        1 for item in decisions if str(item.get("source")) == "MPC_V127_CONTINUOUS"
    )
    rbc = sum(
        1 for item in decisions if str(item.get("source")) == "RBC_SAFETY_V127"
    )
    deadline = sum(
        1
        for item in decisions
        if bool((item.get("diagnostics") or {}).get("optimizer_deadline_exceeded", False))
    )
    runtimes = np.asarray(
        [
            float((item.get("diagnostics") or {}).get("optimizer_elapsed_seconds"))
            for item in decisions
            if (item.get("diagnostics") or {}).get("optimizer_elapsed_seconds") is not None
        ],
        dtype=float,
    )
    if runtimes.size and not np.isfinite(runtimes).all():
        raise RuntimeError("V127 decision log contains non-finite optimizer runtimes")
    return {
        "event_id": str(args.event_id),
        "strategy": "proposed_v127",
        "source_inp_path": str(Path(args.inp).resolve()),
        "tfv_m3": 0.0,
        "global_peak_flood_rate_m3s": float(meta["global_peak_flood_rate_m3s"]),
        "flow_routing_error_pct": float(meta["flow_routing_error_pct"]),
        "decisions": int(meta["decisions"]),
        "continuous_decisions": continuous,
        "rbc_safety_fallback_decisions": rbc,
        "optimizer_deadline_fallbacks": deadline,
        "optimizer_runtime_mean_s": float(runtimes.mean()) if runtimes.size else 0.0,
        "optimizer_runtime_p95_s": float(np.quantile(runtimes, 0.95)) if runtimes.size else 0.0,
        "optimizer_runtime_max_s": float(runtimes.max()) if runtimes.size else 0.0,
        "metadata_path": str(meta_path.resolve()),
        "node_statistics_path": str(stats_path.resolve()),
        "decision_path": str(decisions_path.resolve()),
    }


def _verify_common_execution(rows: list[dict[str, object]]) -> dict[str, object]:
    source_paths = {str(row.get("source_inp_path", "")) for row in rows}
    engines = {str(row.get("swmm_engine_version", "")) for row in rows}
    starts = {int(row["control_start_minutes"]) for row in rows}
    updates = {int(row["control_update_seconds"]) for row in rows}
    observations = {int(row["observation_update_seconds"]) for row in rows}
    if len(source_paths) != 1 or "" in source_paths:
        raise RuntimeError("seven strategies were not orchestrated from one source-event INP")
    if len(engines) != 1 or "" in engines:
        raise RuntimeError("seven strategies do not share one SWMM engine")
    if len(starts) != 1 or len(updates) != 1 or len(observations) != 1:
        raise RuntimeError("seven strategies do not share identical observation/control clocks")
    if any(row.get("target_write_readback_passed") is not True for row in rows):
        raise RuntimeError("one or more strategies failed target write/readback verification")
    return {
        "source_inp_path": next(iter(source_paths)),
        "swmm_engine_version": next(iter(engines)),
        "control_start_minutes": next(iter(starts)),
        "control_update_seconds": next(iter(updates)),
        "observation_update_seconds": next(iter(observations)),
        "target_write_readback_all_strategies": True,
        "source_file_sha_is_provenance_not_execution_gate": True,
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
    p.add_argument("--optimizer-deadline-seconds", type=float, default=480.0)
    p.add_argument("--decision-runtime-budget-seconds", type=float, default=540.0)
    p.add_argument("--pfv-soft-margin-m3", type=float, default=100.0)
    p.add_argument("--pfv-penalty-weight", type=float, default=1.0)
    args = p.parse_args()

    required = [
        Path(args.inp),
        Path(args.sensors),
        Path(args.priority_nodes),
        Path(args.config),
        Path(args.native_controls_template),
        Path(args.graph),
        Path(args.step1),
        Path(args.step2),
        Path(args.continuous_gate),
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0.0 < float(args.optimizer_deadline_seconds) < float(
        args.decision_runtime_budget_seconds
    ) < 600.0:
        raise ValueError(
            "V127 seven-strategy deadlines must satisfy optimizer < controller < 600 s"
        )

    priority = _priority_nodes(Path(args.priority_nodes))
    source_inp_path = str(Path(args.inp).resolve())
    root = Path(args.out_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    runtime_cache = root / "_runtime_inp"
    rows: list[dict[str, object]] = []
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
        row["source_inp_path"] = source_inp_path
        rows.append(_exact_peak(_augment(row, priority)))
    rows.append(_exact_peak(_augment(_run_proposed(args, root), priority)))
    common_execution = _verify_common_execution(rows)

    no_control = next(row for row in rows if row["strategy"] == "no_control")
    nc_tfv = float(no_control["tfv_m3"])
    nc_pfv = float(no_control["priority8_pfv_m3"])
    for row in rows:
        tfv = float(row["tfv_m3"])
        pfv = float(row["priority8_pfv_m3"])
        row["tfv_reduction_vs_no_control_pct"] = (
            100.0 * (nc_tfv - tfv) / nc_tfv if nc_tfv > 0 else 0.0
        )
        row["pfv_change_vs_no_control_pct"] = (
            100.0 * (pfv - nc_pfv) / nc_pfv if nc_pfv > 0 else 0.0
        )

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
        "global_peak_semantics": "routing-step frozen-decision replay for every strategy",
        "target_write_readback_all_strategies": True,
        "common_execution": common_execution,
        "rows": rows,
        "comparison_csv": str(csv_path.resolve()),
    }
    (root / "PROJECT7_V127_SEVEN_STRATEGY_COMPARISON.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
