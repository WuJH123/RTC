"""Run Project7 V128 Proposed plus six fixed authoritative SWMM baselines."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy as np

V128_SEVEN_STRATEGY_CONTRACT = (
    "PROJECT7_V128_SEVEN_STRATEGY_AUTHORITATIVE_SWMM_COMPARISON_V1_TYPED_RUNTIME_ACCEPTED"
)


def _load_v127_runner() -> ModuleType:
    path = Path(__file__).with_name("run_seven_strategies_v127.py")
    spec = importlib.util.spec_from_file_location("_rtc_v127_seven", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical V127 seven-strategy runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_proposed_v128(args, root: Path) -> dict[str, object]:
    out = root / "proposed_v128"
    out.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.event_id}__proposed_v128"
    script = Path(__file__).resolve().parent / "run_policy_v128.py"
    command = [
        sys.executable,
        str(script),
        "--inp", str(Path(args.inp).resolve()),
        "--out-dir", str(out),
        "--run-id", run_id,
        "--sensors", str(Path(args.sensors).resolve()),
        "--priority-nodes", str(Path(args.priority_nodes).resolve()),
        "--config", str(Path(args.config).resolve()),
        "--graph", str(Path(args.graph).resolve()),
        "--step1", str(Path(args.step1).resolve()),
        "--step2", str(Path(args.step2).resolve()),
        "--continuous-evidence", str(Path(args.continuous_gate).resolve()),
        "--device", str(args.device),
        "--lbfgsb-maxiter", str(args.lbfgsb_maxiter),
        "--optimizer-deadline-seconds", str(args.optimizer_deadline_seconds),
        "--decision-runtime-budget-seconds", str(args.decision_runtime_budget_seconds),
        "--pfv-soft-margin-m3", str(args.pfv_soft_margin_m3),
        "--pfv-penalty-weight", str(args.pfv_penalty_weight),
    ]
    engineering = getattr(args, "engineering_envelope", None)
    if engineering:
        command += ["--engineering-envelope", str(Path(engineering).resolve())]
    subprocess.run(command, check=True)

    meta_path = out / f"{run_id}.json"
    stats_path = out / f"{run_id}.node_statistics.csv.gz"
    decisions_path = out / f"{run_id}.decisions.jsonl"
    runtime_acceptance_path = out / f"{run_id}.runtime_acceptance.json"
    required = (meta_path, stats_path, decisions_path, runtime_acceptance_path)
    if any(not path.is_file() for path in required):
        raise RuntimeError("V128 Proposed authoritative outputs are incomplete")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    acceptance = json.loads(runtime_acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("passed") is not True:
        raise RuntimeError("V128 Proposed failed measured 600-s runtime acceptance")
    decisions = [
        json.loads(line)
        for line in decisions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    continuous = sum(
        1 for item in decisions if str(item.get("source")) == "MPC_V128_CONTINUOUS"
    )
    rbc = sum(
        1 for item in decisions if str(item.get("source")) == "RBC_SAFETY_V128"
    )
    deadline = sum(
        1
        for item in decisions
        if bool((item.get("diagnostics") or {}).get("optimizer_deadline_exceeded", False))
    )
    optimizer_runtimes = np.asarray(
        [
            float((item.get("diagnostics") or {}).get("optimizer_elapsed_seconds"))
            for item in decisions
            if (item.get("diagnostics") or {}).get("optimizer_elapsed_seconds") is not None
        ],
        dtype=float,
    )
    if optimizer_runtimes.size and not np.isfinite(optimizer_runtimes).all():
        raise RuntimeError("V128 decision log contains non-finite optimizer runtimes")
    measured = acceptance["decision_runtime_seconds"]
    return {
        "event_id": str(args.event_id),
        "strategy": "proposed_v128",
        "source_inp_path": str(Path(args.inp).resolve()),
        "tfv_m3": 0.0,
        "global_peak_flood_rate_m3s": float(meta["global_peak_flood_rate_m3s"]),
        "flow_routing_error_pct": float(meta["flow_routing_error_pct"]),
        "decisions": int(meta["decisions"]),
        "continuous_decisions": continuous,
        "rbc_safety_fallback_decisions": rbc,
        "optimizer_deadline_fallbacks": deadline,
        "optimizer_runtime_mean_s": float(optimizer_runtimes.mean()) if optimizer_runtimes.size else 0.0,
        "optimizer_runtime_p95_s": float(np.quantile(optimizer_runtimes, 0.95)) if optimizer_runtimes.size else 0.0,
        "optimizer_runtime_max_s": float(optimizer_runtimes.max()) if optimizer_runtimes.size else 0.0,
        "decision_runtime_mean_s": float(measured["mean"]),
        "decision_runtime_p50_s": float(measured["p50"]),
        "decision_runtime_p95_s": float(measured["p95"]),
        "decision_runtime_max_s": float(measured["max"]),
        "measured_600s_runtime_passed": True,
        "engineering_envelope_semantic_sha256": str(meta["engineering_envelope_semantic_sha256"]),
        "metadata_path": str(meta_path.resolve()),
        "node_statistics_path": str(stats_path.resolve()),
        "decision_path": str(decisions_path.resolve()),
        "runtime_acceptance_path": str(runtime_acceptance_path.resolve()),
    }


def main() -> None:
    runner = _load_v127_runner()
    # Reuse the already audited baseline execution/statistics/replay code. Only Proposed and
    # artifact identity differ. Keep the V127 CLI parser for all common arguments.
    runner._run_proposed = _run_proposed_v128
    runner.V127_SEVEN_STRATEGY_CONTRACT = V128_SEVEN_STRATEGY_CONTRACT

    # The common V127 parser calls this argument --continuous-gate. Expose the V128-facing
    # name while translating only the argv token; the value itself is V128 evidence.
    if "--continuous-evidence" in sys.argv:
        sys.argv[sys.argv.index("--continuous-evidence")] = "--continuous-gate"
    runner.main()

    # The shared runner intentionally retains its historical output filenames. Rename and
    # stamp the final artifacts so downstream code cannot confuse V128 with V127.
    out_index = sys.argv.index("--out-dir")
    root = Path(sys.argv[out_index + 1]).resolve()
    old_json = root / "PROJECT7_V127_SEVEN_STRATEGY_COMPARISON.json"
    old_csv = root / "PROJECT7_V127_SEVEN_STRATEGY_COMPARISON.csv"
    new_json = root / "PROJECT7_V128_SEVEN_STRATEGY_COMPARISON.json"
    new_csv = root / "PROJECT7_V128_SEVEN_STRATEGY_COMPARISON.csv"
    if not old_json.is_file() or not old_csv.is_file():
        raise RuntimeError("shared seven-strategy runner did not emit expected artifacts")
    payload = json.loads(old_json.read_text(encoding="utf-8"))
    payload["contract"] = V128_SEVEN_STRATEGY_CONTRACT
    payload["proposed_contract"] = "V128 typed Step2 + envelope-aware continuous MPC"
    payload["comparison_csv"] = str(new_csv.resolve())
    old_csv.replace(new_csv)
    old_json.unlink()
    new_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
