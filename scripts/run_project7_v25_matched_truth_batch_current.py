"""Run only missing current-V23 hydraulic candidate branches and finalize each output.

The audit JSON is the authority for the query list.  Existing completed output directories are
detected by their matched-record JSON and are never rerun.  This driver creates no rainfall and
never reruns the shared HOLD branch.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _completed(out_root: Path) -> set[str]:
    result: set[str] = set()
    for path in out_root.rglob("V25_MATCHED_COUNTERFACTUAL_RECORD.json") if out_root.exists() else ():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("query_set_id"):
                result.add(str(payload["query_set_id"]))
        except Exception:
            continue
    return result


def _run_one(job: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(str(job["out_dir"])).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_project7_v25_matched_counterfactual_current.py").resolve()
    finalizer = Path(__file__).with_name("finalize_project7_v25_matched_counterfactual_current.py").resolve()
    common = [
        "--asset-manifest", args.asset_manifest,
        "--records-jsonl", args.records_jsonl,
        "--query-set-id", str(job["query_set_id"]),
        "--v15-rank-checkpoint", args.v15_rank_checkpoint,
        "--v21-boundary-checkpoint", args.v21_boundary_checkpoint,
        "--out-dir", str(out_dir),
        "--priority-nodes", args.priority_nodes,
        "--device", args.device,
        "--decision-runtime-budget-seconds", str(args.decision_runtime_budget_seconds),
        "--probe-chunk-size", str(args.probe_chunk_size),
    ]
    env = os.environ.copy()
    cmd = [sys.executable, str(runner), *common]
    started = __import__("time").time()
    proc = subprocess.run(cmd, cwd=str(Path(__file__).parents[1]), env=env, text=True, capture_output=True)
    log = {
        "query_set_id": str(job["query_set_id"]),
        "event_id": str(job["event_id"]),
        "run_command": cmd,
        "runner_exit_code": int(proc.returncode),
        "runner_stdout": proc.stdout,
        "runner_stderr": proc.stderr,
    }
    record_path = out_dir / "V25_MATCHED_COUNTERFACTUAL_RECORD.json"
    if not record_path.is_file():
        metadata = [p for p in out_dir.glob("*.json") if p.name != record_path.name]
        if metadata:
            final_cmd = [sys.executable, str(finalizer), *common]
            final_proc = subprocess.run(final_cmd, cwd=str(Path(__file__).parents[1]), env=env, text=True, capture_output=True)
            log.update({
                "finalizer_command": final_cmd,
                "finalizer_exit_code": int(final_proc.returncode),
                "finalizer_stdout": final_proc.stdout,
                "finalizer_stderr": final_proc.stderr,
            })
    log["elapsed_seconds"] = __import__("time").time() - started
    log["record_path"] = str(record_path) if record_path.is_file() else None
    (out_dir / "V25_MATCHED_COUNTERFACTUAL_EXECUTION.json").write_text(
        json.dumps(log, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not record_path.is_file():
        raise RuntimeError(f"query did not produce a finalized matched record: {out_dir}")
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    return {
        "query_set_id": str(job["query_set_id"]),
        "event_id": str(job["event_id"]),
        "out_dir": str(out_dir),
        "runner_exit_code": int(proc.returncode),
        "record_path": str(record_path),
        "h120_delta_tfv_m3": float(payload["true_policy_return_delta_tfv_h120_m3"]),
        "new_swmm_runs": int(payload["new_swmm_runs"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--v15-rank-checkpoint", required=True)
    parser.add_argument("--v21-boundary-checkpoint", required=True)
    parser.add_argument("--priority-nodes", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--completed-root", action="append", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-chunk-size", type=int, default=24)
    parser.add_argument("--decision-runtime-budget-seconds", type=float, default=180.0)
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    done = _completed(out_root)
    for root in args.completed_root:
        done.update(_completed(Path(root).resolve()))
    jobs = [
        row for row in audit["query_reports"]
        if not bool(row.get("hydraulic_exact_match")) and str(row["query_set_id"]) not in done
    ]
    if not jobs:
        raise RuntimeError("no missing current-V23 hydraulic queries remain; refusing a no-op run")
    for job in jobs:
        job["out_dir"] = str(out_root / f"query_{_safe(str(job['event_id']))}_{str(job['query_set_id'])[:12]}")
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    workers = max(1, min(int(args.max_workers), len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, job, args): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(json.dumps({"completed": result}, sort_keys=True), flush=True)
            except Exception as exc:
                failure = {"query_set_id": job["query_set_id"], "event_id": job["event_id"], "error": f"{type(exc).__name__}: {exc}"}
                failures.append(failure)
                print(json.dumps({"failed": failure}, sort_keys=True), flush=True)
    summary = {
        "contract": "PROJECT7_V25_MISSING_CURRENT_V23_CANDIDATE_MATCHED_BATCH_V1",
        "development_only": True,
        "formal_evidence": False,
        "new_rainfall_generated": False,
        "new_training_scenario_generated": False,
        "query_count_requested": len(jobs),
        "query_count_completed": len(results),
        "new_swmm_runs": sum(int(row.get("new_swmm_runs", 0)) for row in results),
        "failures": failures,
        "results": sorted(results, key=lambda row: row["query_set_id"]),
        "max_workers": workers,
    }
    summary_path = out_root / "V25_MATCHED_COUNTERFACTUAL_BATCH_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary_path": str(summary_path), **{key: summary[key] for key in ("query_count_requested", "query_count_completed", "new_swmm_runs", "failures")}}, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
