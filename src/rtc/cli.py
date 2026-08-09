from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .contracts import load_priority_nodes
from .data_design import design_independent_actuator_probes, summarise_probe_design
from .graph import infer_flow_units, infer_system_units
from .inp import discover_actuators, discover_nodes
from .inp_runtime import build_runtime_inp, sha256_file


def audit_inp_main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit SWMM actuator/state/runtime contract before expensive data generation"
    )
    parser.add_argument("--inp", required=True)
    parser.add_argument("--priority")
    parser.add_argument("--out")
    args = parser.parse_args()
    catalog = discover_actuators(args.inp)
    nodes = discover_nodes(args.inp)
    priority = load_priority_nodes(args.priority) if args.priority else ()
    missing_priority: list[str] = []
    if priority:
        missing_priority = sorted(set(priority) - set(nodes))
    result = {
        "inp": str(Path(args.inp).resolve()),
        "inp_sha256": sha256_file(args.inp),
        "flow_units": infer_flow_units(args.inp),
        "system_units": infer_system_units(args.inp),
        "node_count": len(nodes),
        "actuator_count": len(catalog.actuators),
        "actuator_types": dict(Counter(a.kind for a in catalog.actuators)),
        "continuous_actuators": sum(a.continuous for a in catalog.actuators),
        "hard_binary_actuators": 0,
        "fixed_active_subset": None,
        "actuator_ids": list(catalog.ids),
        "priority_nodes": list(priority),
        "priority_nodes_present": not missing_priority,
        "missing_priority_nodes": missing_priority,
    }
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if missing_priority:
        raise SystemExit(2)


def design_probes_main() -> None:
    parser = argparse.ArgumentParser(
        description="Design same-checkpoint single-actuator D2 counterfactual probes"
    )
    parser.add_argument("--inp", required=True)
    parser.add_argument("--checkpoints", required=True, help="CSV with checkpoint_id and setting:<id>")
    parser.add_argument("--out", required=True)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--no-center", action="store_true")
    args = parser.parse_args()
    catalog = discover_actuators(args.inp)
    checkpoints = pd.read_csv(args.checkpoints)
    if "scientific_split" not in checkpoints.columns:
        raise ValueError("D2 checkpoint manifest requires scientific_split lineage")
    if (checkpoints["scientific_split"].astype(str) == "final").any():
        raise ValueError("D2 probe design refuses Final checkpoints before Policy Lock")
    manifest = design_independent_actuator_probes(
        checkpoints, catalog, epsilon=args.epsilon, include_center=not args.no_center
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    summary = summarise_probe_design(manifest)
    out.with_suffix(out.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


def _completed_probe(metadata_path: Path, *, action_sha: str) -> bool:
    if not metadata_path.is_file():
        return False
    try:
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        if meta.get("candidate_action_sha256") != action_sha:
            return False
        if meta.get("data_contract") != "D2_CONTROLS_DISABLED_COMPACT_V2":
            return False
        compact = metadata_path.parent / str(meta["compact_file"])
        stats = metadata_path.parent / str(meta["node_statistics_file"])
        return (
            compact.is_file()
            and compact.stat().st_size > 0
            and stats.is_file()
            and stats.stat().st_size > 0
        )
    except Exception:
        return False


def _run_probe_job(job: dict[str, object]) -> dict[str, object]:
    from .swmm_data import run_independent_control_branch

    result = run_independent_control_branch(
        inp_path=str(job["runtime_inp"]),
        checkpoint_minutes=int(job["checkpoint_minutes"]),
        horizon_minutes=int(job["horizon_minutes"]),
        candidate_settings=json.loads(str(job["candidate_settings_json"])),
        output_dir=str(job["out_dir"]),
        branch_id=str(job["branch_id"]),
        python_intervention_seconds=int(job["stride_seconds"]),
        save_raw_csv=bool(job["debug_raw"]),
        keep_engine_files=bool(job["keep_engine_files"]),
    )
    return {
        "branch_id": result.branch_id,
        "metadata_path": result.metadata_path,
        "candidate_action_sha256": str(job["candidate_action_sha256"]),
        "checkpoint_minutes": int(job["checkpoint_minutes"]),
        "checkpoint_id": str(job["checkpoint_id"]),
        "event_id": str(job["event_id"]),
        "rainfall_group": str(job["rainfall_group"]),
        "scientific_split": str(job["scientific_split"]),
        "development_fold": str(job["development_fold"]),
        "flow_routing_error_pct": result.flow_routing_error_pct,
        "status": "completed",
    }


def run_probes_main() -> None:
    """Execute pre-lock D2 in independent processes with resume and controls-disabled semantics."""

    parser = argparse.ArgumentParser(description="Run authoritative compact pre-lock D2 probe branches")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--inp", help="default event INP; may be overridden by manifest inp_path")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--horizon-minutes", type=int, default=60)
    parser.add_argument("--stride-seconds", type=int, default=300)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--swmm-threads-per-process", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--debug-raw", action="store_true")
    parser.add_argument("--keep-engine-files", action="store_true")
    args = parser.parse_args()
    if args.workers <= 0 or args.swmm_threads_per_process <= 0:
        raise ValueError("workers and SWMM threads/process must be positive")
    if args.horizon_minutes <= 0 or args.stride_seconds <= 0:
        raise ValueError("D2 horizon and stride must be positive")

    manifest = pd.read_csv(args.manifest)
    required = {
        "candidate_action_sha256",
        "candidate_settings_json",
        "checkpoint_minutes",
        "scientific_split",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"manifest missing required columns: {missing}")
    if (manifest["scientific_split"].astype(str) == "final").any():
        raise ValueError(
            "rtc-run-probes is a pre-Policy-Lock D2 generator and refuses Final rows"
        )
    dedup_cols = ["candidate_action_sha256", "checkpoint_minutes"]
    for optional in ("event_id", "rainfall_group", "inp_path"):
        if optional in manifest.columns:
            dedup_cols.append(optional)
    rows = manifest.drop_duplicates(dedup_cols).reset_index(drop=True)
    if args.limit is not None:
        rows = rows.head(args.limit)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = out_dir / "_runtime_inp"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_cache: dict[str, str] = {}
    jobs: list[dict[str, object]] = []
    resumed: list[dict[str, object]] = []

    for _, row in rows.iterrows():
        source = row.get("inp_path") if "inp_path" in rows.columns else None
        if pd.isna(source) or not source:
            source = args.inp
        if not source:
            raise ValueError("an INP is required via --inp or manifest inp_path")
        source = str(source)
        source_sha = sha256_file(source)
        if source_sha not in runtime_cache:
            runtime_path = runtime_dir / f"{source_sha[:16]}.no_control.inp"
            build_runtime_inp(
                source,
                runtime_path,
                native_controls=False,
                swmm_threads=args.swmm_threads_per_process,
            )
            runtime_cache[source_sha] = str(runtime_path)
        action_sha = str(row["candidate_action_sha256"])
        event = str(row.get("event_id", "event"))
        rainfall_group = str(row.get("rainfall_group", ""))
        split = str(row.get("scientific_split", ""))
        fold = str(row.get("development_fold", ""))
        checkpoint = int(row["checkpoint_minutes"])
        branch_id = f"{event}__t{checkpoint:04d}__{action_sha[:16]}"
        metadata_path = out_dir / f"{branch_id}.json"
        base = {
            "runtime_inp": runtime_cache[source_sha],
            "out_dir": str(out_dir),
            "branch_id": branch_id,
            "candidate_action_sha256": action_sha,
            "candidate_settings_json": str(row["candidate_settings_json"]),
            "checkpoint_minutes": checkpoint,
            "checkpoint_id": str(row.get("checkpoint_id", f"{event}:t{checkpoint}")),
            "event_id": event,
            "rainfall_group": rainfall_group,
            "scientific_split": split,
            "development_fold": fold,
            "horizon_minutes": args.horizon_minutes,
            "stride_seconds": args.stride_seconds,
            "debug_raw": args.debug_raw,
            "keep_engine_files": args.keep_engine_files,
        }
        if not args.no_resume and _completed_probe(metadata_path, action_sha=action_sha):
            resumed.append(
                {
                    "branch_id": branch_id,
                    "metadata_path": str(metadata_path),
                    "candidate_action_sha256": action_sha,
                    "checkpoint_minutes": checkpoint,
                    "checkpoint_id": base["checkpoint_id"],
                    "event_id": event,
                    "rainfall_group": rainfall_group,
                    "scientific_split": split,
                    "development_fold": fold,
                    "flow_routing_error_pct": float("nan"),
                    "status": "resumed",
                }
            )
        else:
            jobs.append(base)

    results = list(resumed)
    if jobs:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            futures = [pool.submit(_run_probe_job, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda r: str(r["branch_id"]))
    summary_path = out_dir / "RUN_SUMMARY.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(
        json.dumps(
            {
                "branches": len(results),
                "computed": len(jobs),
                "resumed": len(resumed),
                "workers": min(args.workers, max(1, len(jobs))),
                "swmm_threads_per_process": args.swmm_threads_per_process,
                "summary": str(summary_path),
            },
            indent=2,
        )
    )
