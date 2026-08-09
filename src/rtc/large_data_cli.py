from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .inp_runtime import build_runtime_inp, sha256_file


def _complete(metadata: Path, contract: str) -> bool:
    if not metadata.is_file():
        return False
    try:
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        if meta.get("data_contract") != contract:
            return False
        compact = metadata.parent / str(meta["compact_file"])
        stats = metadata.parent / str(meta["node_statistics_file"])
        return (
            compact.is_file()
            and compact.stat().st_size > 0
            and stats.is_file()
            and stats.stat().st_size > 0
        )
    except Exception:
        return False


def _reject_final_rows(frame: pd.DataFrame, *, context: str) -> None:
    if "scientific_split" not in frame.columns:
        raise ValueError(f"{context} requires scientific_split lineage")
    final = frame[frame["scientific_split"].astype(str) == "final"]
    if not final.empty:
        raise ValueError(
            f"{context} is a pre-Policy-Lock data generator and refuses {len(final)} Final rows. "
            "Create an explicit non-Final pilot/development manifest instead of revealing Final truth."
        )


def _d0_job(job: dict[str, object]) -> dict[str, object]:
    from .hydraulic_trajectory import run_hydraulic_trajectory

    result = run_hydraulic_trajectory(
        inp_path=str(job["runtime_inp"]),
        output_dir=str(job["out_dir"]),
        run_id=str(job["run_id"]),
        record_stride_seconds=int(job["record_stride_seconds"]),
    )
    return {
        "event_id": job["event_id"],
        "rainfall_group": job["rainfall_group"],
        "scientific_split": job["scientific_split"],
        "development_fold": job["development_fold"],
        "strategy": job["strategy"],
        "metadata_path": result.metadata_path,
        "flow_routing_error_pct": result.flow_routing_error_pct,
        "status": "completed",
    }


def run_d0_batch_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run compact pre-lock D0 trajectories in independent SWMM processes"
    )
    parser.add_argument(
        "--events",
        required=True,
        help="CSV: event_id,rainfall_group,inp_path,scientific_split[,development_fold]",
    )
    parser.add_argument(
        "--strategy", choices=["no_control", "internal_rtc"], default="no_control"
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--record-stride-seconds", type=int, default=300)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--swmm-threads-per-process", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    events = pd.read_csv(args.events)
    required = {"event_id", "rainfall_group", "inp_path", "scientific_split"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise ValueError(f"event registry missing columns: {missing}")
    _reject_final_rows(events, context="rtc-run-d0-batch")
    if args.record_stride_seconds <= 0:
        raise ValueError("record stride must be positive")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runtime_dir = out / "_runtime_inp"
    runtime_dir.mkdir(exist_ok=True)
    jobs: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    for _, row in events.iterrows():
        source = str(row["inp_path"])
        source_sha = sha256_file(source)
        runtime = runtime_dir / f"{str(row['event_id'])}__{source_sha[:12]}__{args.strategy}.inp"
        build_runtime_inp(
            source,
            runtime,
            native_controls=args.strategy == "internal_rtc",
            swmm_threads=args.swmm_threads_per_process,
        )
        run_id = f"{row['event_id']}__{args.strategy}"
        event_dir = out / str(row["event_id"])
        event_dir.mkdir(exist_ok=True)
        metadata = event_dir / f"{run_id}.json"
        base = {
            "event_id": str(row["event_id"]),
            "rainfall_group": str(row["rainfall_group"]),
            "scientific_split": str(row["scientific_split"]),
            "development_fold": str(row.get("development_fold", "")),
            "strategy": args.strategy,
            "runtime_inp": str(runtime),
            "out_dir": str(event_dir),
            "run_id": run_id,
            "record_stride_seconds": args.record_stride_seconds,
        }
        if not args.no_resume and _complete(metadata, "D0_D1_COMPACT_TRAJECTORY_V2"):
            results.append(
                {
                    **{
                        k: base[k]
                        for k in (
                            "event_id",
                            "rainfall_group",
                            "scientific_split",
                            "development_fold",
                            "strategy",
                        )
                    },
                    "metadata_path": str(metadata),
                    "flow_routing_error_pct": float("nan"),
                    "status": "resumed",
                }
            )
        else:
            jobs.append(base)
    if jobs:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            futures = [pool.submit(_d0_job, j) for j in jobs]
            for future in as_completed(futures):
                results.append(future.result())
    summary = out / f"D0_{args.strategy}_RUN_INDEX.csv"
    pd.DataFrame(results).sort_values("event_id").to_csv(summary, index=False)
    print(
        json.dumps(
            {
                "events": len(results),
                "computed": len(jobs),
                "workers": min(args.workers, max(1, len(jobs))),
                "summary": str(summary),
            },
            indent=2,
        )
    )


def _d3_job(job: dict[str, object]) -> dict[str, object]:
    from .swmm_sequence import run_control_sequence_branch

    result = run_control_sequence_branch(
        inp_path=str(job["runtime_inp"]),
        checkpoint_minutes=int(job["checkpoint_minutes"]),
        settings_sequence=json.loads(str(job["settings_sequence_json"])),
        control_block_seconds=int(job["control_block_seconds"]),
        output_dir=str(job["out_dir"]),
        branch_id=str(job["branch_id"]),
        python_intervention_seconds=int(job["stride_seconds"]),
    )
    return {
        "event_id": job["event_id"],
        "rainfall_group": job["rainfall_group"],
        "scientific_split": job["scientific_split"],
        "development_fold": job["development_fold"],
        "checkpoint_id": job["checkpoint_id"],
        "data_role": job["data_role"],
        "sequence_sha256": job["sequence_sha256"],
        "metadata_path": result.metadata_path,
        "flow_routing_error_pct": result.flow_routing_error_pct,
        "status": "completed",
    }


def run_d3_batch_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run compact pre-lock D3 sequences in independent SWMM processes"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--control-block-seconds", type=int, required=True)
    parser.add_argument("--stride-seconds", type=int, default=300)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--swmm-threads-per-process", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.manifest)
    required = {
        "sequence_sha256",
        "settings_sequence_json",
        "checkpoint_minutes",
        "checkpoint_id",
        "inp_path",
        "scientific_split",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"D3 manifest missing columns: {missing}")
    _reject_final_rows(frame, context="rtc-run-d3-batch")
    if args.control_block_seconds <= 0 or args.stride_seconds <= 0:
        raise ValueError("D3 control block and stride must be positive")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runtime_dir = out / "_runtime_inp"
    runtime_dir.mkdir(exist_ok=True)
    runtime_cache: dict[str, str] = {}
    jobs: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    for _, row in frame.drop_duplicates(["sequence_sha256", "checkpoint_id"]).iterrows():
        source = str(row["inp_path"])
        source_sha = sha256_file(source)
        if source_sha not in runtime_cache:
            runtime = runtime_dir / f"{source_sha[:16]}.no_control.inp"
            build_runtime_inp(
                source,
                runtime,
                native_controls=False,
                swmm_threads=args.swmm_threads_per_process,
            )
            runtime_cache[source_sha] = str(runtime)
        event = str(row.get("event_id", "event"))
        checkpoint = int(row["checkpoint_minutes"])
        seq = str(row["sequence_sha256"])
        branch_id = f"{event}__t{checkpoint:04d}__seq_{seq[:16]}"
        metadata = out / f"{branch_id}.json"
        base = {
            "runtime_inp": runtime_cache[source_sha],
            "out_dir": str(out),
            "branch_id": branch_id,
            "event_id": event,
            "rainfall_group": str(row.get("rainfall_group", "")),
            "scientific_split": str(row.get("scientific_split", "")),
            "development_fold": str(row.get("development_fold", "")),
            "checkpoint_id": str(row["checkpoint_id"]),
            "checkpoint_minutes": checkpoint,
            "data_role": str(row.get("data_role", "D3_MULTI_ACTUATOR_ROLLOUT")),
            "sequence_sha256": seq,
            "settings_sequence_json": str(row["settings_sequence_json"]),
            "control_block_seconds": args.control_block_seconds,
            "stride_seconds": args.stride_seconds,
        }
        if not args.no_resume and _complete(metadata, "D3_CONTROLS_DISABLED_COMPACT_V2"):
            results.append(
                {
                    **{
                        k: base[k]
                        for k in (
                            "event_id",
                            "rainfall_group",
                            "scientific_split",
                            "development_fold",
                            "checkpoint_id",
                            "data_role",
                            "sequence_sha256",
                        )
                    },
                    "metadata_path": str(metadata),
                    "flow_routing_error_pct": float("nan"),
                    "status": "resumed",
                }
            )
        else:
            jobs.append(base)
    if jobs:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            futures = [pool.submit(_d3_job, j) for j in jobs]
            for future in as_completed(futures):
                results.append(future.result())
    summary = out / "D3_RUN_SUMMARY.csv"
    pd.DataFrame(results).sort_values(
        ["event_id", "checkpoint_id", "sequence_sha256"]
    ).to_csv(summary, index=False)
    print(
        json.dumps(
            {
                "branches": len(results),
                "computed": len(jobs),
                "workers": min(args.workers, max(1, len(jobs))),
                "summary": str(summary),
            },
            indent=2,
        )
    )
