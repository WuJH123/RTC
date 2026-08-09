from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .generation_contract import generation_key
from .inp_runtime import build_runtime_inp, sha256_file


def _read_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return value


def _stamp_metadata(
    metadata_path: str | Path,
    *,
    generation_key_sha256: str,
    rtc_source_tree_sha256: str,
    lineage: dict[str, object],
) -> None:
    path = Path(metadata_path)
    meta = _read_json(path)
    artifact_hashes: dict[str, str] = {}
    for field in ("compact_file", "node_statistics_file", "decision_file"):
        raw = meta.get(field)
        if raw:
            artifact = path.parent / str(raw)
            if not artifact.is_file():
                raise RuntimeError(f"generated artifact missing before metadata stamp: {artifact}")
            artifact_hashes[field] = sha256_file(artifact)
    meta["generation_contract"] = "RTC_GENERATION_KEY_V1"
    meta["generation_key_sha256"] = generation_key_sha256
    meta["rtc_source_tree_sha256"] = rtc_source_tree_sha256
    meta["generation_lineage"] = lineage
    meta["generated_artifact_sha256"] = artifact_hashes
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _complete(metadata: Path, contract: str, expected_key: str) -> bool:
    if not metadata.is_file():
        return False
    try:
        meta = _read_json(metadata)
        if meta.get("data_contract") != contract:
            return False
        if meta.get("generation_key_sha256") != expected_key:
            return False
        _, code_sha = generation_key("code_probe", {})
        if meta.get("rtc_source_tree_sha256") != code_sha:
            return False
        hashes = meta.get("generated_artifact_sha256")
        if not isinstance(hashes, dict) or not hashes:
            return False
        for field, expected in hashes.items():
            raw = meta.get(str(field))
            if not raw:
                return False
            artifact = metadata.parent / str(raw)
            if not artifact.is_file() or sha256_file(artifact) != str(expected):
                return False
        return True
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


def _require_unique_events(frame: pd.DataFrame) -> None:
    if "event_id" not in frame.columns or frame["event_id"].astype(str).duplicated().any():
        raise ValueError("event registry must contain one unique row per event_id")


def _d0_generation(job: dict[str, object]) -> tuple[str, str, dict[str, object]]:
    payload = {
        "data_contract": "D0_D1_COMPACT_TRAJECTORY_V2",
        "event_id": str(job["event_id"]),
        "rainfall_group": str(job["rainfall_group"]),
        "scientific_split": str(job["scientific_split"]),
        "development_fold": str(job["development_fold"]),
        "strategy": str(job["strategy"]),
        "source_inp_sha256": str(job["source_inp_sha256"]),
        "runtime_inp_sha256": str(job["runtime_inp_sha256"]),
        "record_stride_seconds": int(job["record_stride_seconds"]),
        "swmm_threads_per_process": int(job["swmm_threads_per_process"]),
    }
    key, code_sha = generation_key("d0_trajectory", payload)
    return key, code_sha, payload


def _d0_job(job: dict[str, object]) -> dict[str, object]:
    from .hydraulic_trajectory import run_hydraulic_trajectory

    result = run_hydraulic_trajectory(
        inp_path=str(job["runtime_inp"]),
        output_dir=str(job["out_dir"]),
        run_id=str(job["run_id"]),
        record_stride_seconds=int(job["record_stride_seconds"]),
    )
    key, code_sha, lineage = _d0_generation(job)
    _stamp_metadata(
        result.metadata_path,
        generation_key_sha256=key,
        rtc_source_tree_sha256=code_sha,
        lineage=lineage,
    )
    return {
        "event_id": job["event_id"],
        "rainfall_group": job["rainfall_group"],
        "scientific_split": job["scientific_split"],
        "development_fold": job["development_fold"],
        "strategy": job["strategy"],
        "generation_key_sha256": key,
        "metadata_path": result.metadata_path,
        "flow_routing_error_pct": result.flow_routing_error_pct,
        "status": "completed",
    }


def run_d0_batch_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run code-bound resumable compact pre-lock D0 trajectories"
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
    _require_unique_events(events)
    _reject_final_rows(events, context="rtc-run-d0-batch")
    if args.record_stride_seconds <= 0 or args.workers <= 0 or args.swmm_threads_per_process <= 0:
        raise ValueError("record stride/workers/SWMM threads must be positive")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runtime_dir = out / "_runtime_inp"
    runtime_dir.mkdir(exist_ok=True)
    jobs: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    for _, row in events.iterrows():
        source = Path(str(row["inp_path"]))
        if not source.is_file():
            raise ValueError(f"event INP missing: {source}")
        source_sha = sha256_file(source)
        runtime = runtime_dir / f"{row['event_id']}__{source_sha[:12]}__{args.strategy}.inp"
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
        base: dict[str, object] = {
            "event_id": str(row["event_id"]),
            "rainfall_group": str(row["rainfall_group"]),
            "scientific_split": str(row["scientific_split"]),
            "development_fold": str(row.get("development_fold", "")),
            "strategy": args.strategy,
            "source_inp_sha256": source_sha,
            "runtime_inp": str(runtime),
            "runtime_inp_sha256": sha256_file(runtime),
            "out_dir": str(event_dir),
            "run_id": run_id,
            "record_stride_seconds": args.record_stride_seconds,
            "swmm_threads_per_process": args.swmm_threads_per_process,
        }
        expected_key, _, _ = _d0_generation(base)
        if not args.no_resume and _complete(
            metadata, "D0_D1_COMPACT_TRAJECTORY_V2", expected_key
        ):
            results.append(
                {
                    "event_id": base["event_id"],
                    "rainfall_group": base["rainfall_group"],
                    "scientific_split": base["scientific_split"],
                    "development_fold": base["development_fold"],
                    "strategy": base["strategy"],
                    "generation_key_sha256": expected_key,
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
                "contract": "D0_BATCH_CODE_BOUND_RESUME_V1",
                "events": len(results),
                "computed": len(jobs),
                "resumed": len(results) - len(jobs),
                "workers": min(args.workers, max(1, len(jobs))),
                "summary": str(summary),
            },
            indent=2,
        )
    )


def _stable_event_seed(base_seed: int, event_id: str, rainfall_group: str) -> int:
    raw = hashlib.sha256(f"{base_seed}|{event_id}|{rainfall_group}".encode("utf-8")).digest()
    return int.from_bytes(raw[:4], "big") & 0x7FFFFFFF


def _d1_generation(job: dict[str, object]) -> tuple[str, str, dict[str, object]]:
    payload = {
        "data_contract": "CLOSED_LOOP_COMPACT_V2",
        "data_role": "D1_DEVELOPMENT_CONTINUOUS_EXPLORATION_V3",
        "event_id": str(job["event_id"]),
        "rainfall_group": str(job["rainfall_group"]),
        "scientific_split": "development",
        "development_fold": "train",
        "source_inp_sha256": str(job["source_inp_sha256"]),
        "runtime_inp_sha256": str(job["runtime_inp_sha256"]),
        "sensor_ids": list(job["sensor_ids"]),
        "seed": int(job["event_seed"]),
        "model_step_seconds": int(job["model_step_seconds"]),
        "control_update_seconds": int(job["control_update_seconds"]),
        "control_start_minutes": int(job["control_start_minutes"]),
        "perturbation_std": float(job["perturbation_std"]),
        "change_probability": float(job["change_probability"]),
        "max_delta": float(job["max_delta"]),
        "swmm_threads_per_process": int(job["swmm_threads_per_process"]),
    }
    key, code_sha = generation_key("d1_exploration", payload)
    return key, code_sha, payload


def _d1_job(job: dict[str, object]) -> dict[str, object]:
    from .closed_loop import run_authoritative_closed_loop
    from .d1_exploration import ContinuousExplorationController

    controller = ContinuousExplorationController(
        seed=int(job["event_seed"]),
        perturbation_std=float(job["perturbation_std"]),
        change_probability=float(job["change_probability"]),
        max_delta_per_update=float(job["max_delta"]),
    )
    result = run_authoritative_closed_loop(
        inp_path=str(job["runtime_inp"]),
        output_dir=str(job["out_dir"]),
        run_id=str(job["run_id"]),
        sensor_nodes=tuple(str(x) for x in job["sensor_ids"]),
        controller=controller,
        control_start_minutes=int(job["control_start_minutes"]),
        control_update_seconds=int(job["control_update_seconds"]),
        observation_update_seconds=int(job["model_step_seconds"]),
        record_stride_seconds=int(job["model_step_seconds"]),
        exact_global_peak=False,
        save_raw_csv=False,
    )
    key, code_sha, lineage = _d1_generation(job)
    _stamp_metadata(
        result.metadata_path,
        generation_key_sha256=key,
        rtc_source_tree_sha256=code_sha,
        lineage=lineage,
    )
    return {
        "event_id": job["event_id"],
        "rainfall_group": job["rainfall_group"],
        "scientific_split": "development",
        "development_fold": "train",
        "strategy": "d1_exploration",
        "generation_key_sha256": key,
        "metadata_path": result.metadata_path,
        "flow_routing_error_pct": result.flow_routing_error_pct,
        "status": "completed",
    }


def run_d1_batch_main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate resumable development/train D1 continuous exploration trajectories"
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--sensors", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-step-seconds", type=int, required=True)
    parser.add_argument("--control-update-seconds", type=int, required=True)
    parser.add_argument("--control-start-minutes", type=int, default=0)
    parser.add_argument("--perturbation-std", type=float, default=0.12)
    parser.add_argument("--change-probability", type=float, default=0.35)
    parser.add_argument("--max-delta", type=float, default=0.20)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--swmm-threads-per-process", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(args.events)
    required = {"event_id", "rainfall_group", "inp_path", "scientific_split", "development_fold"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"D1 event registry missing columns: {missing}")
    _require_unique_events(frame)
    bad = frame[
        (frame["scientific_split"].astype(str) != "development")
        | (frame["development_fold"].astype(str) != "train")
    ]
    if not bad.empty:
        raise ValueError("D1 batch accepts development/train event rows only")
    if args.control_update_seconds % args.model_step_seconds:
        raise ValueError("D1 control update must be an integer multiple of model step")
    sensors = tuple(
        line.strip()
        for line in Path(args.sensors).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not sensors:
        raise ValueError("D1 sensor file is empty")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runtime_dir = out / "_runtime_inp"
    runtime_dir.mkdir(exist_ok=True)
    jobs: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    runtime_cache: dict[str, Path] = {}
    for _, row in frame.iterrows():
        source = Path(str(row["inp_path"]))
        if not source.is_file():
            raise ValueError(f"D1 source INP missing: {source}")
        source_sha = sha256_file(source)
        if source_sha not in runtime_cache:
            runtime = runtime_dir / f"{source_sha[:20]}.d1.no_control.t{args.swmm_threads_per_process}.inp"
            build_runtime_inp(
                source,
                runtime,
                native_controls=False,
                swmm_threads=args.swmm_threads_per_process,
            )
            runtime_cache[source_sha] = runtime
        runtime = runtime_cache[source_sha]
        event_id = str(row["event_id"])
        rainfall_group = str(row["rainfall_group"])
        event_dir = out / event_id
        event_dir.mkdir(exist_ok=True)
        run_id = f"{event_id}__d1"
        metadata = event_dir / f"{run_id}.json"
        base: dict[str, object] = {
            "event_id": event_id,
            "rainfall_group": rainfall_group,
            "source_inp_sha256": source_sha,
            "runtime_inp": str(runtime),
            "runtime_inp_sha256": sha256_file(runtime),
            "sensor_ids": sensors,
            "event_seed": _stable_event_seed(args.seed, event_id, rainfall_group),
            "model_step_seconds": args.model_step_seconds,
            "control_update_seconds": args.control_update_seconds,
            "control_start_minutes": args.control_start_minutes,
            "perturbation_std": args.perturbation_std,
            "change_probability": args.change_probability,
            "max_delta": args.max_delta,
            "swmm_threads_per_process": args.swmm_threads_per_process,
            "out_dir": str(event_dir),
            "run_id": run_id,
        }
        expected_key, _, _ = _d1_generation(base)
        if not args.no_resume and _complete(metadata, "CLOSED_LOOP_COMPACT_V2", expected_key):
            results.append(
                {
                    "event_id": event_id,
                    "rainfall_group": rainfall_group,
                    "scientific_split": "development",
                    "development_fold": "train",
                    "strategy": "d1_exploration",
                    "generation_key_sha256": expected_key,
                    "metadata_path": str(metadata),
                    "flow_routing_error_pct": float("nan"),
                    "status": "resumed",
                }
            )
        else:
            jobs.append(base)
    if jobs:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            futures = [pool.submit(_d1_job, j) for j in jobs]
            for future in as_completed(futures):
                results.append(future.result())
    summary = out / "D1_RUN_INDEX.csv"
    pd.DataFrame(results).sort_values("event_id").to_csv(summary, index=False)
    print(
        json.dumps(
            {
                "contract": "D1_BATCH_CODE_BOUND_RESUME_V1",
                "events": len(results),
                "computed": len(jobs),
                "resumed": len(results) - len(jobs),
                "summary": str(summary),
            },
            indent=2,
        )
    )


def _d3_generation(job: dict[str, object]) -> tuple[str, str, dict[str, object]]:
    payload = {
        "data_contract": "D3_CONTROLS_DISABLED_COMPACT_V2",
        "event_id": str(job["event_id"]),
        "rainfall_group": str(job["rainfall_group"]),
        "scientific_split": str(job["scientific_split"]),
        "development_fold": str(job["development_fold"]),
        "checkpoint_id": str(job["checkpoint_id"]),
        "checkpoint_minutes": int(job["checkpoint_minutes"]),
        "sequence_sha256": str(job["sequence_sha256"]),
        "runtime_inp_sha256": str(job["runtime_inp_sha256"]),
        "control_block_seconds": int(job["control_block_seconds"]),
        "stride_seconds": int(job["stride_seconds"]),
        "swmm_threads_per_process": int(job["swmm_threads_per_process"]),
    }
    key, code_sha = generation_key("d3_sequence", payload)
    return key, code_sha, payload


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
    key, code_sha, lineage = _d3_generation(job)
    _stamp_metadata(
        result.metadata_path,
        generation_key_sha256=key,
        rtc_source_tree_sha256=code_sha,
        lineage=lineage,
    )
    return {
        "event_id": job["event_id"],
        "rainfall_group": job["rainfall_group"],
        "scientific_split": job["scientific_split"],
        "development_fold": job["development_fold"],
        "checkpoint_id": job["checkpoint_id"],
        "data_role": job["data_role"],
        "sequence_sha256": job["sequence_sha256"],
        "generation_key_sha256": key,
        "metadata_path": result.metadata_path,
        "flow_routing_error_pct": result.flow_routing_error_pct,
        "status": "completed",
    }


def run_d3_batch_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run code-bound resumable compact pre-lock D3 sequences"
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
    if args.control_block_seconds % args.stride_seconds:
        raise ValueError("D3 control block must be a multiple of stride")

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
            runtime = runtime_dir / f"{source_sha[:16]}.no_control.t{args.swmm_threads_per_process}.inp"
            build_runtime_inp(
                source,
                runtime,
                native_controls=False,
                swmm_threads=args.swmm_threads_per_process,
            )
            runtime_cache[source_sha] = str(runtime)
        runtime = runtime_cache[source_sha]
        event = str(row.get("event_id", "event"))
        checkpoint = int(row["checkpoint_minutes"])
        seq = str(row["sequence_sha256"])
        branch_id = f"{event}__t{checkpoint:04d}__seq_{seq[:16]}"
        metadata = out / f"{branch_id}.json"
        base: dict[str, object] = {
            "runtime_inp": runtime,
            "runtime_inp_sha256": sha256_file(runtime),
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
            "swmm_threads_per_process": args.swmm_threads_per_process,
        }
        expected_key, _, _ = _d3_generation(base)
        if not args.no_resume and _complete(
            metadata, "D3_CONTROLS_DISABLED_COMPACT_V2", expected_key
        ):
            results.append(
                {
                    "event_id": base["event_id"],
                    "rainfall_group": base["rainfall_group"],
                    "scientific_split": base["scientific_split"],
                    "development_fold": base["development_fold"],
                    "checkpoint_id": base["checkpoint_id"],
                    "data_role": base["data_role"],
                    "sequence_sha256": base["sequence_sha256"],
                    "generation_key_sha256": expected_key,
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
                "contract": "D3_BATCH_CODE_BOUND_RESUME_V1",
                "branches": len(results),
                "computed": len(jobs),
                "resumed": len(results) - len(jobs),
                "workers": min(args.workers, max(1, len(jobs))),
                "summary": str(summary),
            },
            indent=2,
        )
    )
