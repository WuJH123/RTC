from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .generation_contract import generation_key
from .inp_runtime import build_runtime_inp, sha256_file


D0_DATA_CONTRACT = "D0_D1_COMPACT_TRAJECTORY_V3_T0_CAUSAL"


def _read_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return value


def _generation(job: dict[str, object]) -> tuple[str, str, dict[str, object]]:
    payload = {
        "data_contract": D0_DATA_CONTRACT,
        "causal_timing_revision": "T0_INCLUDED_V2",
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
    key, code_sha = generation_key("d0_trajectory_t0", payload)
    return key, code_sha, payload


def _stamp(metadata_path: str | Path, job: dict[str, object]) -> str:
    path = Path(metadata_path)
    meta = _read_json(path)
    if meta.get("data_contract") != D0_DATA_CONTRACT:
        raise RuntimeError("D0 generator returned an incompatible t0 data contract")
    if int(meta.get("initial_observation_elapsed_seconds", -1)) != 0:
        raise RuntimeError("D0 trajectory does not contain the required t=0 frame")
    key, code_sha, lineage = _generation(job)
    hashes: dict[str, str] = {}
    for field in ("compact_file", "node_statistics_file"):
        artifact = path.parent / str(meta[field])
        if not artifact.is_file():
            raise RuntimeError(f"D0 generated artifact missing: {artifact}")
        hashes[field] = sha256_file(artifact)
    meta.update(
        {
            "generation_contract": "RTC_GENERATION_KEY_V1",
            "generation_key_sha256": key,
            "rtc_source_tree_sha256": code_sha,
            "generation_lineage": lineage,
            "generated_artifact_sha256": hashes,
        }
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return key


def _complete(metadata: Path, expected_key: str) -> bool:
    if not metadata.is_file():
        return False
    try:
        meta = _read_json(metadata)
        if meta.get("data_contract") != D0_DATA_CONTRACT:
            return False
        if int(meta.get("initial_observation_elapsed_seconds", -1)) != 0:
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
            artifact = metadata.parent / str(meta[field])
            if not artifact.is_file() or sha256_file(artifact) != str(expected):
                return False
        return True
    except Exception:
        return False


def _job(job: dict[str, object]) -> dict[str, object]:
    from .hydraulic_trajectory import run_hydraulic_trajectory

    result = run_hydraulic_trajectory(
        inp_path=str(job["runtime_inp"]),
        output_dir=str(job["out_dir"]),
        run_id=str(job["run_id"]),
        record_stride_seconds=int(job["record_stride_seconds"]),
    )
    key = _stamp(result.metadata_path, job)
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
        description="Run t=0-inclusive code-bound resumable compact pre-lock D0 trajectories"
    )
    parser.add_argument("--events", required=True)
    parser.add_argument("--strategy", choices=["no_control", "internal_rtc"], default="no_control")
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
    if events["event_id"].astype(str).duplicated().any():
        raise ValueError("event registry must contain one unique row per event_id")
    if (events["scientific_split"].astype(str) == "final").any():
        raise ValueError("rtc-run-d0-batch refuses Final rows before Policy Lock")
    if min(args.record_stride_seconds, args.workers, args.swmm_threads_per_process) <= 0:
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
        job: dict[str, object] = {
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
        expected_key, _, _ = _generation(job)
        if not args.no_resume and _complete(metadata, expected_key):
            results.append(
                {
                    "event_id": job["event_id"],
                    "rainfall_group": job["rainfall_group"],
                    "scientific_split": job["scientific_split"],
                    "development_fold": job["development_fold"],
                    "strategy": job["strategy"],
                    "generation_key_sha256": expected_key,
                    "metadata_path": str(metadata),
                    "flow_routing_error_pct": float("nan"),
                    "status": "resumed",
                }
            )
        else:
            jobs.append(job)
    if jobs:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            futures = [pool.submit(_job, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
    summary = out / f"D0_{args.strategy}_RUN_INDEX.csv"
    pd.DataFrame(results).sort_values("event_id").to_csv(summary, index=False)
    print(
        json.dumps(
            {
                "contract": "D0_BATCH_T0_CAUSAL_RESUME_V2",
                "events": len(results),
                "computed": len(jobs),
                "resumed": len(results) - len(jobs),
                "workers": min(args.workers, max(1, len(jobs))),
                "summary": str(summary),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run_d0_batch_main()
