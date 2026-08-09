from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .generation_contract import generation_key
from .inp_runtime import build_runtime_inp, sha256_file
from .replay_prefix import reference_trajectory_lineage


D3_DATA_CONTRACT = "D3_CONTROLS_DISABLED_COMPACT_V3_PREFIX_VERIFIED"


def _json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return value


def _generation(job: dict[str, object]) -> tuple[str, str, dict[str, object]]:
    payload = {
        "data_contract": D3_DATA_CONTRACT,
        "event_id": str(job["event_id"]),
        "rainfall_group": str(job["rainfall_group"]),
        "scientific_split": str(job["scientific_split"]),
        "development_fold": str(job["development_fold"]),
        "checkpoint_id": str(job["checkpoint_id"]),
        "checkpoint_minutes": int(job["checkpoint_minutes"]),
        "sequence_sha256": str(job["sequence_sha256"]),
        "runtime_inp_sha256": str(job["runtime_inp_sha256"]),
        "reference_metadata_sha256": str(job["reference_metadata_sha256"]),
        "reference_compact_sha256": str(job["reference_compact_sha256"]),
        "reference_swmm_engine_version": str(job["reference_swmm_engine_version"]),
        "control_block_seconds": int(job["control_block_seconds"]),
        "stride_seconds": int(job["stride_seconds"]),
        "swmm_threads_per_process": int(job["swmm_threads_per_process"]),
    }
    key, code_sha = generation_key("d3_sequence", payload)
    return key, code_sha, payload


def _stamp(metadata_path: str | Path, job: dict[str, object]) -> str:
    path = Path(metadata_path)
    meta = _json(path)
    verification = meta.get("same_prefix_verification")
    if not isinstance(verification, dict) or verification.get("passed") is not True:
        raise RuntimeError("Formal D3 branch lacks successful exact No-control prefix verification")
    key, code_sha, lineage = _generation(job)
    hashes: dict[str, str] = {}
    for field in ("compact_file", "node_statistics_file"):
        artifact = path.parent / str(meta[field])
        if not artifact.is_file():
            raise RuntimeError(f"D3 generated artifact missing: {artifact}")
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
        meta = _json(metadata)
        if meta.get("data_contract") != D3_DATA_CONTRACT:
            return False
        verification = meta.get("same_prefix_verification")
        if not isinstance(verification, dict) or verification.get("passed") is not True:
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


def _run(job: dict[str, object]) -> dict[str, object]:
    from .swmm_sequence import run_control_sequence_branch

    result = run_control_sequence_branch(
        inp_path=str(job["runtime_inp"]),
        checkpoint_minutes=int(job["checkpoint_minutes"]),
        settings_sequence=json.loads(str(job["settings_sequence_json"])),
        control_block_seconds=int(job["control_block_seconds"]),
        output_dir=str(job["out_dir"]),
        branch_id=str(job["branch_id"]),
        python_intervention_seconds=int(job["stride_seconds"]),
        reference_trajectory_metadata_path=str(job["reference_metadata_path"]),
    )
    key = _stamp(result.metadata_path, job)
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
        description="Run resumable D3 sequences with exact No-control prefix verification"
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
        "trajectory_metadata_path",
        "scientific_split",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"D3 manifest missing columns: {missing}")
    if (frame["scientific_split"].astype(str) == "final").any():
        raise ValueError("rtc-run-d3-batch refuses Final rows before Policy Lock")
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
        if not Path(source).is_file():
            raise ValueError(f"D3 source INP missing: {source}")
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
        reference_lineage = reference_trajectory_lineage(str(row["trajectory_metadata_path"]))
        branch_id = f"{event}__t{checkpoint:04d}__seq_{seq[:16]}"
        metadata = out / f"{branch_id}.json"
        job: dict[str, object] = {
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
            **reference_lineage,
        }
        expected_key, _, _ = _generation(job)
        if not args.no_resume and _complete(metadata, expected_key):
            results.append(
                {
                    "event_id": job["event_id"],
                    "rainfall_group": job["rainfall_group"],
                    "scientific_split": job["scientific_split"],
                    "development_fold": job["development_fold"],
                    "checkpoint_id": job["checkpoint_id"],
                    "data_role": job["data_role"],
                    "sequence_sha256": job["sequence_sha256"],
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
            futures = [pool.submit(_run, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
    summary = out / "D3_RUN_SUMMARY.csv"
    pd.DataFrame(results).sort_values(
        ["event_id", "checkpoint_id", "sequence_sha256"]
    ).to_csv(summary, index=False)
    print(
        json.dumps(
            {
                "contract": "D3_BATCH_PREFIX_VERIFIED_RESUME_V2",
                "branches": len(results),
                "computed": len(jobs),
                "resumed": len(results) - len(jobs),
                "workers": min(args.workers, max(1, len(jobs))),
                "summary": str(summary),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run_d3_batch_main()
