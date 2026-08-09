from __future__ import annotations

import argparse
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


def _generation(job: dict[str, object]) -> tuple[str, str, dict[str, object]]:
    payload = {
        "data_contract": "D2_CONTROLS_DISABLED_COMPACT_V2",
        "event_id": str(job["event_id"]),
        "rainfall_group": str(job["rainfall_group"]),
        "scientific_split": str(job["scientific_split"]),
        "development_fold": str(job["development_fold"]),
        "checkpoint_id": str(job["checkpoint_id"]),
        "checkpoint_minutes": int(job["checkpoint_minutes"]),
        "candidate_action_sha256": str(job["candidate_action_sha256"]),
        "source_inp_sha256": str(job["source_inp_sha256"]),
        "runtime_inp_sha256": str(job["runtime_inp_sha256"]),
        "horizon_minutes": int(job["horizon_minutes"]),
        "stride_seconds": int(job["stride_seconds"]),
        "swmm_threads_per_process": int(job["swmm_threads_per_process"]),
        "debug_raw": bool(job["debug_raw"]),
    }
    key, code_sha = generation_key("d2_counterfactual", payload)
    return key, code_sha, payload


def _stamp(metadata_path: str | Path, job: dict[str, object]) -> str:
    path = Path(metadata_path)
    meta = _read_json(path)
    key, code_sha, lineage = _generation(job)
    hashes: dict[str, str] = {}
    for field in ("compact_file", "node_statistics_file"):
        artifact = path.parent / str(meta[field])
        if not artifact.is_file():
            raise RuntimeError(f"D2 generated artifact missing: {artifact}")
        hashes[field] = sha256_file(artifact)
    if bool(job["debug_raw"]):
        for field in ("node_file", "actuator_file"):
            raw = meta.get(field)
            if raw:
                artifact = path.parent / str(raw)
                if not artifact.is_file():
                    raise RuntimeError(f"D2 debug artifact missing: {artifact}")
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


def _complete(metadata_path: Path, expected_key: str) -> bool:
    if not metadata_path.is_file():
        return False
    try:
        meta = _read_json(metadata_path)
        if meta.get("data_contract") != "D2_CONTROLS_DISABLED_COMPACT_V2":
            return False
        if meta.get("generation_key_sha256") != expected_key:
            return False
        _, current_code = generation_key("code_probe", {})
        if meta.get("rtc_source_tree_sha256") != current_code:
            return False
        hashes = meta.get("generated_artifact_sha256")
        if not isinstance(hashes, dict) or not hashes:
            return False
        for field, expected in hashes.items():
            raw = meta.get(str(field))
            if not raw:
                return False
            artifact = metadata_path.parent / str(raw)
            if not artifact.is_file() or sha256_file(artifact) != str(expected):
                return False
        return True
    except Exception:
        return False


def _run_job(job: dict[str, object]) -> dict[str, object]:
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
    key = _stamp(result.metadata_path, job)
    return {
        "branch_id": result.branch_id,
        "metadata_path": result.metadata_path,
        "candidate_action_sha256": str(job["candidate_action_sha256"]),
        "generation_key_sha256": key,
        "checkpoint_minutes": int(job["checkpoint_minutes"]),
        "checkpoint_id": str(job["checkpoint_id"]),
        "event_id": str(job["event_id"]),
        "rainfall_group": str(job["rainfall_group"]),
        "scientific_split": str(job["scientific_split"]),
        "development_fold": str(job["development_fold"]),
        "flow_routing_error_pct": result.flow_routing_error_pct,
        "status": "completed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run authoritative code-bound resumable pre-lock D2 counterfactual branches"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--inp", help="default event INP; manifest inp_path takes precedence")
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
    if min(args.workers, args.swmm_threads_per_process, args.horizon_minutes, args.stride_seconds) <= 0:
        raise ValueError("workers/threads/horizon/stride must be positive")
    if (args.horizon_minutes * 60) % args.stride_seconds:
        raise ValueError("D2 horizon must align with stride")

    manifest = pd.read_csv(args.manifest)
    required = {
        "candidate_action_sha256",
        "candidate_settings_json",
        "checkpoint_minutes",
        "checkpoint_id",
        "scientific_split",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"D2 manifest missing required columns: {missing}")
    if (manifest["scientific_split"].astype(str) == "final").any():
        raise ValueError("rtc-run-probes refuses Final rows before Policy Lock")
    dedup_cols = ["candidate_action_sha256", "checkpoint_id"]
    if "event_id" in manifest.columns:
        dedup_cols.insert(0, "event_id")
    rows = manifest.drop_duplicates(dedup_cols).reset_index(drop=True)
    if args.limit is not None:
        rows = rows.head(args.limit)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runtime_dir = out / "_runtime_inp"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_cache: dict[str, str] = {}
    jobs: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        source_raw = row.get("inp_path", "")
        source = "" if pd.isna(source_raw) else str(source_raw).strip()
        if not source:
            source = str(args.inp or "")
        if not source or not Path(source).is_file():
            raise ValueError(f"D2 source INP missing: {source}")
        source_sha = sha256_file(source)
        if source_sha not in runtime_cache:
            runtime = runtime_dir / f"{source_sha[:20]}.d2.no_control.t{args.swmm_threads_per_process}.inp"
            build_runtime_inp(
                source,
                runtime,
                native_controls=False,
                swmm_threads=args.swmm_threads_per_process,
            )
            runtime_cache[source_sha] = str(runtime)
        runtime = runtime_cache[source_sha]
        event = str(row.get("event_id", "event"))
        action_sha = str(row["candidate_action_sha256"])
        checkpoint = int(row["checkpoint_minutes"])
        branch_id = f"{event}__t{checkpoint:04d}__{action_sha[:16]}"
        metadata = out / f"{branch_id}.json"
        job: dict[str, object] = {
            "runtime_inp": runtime,
            "runtime_inp_sha256": sha256_file(runtime),
            "source_inp_sha256": source_sha,
            "out_dir": str(out),
            "branch_id": branch_id,
            "candidate_action_sha256": action_sha,
            "candidate_settings_json": str(row["candidate_settings_json"]),
            "checkpoint_minutes": checkpoint,
            "checkpoint_id": str(row["checkpoint_id"]),
            "event_id": event,
            "rainfall_group": str(row.get("rainfall_group", "")),
            "scientific_split": str(row["scientific_split"]),
            "development_fold": str(row.get("development_fold", "")),
            "horizon_minutes": args.horizon_minutes,
            "stride_seconds": args.stride_seconds,
            "swmm_threads_per_process": args.swmm_threads_per_process,
            "debug_raw": args.debug_raw,
            "keep_engine_files": args.keep_engine_files,
        }
        expected_key, _, _ = _generation(job)
        if not args.no_resume and _complete(metadata, expected_key):
            results.append(
                {
                    "branch_id": branch_id,
                    "metadata_path": str(metadata),
                    "candidate_action_sha256": action_sha,
                    "generation_key_sha256": expected_key,
                    "checkpoint_minutes": checkpoint,
                    "checkpoint_id": job["checkpoint_id"],
                    "event_id": event,
                    "rainfall_group": job["rainfall_group"],
                    "scientific_split": job["scientific_split"],
                    "development_fold": job["development_fold"],
                    "flow_routing_error_pct": float("nan"),
                    "status": "resumed",
                }
            )
        else:
            jobs.append(job)
    if jobs:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            futures = [pool.submit(_run_job, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda r: str(r["branch_id"]))
    summary = out / "RUN_SUMMARY.csv"
    pd.DataFrame(results).to_csv(summary, index=False)
    print(
        json.dumps(
            {
                "contract": "D2_BATCH_CODE_BOUND_RESUME_V1",
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
    main()
