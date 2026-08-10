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
from .simulation_assets import (
    SimulationAssetRegistry,
    assert_endpoint_available,
    d2_identity,
    register_d2_metadata,
)


D2_DATA_CONTRACT = "D2_CONTROLS_DISABLED_COMPACT_V3_PREFIX_VERIFIED"


def _read_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return value


def _parse_snapshot_horizons(
    raw: str | None, *, horizon_minutes: int, stride_seconds: int
) -> tuple[int, ...]:
    if not raw:
        return ()
    values = tuple(sorted({int(token.strip()) for token in raw.split(",") if token.strip()}))
    for value in values:
        if value <= 0 or value > horizon_minutes:
            raise ValueError("snapshot horizon must be positive and <= the executed D2 horizon")
        if (value * 60) % stride_seconds:
            raise ValueError("snapshot horizons must align with D2 stride")
    return values


def _generation(job: dict[str, object]) -> tuple[str, str, dict[str, object]]:
    payload = {
        "data_contract": D2_DATA_CONTRACT,
        "event_id": str(job["event_id"]),
        "rainfall_group": str(job["rainfall_group"]),
        "scientific_split": str(job["scientific_split"]),
        "development_fold": str(job["development_fold"]),
        "checkpoint_id": str(job["checkpoint_id"]),
        "checkpoint_minutes": int(job["checkpoint_minutes"]),
        "candidate_action_sha256": str(job["candidate_action_sha256"]),
        "source_inp_sha256": str(job["source_inp_sha256"]),
        "runtime_inp_sha256": str(job["runtime_inp_sha256"]),
        "reference_metadata_sha256": str(job["reference_metadata_sha256"]),
        "reference_compact_sha256": str(job["reference_compact_sha256"]),
        "reference_swmm_engine_version": str(job["reference_swmm_engine_version"]),
        "horizon_minutes": int(job["horizon_minutes"]),
        "snapshot_horizons_minutes": list(job["snapshot_horizons_minutes"]),
        "stride_seconds": int(job["stride_seconds"]),
        "swmm_threads_per_process": int(job["swmm_threads_per_process"]),
        "debug_raw": bool(job["debug_raw"]),
    }
    key, code_sha = generation_key("d2_counterfactual", payload)
    return key, code_sha, payload


def _snapshot_hashes(meta_path: Path, meta: dict[str, object]) -> dict[str, str]:
    raw = meta.get("horizon_snapshot_files")
    if not isinstance(raw, dict):
        return {}
    hashes: dict[str, str] = {}
    for minutes, name in raw.items():
        artifact = meta_path.parent / str(name)
        if not artifact.is_file():
            raise RuntimeError(f"D2 horizon snapshot missing: {artifact}")
        hashes[str(minutes)] = sha256_file(artifact)
    return hashes


def _supports_snapshots(metadata_path: str | Path, required: tuple[int, ...]) -> bool:
    if not required:
        return True
    try:
        path = Path(metadata_path)
        meta = _read_json(path)
        files = meta.get("horizon_snapshot_files")
        hashes = meta.get("generated_horizon_snapshot_sha256")
        if not isinstance(files, dict):
            return False
        for minutes in required:
            name = files.get(str(minutes))
            if not name:
                return False
            artifact = path.parent / str(name)
            if not artifact.is_file():
                return False
            if isinstance(hashes, dict):
                expected = hashes.get(str(minutes))
                if expected and sha256_file(artifact) != str(expected):
                    return False
        return True
    except Exception:
        return False


def _stamp(metadata_path: str | Path, job: dict[str, object]) -> str:
    path = Path(metadata_path)
    meta = _read_json(path)
    verification = meta.get("same_prefix_verification")
    if not isinstance(verification, dict) or verification.get("passed") is not True:
        raise RuntimeError("Formal D2 branch lacks successful exact No-control prefix verification")
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
    snapshot_hashes = _snapshot_hashes(path, meta)
    required = {str(v) for v in job["snapshot_horizons_minutes"]}
    if not required.issubset(snapshot_hashes):
        raise RuntimeError("D2 generator failed to persist all requested exact horizon snapshots")
    meta.update(
        {
            "generation_contract": "RTC_GENERATION_KEY_V2_INPUT_CONFIG_BOUND",
            "generation_key_sha256": key,
            "rtc_source_tree_sha256": code_sha,
            "generation_lineage": lineage,
            "generated_artifact_sha256": hashes,
            "generated_horizon_snapshot_sha256": snapshot_hashes,
            "simulation_identity_contract": "RTC_SIMULATION_IDENTITY_V1_STATE_ACTION_ENGINE_BOUND",
            "simulation_identity_sha256": str(job["simulation_identity_sha256"]),
            "simulation_family_sha256": str(job["simulation_family_sha256"]),
            "simulation_identity": job["simulation_identity"],
            "endpoint_preflight": job["endpoint_preflight"],
            "asset_qualification": "VALID_REUSABLE",
        }
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return key


def _complete(
    metadata_path: Path, expected_key: str, required_snapshots: tuple[int, ...]
) -> bool:
    if not metadata_path.is_file():
        return False
    try:
        meta = _read_json(metadata_path)
        if meta.get("data_contract") != D2_DATA_CONTRACT:
            return False
        verification = meta.get("same_prefix_verification")
        if not isinstance(verification, dict) or verification.get("passed") is not True:
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
        return _supports_snapshots(metadata_path, required_snapshots)
    except Exception:
        return False


def _result_record(
    *,
    job: dict[str, object],
    metadata_path: str,
    status: str,
    flow_routing_error_pct: float = float("nan"),
    generation_key_sha256: str = "",
) -> dict[str, object]:
    return {
        "branch_id": str(job["branch_id"]),
        "metadata_path": str(metadata_path),
        "candidate_action_sha256": str(job["candidate_action_sha256"]),
        "generation_key_sha256": generation_key_sha256,
        "simulation_identity_sha256": str(job["simulation_identity_sha256"]),
        "simulation_family_sha256": str(job["simulation_family_sha256"]),
        "checkpoint_minutes": int(job["checkpoint_minutes"]),
        "checkpoint_id": str(job["checkpoint_id"]),
        "event_id": str(job["event_id"]),
        "rainfall_group": str(job["rainfall_group"]),
        "scientific_split": str(job["scientific_split"]),
        "development_fold": str(job["development_fold"]),
        "flow_routing_error_pct": flow_routing_error_pct,
        "status": status,
    }


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
        reference_trajectory_metadata_path=str(job["reference_metadata_path"]),
        save_raw_csv=bool(job["debug_raw"]),
        keep_engine_files=bool(job["keep_engine_files"]),
        snapshot_horizons_minutes=tuple(job["snapshot_horizons_minutes"]),
    )
    key = _stamp(result.metadata_path, job)
    return _result_record(
        job=job,
        metadata_path=result.metadata_path,
        status="completed",
        flow_routing_error_pct=result.flow_routing_error_pct,
        generation_key_sha256=key,
    )


def _write_census(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run authoritative D2 branches with exact-prefix verification, endpoint preflight "
            "and optional cross-directory local simulation-asset reuse"
        )
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
    parser.add_argument(
        "--asset-root",
        help=(
            "local large-data simulation asset store; exact physical-state/action identities "
            "may be reused across output directories and studies"
        ),
    )
    parser.add_argument(
        "--snapshot-horizons-minutes",
        help=(
            "comma-separated exact cumulative SWMM endpoint snapshots captured during this "
            "single run, e.g. 210,240,300,360"
        ),
    )
    parser.add_argument("--census-out", help="pre-run request/dedup/cache/endpoint census JSON")
    args = parser.parse_args()
    if min(args.workers, args.swmm_threads_per_process, args.horizon_minutes, args.stride_seconds) <= 0:
        raise ValueError("workers/threads/horizon/stride must be positive")
    if (args.horizon_minutes * 60) % args.stride_seconds:
        raise ValueError("D2 horizon must align with stride")
    snapshots = _parse_snapshot_horizons(
        args.snapshot_horizons_minutes,
        horizon_minutes=args.horizon_minutes,
        stride_seconds=args.stride_seconds,
    )

    manifest = pd.read_csv(args.manifest)
    required = {
        "candidate_action_sha256",
        "candidate_settings_json",
        "checkpoint_minutes",
        "checkpoint_id",
        "trajectory_metadata_path",
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
    requested_rows = len(manifest)
    unique_rows = manifest.drop_duplicates(dedup_cols).reset_index(drop=True)
    unique_total = len(unique_rows)
    rows = unique_rows if args.limit is None else unique_rows.head(args.limit)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    census_path = Path(args.census_out) if args.census_out else out / "REQUEST_CENSUS.json"
    registry = SimulationAssetRegistry(args.asset_root) if args.asset_root else None

    pre_jobs: list[dict[str, object]] = []
    endpoint_failures: list[dict[str, object]] = []
    exact_asset_hits = 0
    covering_trajectory_hits = 0
    exact_hits_missing_snapshots = 0
    for _, row in rows.iterrows():
        source_raw = row.get("inp_path", "")
        source = "" if pd.isna(source_raw) else str(source_raw).strip()
        if not source:
            source = str(args.inp or "")
        if not source or not Path(source).is_file():
            raise ValueError(f"D2 source INP missing: {source}")
        event = str(row.get("event_id", "event"))
        action_sha = str(row["candidate_action_sha256"])
        checkpoint = int(row["checkpoint_minutes"])
        checkpoint_seconds = checkpoint * 60
        reference_path = str(row["trajectory_metadata_path"])
        reference_lineage = reference_trajectory_lineage(reference_path)
        try:
            endpoint = assert_endpoint_available(
                source,
                checkpoint_seconds=checkpoint_seconds,
                horizon_seconds=args.horizon_minutes * 60,
            )
        except ValueError as exc:
            endpoint_failures.append(
                {
                    "event_id": event,
                    "checkpoint_id": str(row["checkpoint_id"]),
                    "checkpoint_minutes": checkpoint,
                    "candidate_action_sha256": action_sha,
                    "error": str(exc),
                }
            )
            continue
        sim_key, family_key, identity = d2_identity(
            inp_path=source,
            reference_metadata_path=reference_path,
            checkpoint_seconds=checkpoint_seconds,
            candidate_action_sha256=action_sha,
            swmm_engine_version=str(reference_lineage["reference_swmm_engine_version"]),
            stride_seconds=args.stride_seconds,
            horizon_seconds=args.horizon_minutes * 60,
        )
        branch_id = f"{event}__t{checkpoint:04d}__{action_sha[:16]}"
        job: dict[str, object] = {
            "source": source,
            "source_inp_sha256": sha256_file(source),
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
            "snapshot_horizons_minutes": snapshots,
            "stride_seconds": args.stride_seconds,
            "swmm_threads_per_process": args.swmm_threads_per_process,
            "debug_raw": args.debug_raw,
            "keep_engine_files": args.keep_engine_files,
            "simulation_identity_sha256": sim_key,
            "simulation_family_sha256": family_key,
            "simulation_identity": identity,
            "endpoint_preflight": endpoint,
            **reference_lineage,
        }
        if registry is not None and not args.no_resume:
            hit = registry.lookup_exact(sim_key)
            if hit is not None and _supports_snapshots(hit.metadata_path, snapshots):
                exact_asset_hits += 1
                job["asset_hit_metadata_path"] = hit.metadata_path
            else:
                if hit is not None:
                    exact_hits_missing_snapshots += 1
                covering = registry.lookup_covering(
                    family_key, horizon_seconds=args.horizon_minutes * 60
                )
                if covering is not None:
                    covering_trajectory_hits += 1
        pre_jobs.append(job)

    census = {
        "contract": "RTC_D2_PRE_RUN_CENSUS_V2_SNAPSHOT_AWARE",
        "manifest": str(Path(args.manifest).resolve()),
        "requested_rows": int(requested_rows),
        "unique_projected_actions": int(unique_total),
        "selected_unique_actions": int(len(rows)),
        "deduplicated_rows": int(requested_rows - unique_total),
        "limit_applied": args.limit,
        "endpoint_invalid": int(len(endpoint_failures)),
        "endpoint_failures": endpoint_failures,
        "asset_registry_enabled": registry is not None,
        "exact_asset_hits": int(exact_asset_hits),
        "exact_hits_missing_requested_snapshots": int(exact_hits_missing_snapshots),
        "covering_trajectory_hits": int(covering_trajectory_hits),
        "covering_reuse_scope": (
            "trajectory/timing prefix only unless an exact cumulative endpoint snapshot exists"
        ),
        "need_execution_before_local_resume": int(len(pre_jobs) - exact_asset_hits),
        "horizon_minutes": int(args.horizon_minutes),
        "snapshot_horizons_minutes": list(snapshots),
        "stride_seconds": int(args.stride_seconds),
    }
    _write_census(census_path, census)
    if endpoint_failures:
        raise ValueError(
            f"D2 preflight rejected {len(endpoint_failures)} branch identities before SWMM; "
            f"see {census_path}"
        )

    runtime_dir = out / "_runtime_inp"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_cache: dict[str, str] = {}
    jobs: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    for job in pre_jobs:
        if job.get("asset_hit_metadata_path"):
            results.append(
                _result_record(
                    job=job,
                    metadata_path=str(job["asset_hit_metadata_path"]),
                    status="asset_reused",
                )
            )
            continue
        source = str(job["source"])
        source_sha = str(job["source_inp_sha256"])
        if source_sha not in runtime_cache:
            runtime = runtime_dir / (
                f"{source_sha[:20]}.d2.no_control.t{args.swmm_threads_per_process}.inp"
            )
            build_runtime_inp(
                source,
                runtime,
                native_controls=False,
                swmm_threads=args.swmm_threads_per_process,
            )
            runtime_cache[source_sha] = str(runtime)
        runtime = runtime_cache[source_sha]
        job["runtime_inp"] = runtime
        job["runtime_inp_sha256"] = sha256_file(runtime)
        job["out_dir"] = str(out)
        metadata = out / f"{job['branch_id']}.json"
        expected_key, _, _ = _generation(job)
        if not args.no_resume and _complete(metadata, expected_key, snapshots):
            results.append(
                _result_record(
                    job=job,
                    metadata_path=str(metadata),
                    status="resumed",
                    generation_key_sha256=expected_key,
                )
            )
            if registry is not None:
                register_d2_metadata(registry, metadata)
        else:
            jobs.append(job)

    if jobs:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            futures = [pool.submit(_run_job, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if registry is not None:
                    register_d2_metadata(registry, str(result["metadata_path"]))
    results.sort(key=lambda r: str(r["branch_id"]))
    summary = out / "RUN_SUMMARY.csv"
    pd.DataFrame(results).to_csv(summary, index=False)
    print(
        json.dumps(
            {
                "contract": "D2_BATCH_PREFIX_VERIFIED_ASSET_REUSE_V4_SNAPSHOT_AWARE",
                "branches": len(results),
                "computed": len(jobs),
                "local_resumed": sum(r["status"] == "resumed" for r in results),
                "asset_reused": sum(r["status"] == "asset_reused" for r in results),
                "workers": min(args.workers, max(1, len(jobs))),
                "snapshot_horizons_minutes": list(snapshots),
                "census": str(census_path),
                "summary": str(summary),
                "asset_root": None if registry is None else str(registry.root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
