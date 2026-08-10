from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from .generation_contract import generation_key
from .inp_runtime import build_runtime_inp, sha256_file
from .preflight_identity_cache import PreflightIdentityCache, PreflightProgress
from .simulation_assets import (
    SimulationAssetRegistry,
    d2_identity_from_precomputed,
    register_stamped_d2_metadata_many,
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
        inp_sha256=str(job["runtime_inp_sha256"]),
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
    parser.add_argument(
        "--census-only",
        action="store_true",
        help="complete identity/endpoint/asset preflight and exit without launching SWMM",
    )
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
    progress = PreflightProgress(out / "PRECHECK_PROGRESS.json", total=len(rows))
    registry = SimulationAssetRegistry(args.asset_root) if args.asset_root else None
    registry_snapshot = (
        None if registry is None or args.no_resume else registry.preflight_snapshot()
    )

    def resolve_source(row: pd.Series) -> str:
        source_raw = row.get("inp_path", "")
        source = "" if pd.isna(source_raw) else str(source_raw).strip()
        if not source:
            source = str(args.inp or "")
        if not source or not Path(source).is_file():
            raise ValueError(f"D2 source INP missing: {source}")
        return str(Path(source).resolve())

    rows = rows.copy()
    rows["_resolved_source"] = [resolve_source(row) for _, row in rows.iterrows()]
    sort_columns = ["_resolved_source", "checkpoint_minutes", "candidate_action_sha256"]
    if "event_id" in rows.columns:
        sort_columns.insert(1, "event_id")
    rows = rows.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    event_checkpoints: dict[str, set[int]] = defaultdict(set)
    reference_checkpoints: dict[str, set[int]] = defaultdict(set)
    for _, row in rows.iterrows():
        event_checkpoints[str(row["_resolved_source"])].add(int(row["checkpoint_minutes"]) * 60)
        reference_checkpoints[str(Path(row["trajectory_metadata_path"]).resolve())].add(
            int(row["checkpoint_minutes"]) * 60
        )
    identity_cache = PreflightIdentityCache.build(
        event_paths_to_checkpoints=event_checkpoints,
        reference_paths_to_checkpoints=reference_checkpoints,
        cache_path=out / "PRECHECK_CACHE.json",
        progress=progress,
    )

    pre_jobs: list[dict[str, object]] = []
    endpoint_failures: list[dict[str, object]] = []
    exact_asset_hits = 0
    covering_trajectory_hits = 0
    exact_hits_missing_snapshots = 0
    for processed, (_, row) in enumerate(rows.iterrows(), start=1):
        source = str(row["_resolved_source"])
        event_context = identity_cache.event(source)
        event = str(row.get("event_id", "event"))
        action_sha = str(row["candidate_action_sha256"])
        checkpoint = int(row["checkpoint_minutes"])
        checkpoint_seconds = checkpoint * 60
        reference_path = str(Path(row["trajectory_metadata_path"]).resolve())
        reference_context = identity_cache.reference(reference_path)
        try:
            endpoint = event_context.endpoint_preflight(
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
            progress.update(
                stage="CANDIDATE_PREFLIGHT",
                processed=processed,
                events_cached=len(identity_cache.events),
                references_cached=len(identity_cache.references),
                checkpoints_cached=sum(
                    len(context.checkpoint_state_sha256_by_elapsed)
                    for context in identity_cache.references.values()
                ),
                endpoint_invalid=len(endpoint_failures),
                exact_asset_candidates=exact_asset_hits,
                covering_asset_candidates=covering_trajectory_hits,
            )
            continue
        sim_key, family_key, identity = d2_identity_from_precomputed(
            physical_network_sha256=event_context.physical_network_sha256,
            event_prefix_family_sha256=event_context.event_prefix_family_sha256,
            checkpoint_seconds=checkpoint_seconds,
            checkpoint_state_sha256_value=reference_context.checkpoint_state_sha256(
                checkpoint_seconds
            ),
            candidate_action_sha256=action_sha,
            swmm_engine_version=reference_context.swmm_engine_version,
            stride_seconds=args.stride_seconds,
            horizon_seconds=args.horizon_minutes * 60,
        )
        branch_id = f"{event}__t{checkpoint:04d}__{action_sha[:16]}"
        job: dict[str, object] = {
            "source": source,
            "source_inp_sha256": event_context.source_file.sha256,
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
            "reference_metadata_path": reference_context.metadata_path,
            **reference_context.lineage,
        }
        if registry_snapshot is not None:
            hit = registry_snapshot.lookup_exact(sim_key)
            if hit is not None and _supports_snapshots(hit.metadata_path, snapshots):
                exact_asset_hits += 1
                job["asset_hit_metadata_path"] = hit.metadata_path
            else:
                if hit is not None:
                    exact_hits_missing_snapshots += 1
                covering = registry_snapshot.lookup_covering(
                    family_key, horizon_seconds=args.horizon_minutes * 60
                )
                if covering is not None:
                    covering_trajectory_hits += 1
        pre_jobs.append(job)
        if processed == 1 or processed % 100 == 0 or processed == len(rows):
            progress.update(
                stage="CANDIDATE_PREFLIGHT",
                processed=processed,
                events_cached=len(identity_cache.events),
                references_cached=len(identity_cache.references),
                checkpoints_cached=sum(
                    len(context.checkpoint_state_sha256_by_elapsed)
                    for context in identity_cache.references.values()
                ),
                endpoint_invalid=len(endpoint_failures),
                exact_asset_candidates=exact_asset_hits,
                covering_asset_candidates=covering_trajectory_hits,
            )

    identity_cache.assert_unchanged()
    census = {
        "contract": "RTC_D2_PRE_RUN_CENSUS_V2_SNAPSHOT_AWARE",
        "manifest": str(Path(args.manifest).resolve()),
        "requested_rows": int(requested_rows),
        "unique_projected_actions": int(unique_total),
        "selected_unique_actions": len(rows),
        "deduplicated_rows": int(requested_rows - unique_total),
        "limit_applied": args.limit,
        "endpoint_invalid": len(endpoint_failures),
        "endpoint_failures": endpoint_failures,
        "asset_registry_enabled": registry is not None,
        "registry_snapshot": registry_snapshot is not None,
        "exact_asset_hits": int(exact_asset_hits),
        "exact_hits_missing_requested_snapshots": int(exact_hits_missing_snapshots),
        "covering_trajectory_hits": int(covering_trajectory_hits),
        "covering_reuse_scope": (
            "trajectory/timing prefix only unless an exact cumulative endpoint snapshot exists"
        ),
        "need_execution_before_local_resume": int(len(pre_jobs) - exact_asset_hits),
        "event_context_count": len(identity_cache.events),
        "reference_context_count": len(identity_cache.references),
        "checkpoint_context_count": int(
            sum(
                len(context.checkpoint_state_sha256_by_elapsed)
                for context in identity_cache.references.values()
            )
        ),
        "identity_contract": "RTC_SIMULATION_IDENTITY_V1_STATE_ACTION_ENGINE_BOUND",
        "horizon_minutes": int(args.horizon_minutes),
        "snapshot_horizons_minutes": list(snapshots),
        "stride_seconds": int(args.stride_seconds),
        "census_only": bool(args.census_only),
    }
    _write_census(census_path, census)
    progress.update(
        stage="CENSUS_WRITTEN",
        processed=len(rows),
        endpoint_invalid=len(endpoint_failures),
        exact_asset_candidates=exact_asset_hits,
        covering_asset_candidates=covering_trajectory_hits,
        event_context_count=len(identity_cache.events),
        reference_context_count=len(identity_cache.references),
        checkpoint_context_count=census["checkpoint_context_count"],
    )
    if endpoint_failures:
        raise ValueError(
            f"D2 preflight rejected {len(endpoint_failures)} branch identities before SWMM; "
            f"see {census_path}"
        )
    if args.census_only:
        print(json.dumps({"census": str(census_path), "census_only": True}, indent=2))
        return

    runtime_dir = out / "_runtime_inp"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_cache: dict[str, dict[str, str]] = {}
    jobs: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    pending_registry_paths: list[str] = []

    def flush_registry() -> None:
        if registry is not None and pending_registry_paths:
            register_stamped_d2_metadata_many(registry, pending_registry_paths)
            pending_registry_paths.clear()

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
            contract = build_runtime_inp(
                source,
                runtime,
                native_controls=False,
                swmm_threads=args.swmm_threads_per_process,
                source_sha256=source_sha,
            )
            runtime_cache[source_sha] = {
                "path": str(runtime.resolve()),
                "sha256": contract.runtime_sha256,
            }
        runtime = runtime_cache[source_sha]
        job["runtime_inp"] = runtime["path"]
        job["runtime_inp_sha256"] = runtime["sha256"]
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
                pending_registry_paths.append(str(metadata))
                if len(pending_registry_paths) >= 32:
                    flush_registry()
        else:
            jobs.append(job)

    # Detect a source/reference mutation that happened during runtime-INP preparation before
    # any worker can launch.  The identity cache remains the single content authority.
    identity_cache.assert_unchanged()
    if jobs:
        progress.update(stage="SWMM_EXECUTION", branches_total=len(jobs), branches_completed=0)
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            futures = [pool.submit(_run_job, job) for job in jobs]
            for completed, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                if registry is not None:
                    pending_registry_paths.append(str(result["metadata_path"]))
                    if len(pending_registry_paths) >= 32:
                        flush_registry()
                if completed == 1 or completed % 100 == 0 or completed == len(jobs):
                    progress.update(
                        stage="SWMM_EXECUTION",
                        branches_total=len(jobs),
                        branches_completed=completed,
                        registry_pending=len(pending_registry_paths),
                    )
    flush_registry()
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
