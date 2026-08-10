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
from .simulation_asset_types import (
    d3_identity_from_precomputed,
    register_stamped_d3_metadata_many,
)
from .simulation_assets import SimulationAssetRegistry

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
            "generation_contract": "RTC_GENERATION_KEY_V2_INPUT_CONFIG_BOUND",
            "generation_key_sha256": key,
            "rtc_source_tree_sha256": code_sha,
            "generation_lineage": lineage,
            "generated_artifact_sha256": hashes,
            "endpoint_preflight": job.get("endpoint_preflight"),
            "simulation_identity_contract": "RTC_SIMULATION_IDENTITY_V1_STATE_ACTION_ENGINE_BOUND",
            "simulation_identity_sha256": str(job["simulation_identity_sha256"]),
            "simulation_family_sha256": str(job["simulation_family_sha256"]),
            "simulation_identity": job["simulation_identity"],
            "asset_qualification": "VALID_REUSABLE",
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


def _result(job: dict[str, object], *, metadata_path: str, status: str, generation_key: str = "", flow_error: float = float("nan")) -> dict[str, object]:
    return {
        "event_id": job["event_id"],
        "rainfall_group": job["rainfall_group"],
        "scientific_split": job["scientific_split"],
        "development_fold": job["development_fold"],
        "checkpoint_id": job["checkpoint_id"],
        "data_role": job["data_role"],
        "pulse_actuator_id": job.get("pulse_actuator_id", ""),
        "pulse_delta": job.get("pulse_delta", 0.0),
        "sequence_sha256": job["sequence_sha256"],
        "simulation_identity_sha256": job["simulation_identity_sha256"],
        "simulation_family_sha256": job["simulation_family_sha256"],
        "generation_key_sha256": generation_key,
        "metadata_path": metadata_path,
        "flow_routing_error_pct": flow_error,
        "status": status,
    }


def _run(job: dict[str, object]) -> dict[str, object]:
    from .swmm_sequence import run_control_sequence_branch

    result = run_control_sequence_branch(
        inp_path=str(job["runtime_inp"]),
        inp_sha256=str(job["runtime_inp_sha256"]),
        checkpoint_minutes=int(job["checkpoint_minutes"]),
        settings_sequence=json.loads(str(job["settings_sequence_json"])),
        control_block_seconds=int(job["control_block_seconds"]),
        output_dir=str(job["out_dir"]),
        branch_id=str(job["branch_id"]),
        python_intervention_seconds=int(job["stride_seconds"]),
        reference_trajectory_metadata_path=str(job["reference_metadata_path"]),
    )
    key = _stamp(result.metadata_path, job)
    return _result(
        job,
        metadata_path=result.metadata_path,
        status="completed",
        generation_key=key,
        flow_error=result.flow_routing_error_pct,
    )


def run_d3_batch_main() -> None:
    parser = argparse.ArgumentParser(
        description="Run resumable/reusable D3 sequences with exact No-control prefix verification"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--control-block-seconds", type=int, required=True)
    parser.add_argument("--stride-seconds", type=int, default=300)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--swmm-threads-per-process", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--asset-root", help="local cross-directory simulation asset registry")
    parser.add_argument("--census-out", help="pre-run request/asset/endpoint census JSON")
    parser.add_argument(
        "--census-only",
        action="store_true",
        help="complete identity/endpoint/asset preflight and exit without launching SWMM",
    )
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

    requested_rows = len(frame)
    dedup = frame.drop_duplicates(["sequence_sha256", "checkpoint_id"]).reset_index(drop=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    census = Path(args.census_out) if args.census_out else out / "REQUEST_CENSUS.json"
    progress = PreflightProgress(out / "PRECHECK_PROGRESS.json", total=len(dedup))
    registry = SimulationAssetRegistry(args.asset_root) if args.asset_root else None
    registry_snapshot = (
        None if registry is None or args.no_resume else registry.preflight_snapshot()
    )
    dedup = dedup.copy()
    dedup["_resolved_source"] = [
        str(Path(source).resolve())
        for source in dedup["inp_path"].astype(str)
    ]
    for source in dedup["_resolved_source"]:
        if not Path(source).is_file():
            raise ValueError(f"D3 source INP missing: {source}")
    sort_columns = ["_resolved_source", "checkpoint_minutes", "sequence_sha256"]
    if "event_id" in dedup.columns:
        sort_columns.insert(1, "event_id")
    dedup = dedup.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    event_checkpoints: dict[str, set[int]] = defaultdict(set)
    reference_checkpoints: dict[str, set[int]] = defaultdict(set)
    for _, row in dedup.iterrows():
        checkpoint_seconds = int(row["checkpoint_minutes"]) * 60
        event_checkpoints[str(row["_resolved_source"])].add(checkpoint_seconds)
        reference_checkpoints[str(Path(row["trajectory_metadata_path"]).resolve())].add(
            checkpoint_seconds
        )
    identity_cache = PreflightIdentityCache.build(
        event_paths_to_checkpoints=event_checkpoints,
        reference_paths_to_checkpoints=reference_checkpoints,
        cache_path=out / "PRECHECK_CACHE.json",
        progress=progress,
    )

    endpoint_failures: list[dict[str, object]] = []
    preflight: dict[int, dict[str, object]] = {}
    asset_hits = 0
    for processed, (idx, row) in enumerate(dedup.iterrows(), start=1):
        source = str(row["_resolved_source"])
        event_context = identity_cache.event(source)
        sequence = json.loads(str(row["settings_sequence_json"]))
        horizon_seconds = len(sequence) * args.control_block_seconds
        checkpoint_seconds = int(row["checkpoint_minutes"]) * 60
        reference_path = str(Path(row["trajectory_metadata_path"]).resolve())
        reference_context = identity_cache.reference(reference_path)
        try:
            endpoint = event_context.endpoint_preflight(
                checkpoint_seconds=checkpoint_seconds,
                horizon_seconds=horizon_seconds,
            )
        except ValueError as exc:
            endpoint_failures.append(
                {
                    "event_id": str(row.get("event_id", "")),
                    "checkpoint_id": str(row["checkpoint_id"]),
                    "sequence_sha256": str(row["sequence_sha256"]),
                    "error": str(exc),
                }
            )
            progress.update(
                stage="CANDIDATE_PREFLIGHT",
                processed=processed,
                endpoint_invalid=len(endpoint_failures),
                exact_asset_candidates=asset_hits,
            )
            continue
        sim_key, family_key, identity = d3_identity_from_precomputed(
            physical_network_sha256=event_context.physical_network_sha256,
            event_prefix_family_sha256=event_context.event_prefix_family_sha256,
            checkpoint_seconds=checkpoint_seconds,
            checkpoint_state_sha256_value=reference_context.checkpoint_state_sha256(
                checkpoint_seconds
            ),
            sequence_sha256=str(row["sequence_sha256"]),
            swmm_engine_version=reference_context.swmm_engine_version,
            stride_seconds=args.stride_seconds,
            control_block_seconds=args.control_block_seconds,
            horizon_seconds=horizon_seconds,
        )
        hit = None if registry_snapshot is None else registry_snapshot.lookup_exact(sim_key)
        if hit is not None:
            asset_hits += 1
        preflight[idx] = {
            "source": source,
            "source_inp_sha256": event_context.source_file.sha256,
            "sequence_length": len(sequence),
            "endpoint": endpoint,
            "simulation_identity_sha256": sim_key,
            "simulation_family_sha256": family_key,
            "simulation_identity": identity,
            "asset_hit_metadata_path": None if hit is None else hit.metadata_path,
            "reference_metadata_path": reference_context.metadata_path,
            **reference_context.lineage,
        }
        if processed == 1 or processed % 100 == 0 or processed == len(dedup):
            progress.update(
                stage="CANDIDATE_PREFLIGHT",
                processed=processed,
                endpoint_invalid=len(endpoint_failures),
                exact_asset_candidates=asset_hits,
                events_cached=len(identity_cache.events),
                references_cached=len(identity_cache.references),
                checkpoints_cached=sum(
                    len(context.checkpoint_state_sha256_by_elapsed)
                    for context in identity_cache.references.values()
                ),
            )

    identity_cache.assert_unchanged()
    census_payload = {
        "contract": "RTC_D3_PRE_RUN_CENSUS_V2_ASSET_AWARE",
        "requested_rows": int(requested_rows),
        "unique_sequences": len(dedup),
        "deduplicated_rows": int(requested_rows - len(dedup)),
        "endpoint_invalid": len(endpoint_failures),
        "endpoint_failures": endpoint_failures,
        "asset_registry_enabled": registry is not None,
        "registry_snapshot": registry_snapshot is not None,
        "exact_asset_hits": int(asset_hits),
        "need_execution_before_local_resume": int(len(preflight) - asset_hits),
        "event_context_count": len(identity_cache.events),
        "reference_context_count": len(identity_cache.references),
        "checkpoint_context_count": int(
            sum(
                len(context.checkpoint_state_sha256_by_elapsed)
                for context in identity_cache.references.values()
            )
        ),
        "census_only": bool(args.census_only),
    }
    census.parent.mkdir(parents=True, exist_ok=True)
    census.write_text(json.dumps(census_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    progress.update(
        stage="CENSUS_WRITTEN",
        processed=len(dedup),
        endpoint_invalid=len(endpoint_failures),
        exact_asset_candidates=asset_hits,
    )
    if endpoint_failures:
        raise ValueError(
            f"D3 endpoint preflight rejected {len(endpoint_failures)} sequences before SWMM; see {census}"
        )
    if args.census_only:
        print(json.dumps({"census": str(census), "census_only": True}, indent=2))
        return

    runtime_dir = out / "_runtime_inp"
    runtime_dir.mkdir(exist_ok=True)
    runtime_cache: dict[str, dict[str, str]] = {}
    jobs: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    pending_registry_paths: list[str] = []

    def flush_registry() -> None:
        if registry is not None and pending_registry_paths:
            register_stamped_d3_metadata_many(registry, pending_registry_paths)
            pending_registry_paths.clear()

    for idx, row in dedup.iterrows():
        info = preflight[idx]
        event = str(row.get("event_id", "event"))
        checkpoint = int(row["checkpoint_minutes"])
        seq = str(row["sequence_sha256"])
        branch_id = f"{event}__t{checkpoint:04d}__seq_{seq[:16]}"
        job: dict[str, object] = {
            "branch_id": branch_id,
            "event_id": event,
            "rainfall_group": str(row.get("rainfall_group", "")),
            "scientific_split": str(row.get("scientific_split", "")),
            "development_fold": str(row.get("development_fold", "")),
            "checkpoint_id": str(row["checkpoint_id"]),
            "checkpoint_minutes": checkpoint,
            "data_role": str(row.get("data_role", "D3_MULTI_ACTUATOR_ROLLOUT")),
            "pulse_actuator_id": str(row.get("pulse_actuator_id", "")),
            "pulse_delta": float(row.get("pulse_delta", 0.0)),
            "sequence_sha256": seq,
            "settings_sequence_json": str(row["settings_sequence_json"]),
            "control_block_seconds": args.control_block_seconds,
            "stride_seconds": args.stride_seconds,
            "swmm_threads_per_process": args.swmm_threads_per_process,
            "endpoint_preflight": info["endpoint"],
            "simulation_identity_sha256": info["simulation_identity_sha256"],
            "simulation_family_sha256": info["simulation_family_sha256"],
            "simulation_identity": info["simulation_identity"],
            **{k: v for k, v in info.items() if k.startswith("reference_")},
        }
        asset_path = info.get("asset_hit_metadata_path")
        if asset_path:
            results.append(_result(job, metadata_path=str(asset_path), status="asset_reused"))
            continue
        source = str(info["source"])
        source_sha = str(info["source_inp_sha256"])
        if source_sha not in runtime_cache:
            runtime = runtime_dir / f"{source_sha[:16]}.no_control.t{args.swmm_threads_per_process}.inp"
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
        metadata = out / f"{branch_id}.json"
        expected_key, _, _ = _generation(job)
        if not args.no_resume and _complete(metadata, expected_key):
            results.append(
                _result(
                    job,
                    metadata_path=str(metadata),
                    status="resumed",
                    generation_key=expected_key,
                )
            )
            if registry is not None:
                pending_registry_paths.append(str(metadata))
                if len(pending_registry_paths) >= 32:
                    flush_registry()
        else:
            jobs.append(job)
    identity_cache.assert_unchanged()
    if jobs:
        progress.update(stage="SWMM_EXECUTION", branches_total=len(jobs), branches_completed=0)
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            futures = [pool.submit(_run, job) for job in jobs]
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
    summary = out / "D3_RUN_SUMMARY.csv"
    pd.DataFrame(results).sort_values(
        ["event_id", "checkpoint_id", "sequence_sha256"]
    ).to_csv(summary, index=False)
    print(
        json.dumps(
            {
                "contract": "D3_BATCH_PREFIX_VERIFIED_ASSET_REUSE_V4",
                "branches": len(results),
                "computed": len(jobs),
                "local_resumed": sum(r["status"] == "resumed" for r in results),
                "asset_reused": sum(r["status"] == "asset_reused" for r in results),
                "workers": min(args.workers, max(1, len(jobs))),
                "census": str(census),
                "summary": str(summary),
                "asset_root": None if registry is None else str(registry.root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    run_d3_batch_main()
