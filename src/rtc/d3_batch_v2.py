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
from .simulation_asset_types import d3_identity, register_d3_metadata
from .simulation_assets import SimulationAssetRegistry, assert_endpoint_available


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
    registry = SimulationAssetRegistry(args.asset_root) if args.asset_root else None
    endpoint_failures: list[dict[str, object]] = []
    preflight: dict[int, dict[str, object]] = {}
    asset_hits = 0
    for idx, row in dedup.iterrows():
        source = str(row["inp_path"])
        if not Path(source).is_file():
            raise ValueError(f"D3 source INP missing: {source}")
        sequence = json.loads(str(row["settings_sequence_json"]))
        horizon_seconds = len(sequence) * args.control_block_seconds
        checkpoint_seconds = int(row["checkpoint_minutes"]) * 60
        reference_path = str(row["trajectory_metadata_path"])
        reference_lineage = reference_trajectory_lineage(reference_path)
        try:
            endpoint = assert_endpoint_available(
                source,
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
            continue
        sim_key, family_key, identity = d3_identity(
            inp_path=source,
            reference_metadata_path=reference_path,
            checkpoint_seconds=checkpoint_seconds,
            sequence_sha256=str(row["sequence_sha256"]),
            swmm_engine_version=str(reference_lineage["reference_swmm_engine_version"]),
            stride_seconds=args.stride_seconds,
            control_block_seconds=args.control_block_seconds,
            horizon_seconds=horizon_seconds,
        )
        hit = None if registry is None or args.no_resume else registry.lookup_exact(sim_key)
        if hit is not None:
            asset_hits += 1
        preflight[idx] = {
            "sequence_length": len(sequence),
            "endpoint": endpoint,
            "simulation_identity_sha256": sim_key,
            "simulation_family_sha256": family_key,
            "simulation_identity": identity,
            "asset_hit_metadata_path": None if hit is None else hit.metadata_path,
            **reference_lineage,
        }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    census = out / "REQUEST_CENSUS.json"
    census.write_text(
        json.dumps(
            {
                "contract": "RTC_D3_PRE_RUN_CENSUS_V2_ASSET_AWARE",
                "requested_rows": int(requested_rows),
                "unique_sequences": int(len(dedup)),
                "deduplicated_rows": int(requested_rows - len(dedup)),
                "endpoint_invalid": len(endpoint_failures),
                "endpoint_failures": endpoint_failures,
                "asset_registry_enabled": registry is not None,
                "exact_asset_hits": int(asset_hits),
                "need_execution_before_local_resume": int(len(preflight) - asset_hits),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if endpoint_failures:
        raise ValueError(
            f"D3 endpoint preflight rejected {len(endpoint_failures)} sequences before SWMM; see {census}"
        )

    runtime_dir = out / "_runtime_inp"
    runtime_dir.mkdir(exist_ok=True)
    runtime_cache: dict[str, str] = {}
    jobs: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
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
        job["runtime_inp"] = runtime
        job["runtime_inp_sha256"] = sha256_file(runtime)
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
                register_d3_metadata(registry, metadata)
        else:
            jobs.append(job)
    if jobs:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            futures = [pool.submit(_run, job) for job in jobs]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if registry is not None:
                    register_d3_metadata(registry, str(result["metadata_path"]))
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
