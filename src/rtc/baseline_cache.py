from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .baselines import FIXED_BASELINE_IDS, fixed_baseline_controller
from .closed_loop import run_authoritative_closed_loop
from .formalize_run import formalize_run
from .generation_contract import canonical_json, generation_key
from .inp_lineage import physical_contract_sha256
from .inp_runtime import build_runtime_inp, section_has_payload, sha256_file


CACHE_CONTRACT = "FIXED_BASELINE_CACHE_V2_CODE_BOUND"


def baseline_cache_key(
    *,
    source_inp_sha256: str,
    physical_network_sha256: str,
    strategy: str,
    model_step_seconds: int,
    control_update_seconds: int,
    record_stride_seconds: int,
    control_start_minutes: int,
    swmm_threads_per_process: int,
) -> str:
    key, _ = generation_key(
        "fixed_baseline",
        {
            "cache_contract": CACHE_CONTRACT,
            "source_inp_sha256": source_inp_sha256,
            "physical_network_sha256": physical_network_sha256,
            "strategy": strategy,
            "model_step_seconds": int(model_step_seconds),
            "control_update_seconds": int(control_update_seconds),
            "record_stride_seconds": int(record_stride_seconds),
            "control_start_minutes": int(control_start_minutes),
            "swmm_threads_per_process": int(swmm_threads_per_process),
            "exact_global_peak_in_main": False,
        },
    )
    return key


def _read_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return value


def _decision_rows(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if raw.strip():
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("decision log row must be a JSON object")
            rows.append(value)
    return rows


def validate_fixed_baseline_run(
    *,
    strategy: str,
    main_metadata_path: str | Path,
    source_physical_sha256: str,
) -> dict[str, object]:
    """Fail closed if an executed baseline does not match its scientific semantics."""

    meta_path = Path(main_metadata_path)
    meta = _read_json(meta_path)
    if meta.get("data_contract") != "CLOSED_LOOP_COMPACT_V2":
        raise ValueError("baseline cache requires CLOSED_LOOP_COMPACT_V2 evidence")
    runtime_inp = Path(str(meta["inp_path"]))
    if not runtime_inp.is_file():
        raise ValueError("baseline runtime INP disappeared")
    if physical_contract_sha256(runtime_inp) != source_physical_sha256:
        raise ValueError("baseline runtime changed the physical hydraulic network")
    controls = bool(section_has_payload(runtime_inp, "CONTROLS"))
    decisions = _decision_rows(meta_path.parent / str(meta["decision_file"]))
    controller_present = bool(meta.get("controller_present"))

    if strategy == "internal_rtc":
        if not controls or controller_present or decisions:
            raise ValueError("Internal-RTC must retain native controls and have no Python writes")
    elif strategy == "no_control":
        if controls or controller_present or decisions:
            raise ValueError("No-control must disable native controls and have no Python writes")
    elif strategy == "hold":
        # Retained only for engineering/debug regression tests; not a Formal comparator.
        if controls or not controller_present or not decisions:
            raise ValueError("Hold must use Python writes on a controls-disabled runtime")
        first = decisions[0].get("settings")
        if not isinstance(first, dict):
            raise ValueError("Hold decision log lacks settings")
        for row in decisions:
            if row.get("source") != "FROZEN_HOLD" or row.get("settings") != first:
                raise ValueError("Hold must freeze exactly one actuator vector for the whole control period")
    elif strategy in {"all_open", "all_closed"}:
        expected = 1.0 if strategy == "all_open" else 0.0
        expected_source = "ALL_OPEN" if strategy == "all_open" else "ALL_CLOSED"
        if controls or not controller_present or not decisions:
            raise ValueError(f"{strategy} must use Python writes on a controls-disabled runtime")
        for row in decisions:
            if row.get("source") != expected_source:
                raise ValueError(f"{strategy} decision source mismatch")
            settings = row.get("settings")
            if not isinstance(settings, dict) or not settings:
                raise ValueError(f"{strategy} decision log lacks settings")
            if any(abs(float(v) - expected) > 1e-9 for v in settings.values()):
                raise ValueError(f"{strategy} did not command every eligible actuator to {expected}")
    else:
        raise ValueError(f"unsupported fixed baseline: {strategy}")

    compact = meta_path.parent / str(meta["compact_file"])
    stats = meta_path.parent / str(meta["node_statistics_file"])
    decision = meta_path.parent / str(meta["decision_file"])
    for p in (compact, stats, decision):
        if not p.is_file():
            raise ValueError(f"baseline evidence missing: {p}")
    return {
        "strategy": strategy,
        "native_controls_enabled": controls,
        "controller_present": controller_present,
        "decision_count": len(decisions),
        "runtime_inp": str(runtime_inp.resolve()),
        "runtime_inp_sha256": sha256_file(runtime_inp),
        "compact_path": str(compact.resolve()),
        "compact_sha256": sha256_file(compact),
        "node_statistics_path": str(stats.resolve()),
        "node_statistics_sha256": sha256_file(stats),
        "decision_log_path": str(decision.resolve()),
        "decision_log_sha256": sha256_file(decision),
    }


def _cache_complete(
    sidecar_path: Path, expected_key: str, *, require_formal: bool
) -> dict[str, object] | None:
    if not sidecar_path.is_file():
        return None
    try:
        sidecar = _read_json(sidecar_path)
        if sidecar.get("contract") != CACHE_CONTRACT:
            return None
        if sidecar.get("cache_key_sha256") != expected_key:
            return None
        # Recomputing the current key already binds the installed source tree. The explicit
        # code hash is also kept in evidence so stale reuse is auditable rather than opaque.
        _, current_code_sha = generation_key("code_probe", {})
        if sidecar.get("rtc_source_tree_sha256") != current_code_sha:
            return None
        main = Path(str(sidecar["main_metadata_path"]))
        if not main.is_file() or sha256_file(main) != str(sidecar["main_metadata_sha256"]):
            return None
        evidence = sidecar.get("evidence")
        if not isinstance(evidence, dict):
            return None
        for path_key, hash_key in (
            ("runtime_inp", "runtime_inp_sha256"),
            ("compact_path", "compact_sha256"),
            ("node_statistics_path", "node_statistics_sha256"),
            ("decision_log_path", "decision_log_sha256"),
        ):
            p = Path(str(evidence[path_key]))
            if not p.is_file() or sha256_file(p) != str(evidence[hash_key]):
                return None
        if require_formal:
            formal = Path(str(sidecar.get("formal_manifest_path", "")))
            if not formal.is_file() or sha256_file(formal) != str(
                sidecar.get("formal_manifest_sha256", "")
            ):
                return None
        return sidecar
    except Exception:
        return None


def _runtime_inp(
    *, source: Path, runtime_dir: Path, strategy: str, swmm_threads: int
) -> Path:
    source_sha = sha256_file(source)
    native = strategy == "internal_rtc"
    policy = "internal" if native else "no_control"
    runtime = runtime_dir / f"{source_sha[:24]}.{policy}.t{swmm_threads}.inp"
    if not runtime.is_file():
        build_runtime_inp(
            source,
            runtime,
            native_controls=native,
            swmm_threads=swmm_threads,
        )
    return runtime


def _run_job(job: dict[str, object]) -> dict[str, object]:
    strategy = str(job["strategy"])
    result = run_authoritative_closed_loop(
        inp_path=str(job["runtime_inp"]),
        output_dir=str(job["event_dir"]),
        run_id=str(job["run_id"]),
        sensor_nodes=(),
        controller=fixed_baseline_controller(strategy),
        control_start_minutes=int(job["control_start_minutes"]),
        control_update_seconds=int(job["control_update_seconds"]),
        observation_update_seconds=int(job["model_step_seconds"]),
        record_stride_seconds=int(job["record_stride_seconds"]),
        exact_global_peak=False,
        save_raw_csv=False,
        keep_engine_files=False,
    )
    evidence = validate_fixed_baseline_run(
        strategy=strategy,
        main_metadata_path=result.metadata_path,
        source_physical_sha256=str(job["physical_network_sha256"]),
    )
    formal_path = ""
    formal_sha = ""
    if bool(job["formalize"]):
        formal_file = Path(str(job["event_dir"])) / f"{job['run_id']}.formal_manifest.json"
        formalize_run(
            main_metadata_path=result.metadata_path,
            strategy=strategy,
            event_id=str(job["event_id"]),
            rainfall_group=str(job["rainfall_group"]),
            output_path=formal_file,
        )
        formal_path = str(formal_file.resolve())
        formal_sha = sha256_file(formal_file)
    _, code_sha = generation_key("code_probe", {})
    sidecar = {
        "contract": CACHE_CONTRACT,
        "cache_key_sha256": str(job["cache_key_sha256"]),
        "rtc_source_tree_sha256": code_sha,
        "event_id": str(job["event_id"]),
        "rainfall_group": str(job["rainfall_group"]),
        "scientific_split": str(job["scientific_split"]),
        "development_fold": str(job["development_fold"]),
        "strategy": strategy,
        "source_inp": str(Path(str(job["source_inp"])).resolve()),
        "source_inp_sha256": str(job["source_inp_sha256"]),
        "physical_network_sha256": str(job["physical_network_sha256"]),
        "runtime": {
            "model_step_seconds": int(job["model_step_seconds"]),
            "control_update_seconds": int(job["control_update_seconds"]),
            "record_stride_seconds": int(job["record_stride_seconds"]),
            "control_start_minutes": int(job["control_start_minutes"]),
            "swmm_threads_per_process": int(job["swmm_threads_per_process"]),
        },
        "main_metadata_path": str(Path(result.metadata_path).resolve()),
        "main_metadata_sha256": sha256_file(result.metadata_path),
        "evidence": evidence,
        "formal_manifest_path": formal_path or None,
        "formal_manifest_sha256": formal_sha or None,
    }
    sidecar_path = Path(str(job["sidecar_path"]))
    sidecar_path.write_text(canonical_json(sidecar) + "\n", encoding="utf-8")
    return {
        "event_id": sidecar["event_id"],
        "rainfall_group": sidecar["rainfall_group"],
        "scientific_split": sidecar["scientific_split"],
        "development_fold": sidecar["development_fold"],
        "strategy": strategy,
        "cache_key_sha256": sidecar["cache_key_sha256"],
        "metadata_path": sidecar["main_metadata_path"],
        "compact_path": evidence["compact_path"],
        "node_statistics_path": evidence["node_statistics_path"],
        "decision_log_path": evidence["decision_log_path"],
        "formal_manifest_path": formal_path,
        "sidecar_path": str(sidecar_path.resolve()),
        "flow_routing_error_pct": result.flow_routing_error_pct,
        "status": "completed",
    }


def _parse_strategies(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return FIXED_BASELINE_IDS
    values = tuple(x.strip() for x in raw.split(",") if x.strip())
    invalid = sorted(set(values) - set(FIXED_BASELINE_IDS))
    if invalid:
        raise ValueError(f"unsupported fixed baselines: {invalid}")
    if len(values) != len(set(values)):
        raise ValueError("duplicate strategies requested")
    return values


def _config_values(path: str | Path) -> dict[str, int]:
    cfg = _read_json(path)
    values = {
        "model_step_seconds": int(cfg["model_step_seconds"]),
        "control_update_seconds": int(cfg["control_update_seconds"]),
        "record_stride_seconds": int(cfg.get("record_stride_seconds", cfg["model_step_seconds"])),
        "control_start_minutes": int(cfg.get("control_start_minutes", 0)),
    }
    if values["model_step_seconds"] <= 0:
        raise ValueError("model_step_seconds must be positive")
    if values["control_update_seconds"] % values["model_step_seconds"]:
        raise ValueError("control_update_seconds must be an integer multiple of model_step_seconds")
    if values["record_stride_seconds"] != values["model_step_seconds"]:
        raise ValueError("Formal baseline record stride must equal model step")
    if cfg.get("exact_global_peak") is not False:
        raise ValueError("baseline main runs must set exact_global_peak=false; Formal peak is replayed")
    return values


def build_baseline_cache(
    *,
    event_registry: pd.DataFrame,
    output_dir: str | Path,
    config_path: str | Path,
    strategies: Iterable[str],
    stage: str,
    workers: int,
    swmm_threads_per_process: int,
    force: bool = False,
    formalize_final: bool = False,
) -> pd.DataFrame:
    required = {"event_id", "rainfall_group", "inp_path", "scientific_split"}
    missing = sorted(required - set(event_registry.columns))
    if missing:
        raise ValueError(f"event registry missing columns: {missing}")
    if event_registry["event_id"].astype(str).duplicated().any():
        raise ValueError("event registry must contain one row per event_id")
    if stage not in {"prelock", "final"}:
        raise ValueError("stage must be prelock or final")
    if workers <= 0 or swmm_threads_per_process <= 0:
        raise ValueError("workers and SWMM threads/process must be positive")
    cfg = _config_values(config_path)
    strategies = tuple(str(x) for x in strategies)
    if not strategies:
        raise ValueError("at least one fixed baseline is required")

    frame = event_registry.copy()
    frame["scientific_split"] = frame["scientific_split"].astype(str)
    if stage == "prelock":
        frame = frame[frame["scientific_split"] != "final"].copy()
    else:
        frame = frame[frame["scientific_split"] == "final"].copy()
    if frame.empty:
        raise ValueError(f"no {stage} events remain after split filtering")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    runtime_dir = out / "_runtime_inp"
    runtime_dir.mkdir(exist_ok=True)
    jobs: list[dict[str, object]] = []
    results: list[dict[str, object]] = []

    for _, row in frame.iterrows():
        source = Path(str(row["inp_path"]))
        if not source.is_file():
            raise ValueError(f"event INP missing: {source}")
        source_sha = sha256_file(source)
        physical_sha = physical_contract_sha256(source)
        event_id = str(row["event_id"])
        rainfall_group = str(row["rainfall_group"])
        split = str(row["scientific_split"])
        fold = str(row.get("development_fold", ""))
        for strategy in strategies:
            runtime = _runtime_inp(
                source=source,
                runtime_dir=runtime_dir,
                strategy=strategy,
                swmm_threads=swmm_threads_per_process,
            )
            key = baseline_cache_key(
                source_inp_sha256=source_sha,
                physical_network_sha256=physical_sha,
                strategy=strategy,
                swmm_threads_per_process=swmm_threads_per_process,
                **cfg,
            )
            event_dir = out / event_id / strategy
            event_dir.mkdir(parents=True, exist_ok=True)
            run_id = f"{event_id}__{strategy}"
            sidecar_path = event_dir / f"{run_id}.baseline_cache.json"
            cached = None if force else _cache_complete(
                sidecar_path, key, require_formal=formalize_final
            )
            if cached is not None:
                evidence = cached["evidence"]
                assert isinstance(evidence, dict)
                results.append(
                    {
                        "event_id": event_id,
                        "rainfall_group": rainfall_group,
                        "scientific_split": split,
                        "development_fold": fold,
                        "strategy": strategy,
                        "cache_key_sha256": key,
                        "metadata_path": str(cached["main_metadata_path"]),
                        "compact_path": str(evidence["compact_path"]),
                        "node_statistics_path": str(evidence["node_statistics_path"]),
                        "decision_log_path": str(evidence["decision_log_path"]),
                        "formal_manifest_path": str(cached.get("formal_manifest_path") or ""),
                        "sidecar_path": str(sidecar_path.resolve()),
                        "flow_routing_error_pct": np.nan,
                        "status": "resumed",
                    }
                )
                continue
            jobs.append(
                {
                    "event_id": event_id,
                    "rainfall_group": rainfall_group,
                    "scientific_split": split,
                    "development_fold": fold,
                    "strategy": strategy,
                    "source_inp": str(source),
                    "source_inp_sha256": source_sha,
                    "physical_network_sha256": physical_sha,
                    "runtime_inp": str(runtime),
                    "event_dir": str(event_dir),
                    "run_id": run_id,
                    "sidecar_path": str(sidecar_path),
                    "cache_key_sha256": key,
                    "swmm_threads_per_process": swmm_threads_per_process,
                    "formalize": bool(formalize_final),
                    **cfg,
                }
            )

    if jobs:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            futures = [pool.submit(_run_job, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
    return pd.DataFrame(results).sort_values(["event_id", "strategy"]).reset_index(drop=True)


def _write_views(frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    index_path = output_dir / "BASELINE_CACHE_INDEX.csv"
    frame.to_csv(index_path, index=False)
    no_control = frame[frame["strategy"] == "no_control"].copy()
    no_control_path = output_dir / "NO_CONTROL_D0_INDEX.csv"
    no_control.to_csv(no_control_path, index=False)
    step1 = frame[
        (frame["scientific_split"].astype(str) == "development")
        & frame["strategy"].isin(["no_control", "internal_rtc"])
    ].copy()
    step1_path = output_dir / "STEP1_BASELINE_INDEX.csv"
    step1.to_csv(step1_path, index=False)
    final = frame[frame["formal_manifest_path"].astype(str) != ""].copy()
    final_path = output_dir / "FINAL_BASELINE_RUN_INDEX.csv"
    final.to_csv(final_path, index=False)
    return {
        "baseline_index": str(index_path),
        "no_control_d0_index": str(no_control_path),
        "step1_baseline_index": str(step1_path),
        "final_baseline_run_index": str(final_path),
    }


def main() -> None:
    """Debug/compatibility entrypoint; public Formal CLI is baseline_cache_cli.main."""

    parser = argparse.ArgumentParser(description="Build code-bound deterministic baseline cache")
    parser.add_argument("--events", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--strategies")
    parser.add_argument("--stage", choices=["prelock", "final"], default="prelock")
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--swmm-threads-per-process", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    frame = build_baseline_cache(
        event_registry=pd.read_csv(args.events),
        output_dir=args.out_dir,
        config_path=args.config,
        strategies=_parse_strategies(args.strategies),
        stage=args.stage,
        workers=args.workers,
        swmm_threads_per_process=args.swmm_threads_per_process,
        force=args.force,
        formalize_final=False,
    )
    print(json.dumps({"contract": CACHE_CONTRACT, "rows": len(frame)}, indent=2))


if __name__ == "__main__":
    main()
