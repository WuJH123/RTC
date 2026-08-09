from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import baseline_cache as legacy
from .baselines import (
    FIXED_BASELINE_IDS,
    baseline_sensor_nodes,
    canonical_baseline_id,
    fixed_baseline_controller,
)
from .closed_loop import run_authoritative_closed_loop
from .formalize_run_v2 import formalize_run
from .generation_contract import canonical_json, generation_key
from .inp_lineage import physical_contract_sha256
from .inp_runtime import section_has_payload, sha256_file
from .rule_baselines import AUTO_RBC_CONTRACT, AUTO_RBC_SOURCE, EFD_CONTRACT, EFD_SOURCE

CACHE_CONTRACT = legacy.CACHE_CONTRACT


def _read_json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return value


def _rule_max_delta(config_path: str | Path) -> float | None:
    cfg = _read_json(config_path)
    controller = cfg.get("controller", {})
    if not isinstance(controller, dict):
        return None
    raw = controller.get("max_setting_delta_per_update")
    if raw is None:
        return None
    value = float(raw)
    if value < 0:
        raise ValueError("max_setting_delta_per_update must be non-negative")
    return value


def _cache_key(
    *,
    source_inp_sha256: str,
    physical_network_sha256: str,
    strategy: str,
    config_values: dict[str, int],
    rule_max_delta: float | None,
    swmm_threads_per_process: int,
) -> str:
    key, _ = generation_key(
        "fixed_baseline_dynamic_rules_v3",
        {
            "cache_contract": CACHE_CONTRACT,
            "source_inp_sha256": source_inp_sha256,
            "physical_network_sha256": physical_network_sha256,
            "strategy": strategy,
            **{k: int(v) for k, v in config_values.items()},
            "rule_max_setting_delta_per_update": rule_max_delta,
            "swmm_threads_per_process": int(swmm_threads_per_process),
            "exact_global_peak_in_main": False,
        },
    )
    return key


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
    *, strategy: str, main_metadata_path: str | Path, source_physical_sha256: str
) -> dict[str, object]:
    strategy = canonical_baseline_id(strategy)
    meta_path = Path(main_metadata_path)
    meta = _read_json(meta_path)
    if meta.get("data_contract") != "CLOSED_LOOP_COMPACT_V2":
        raise ValueError("baseline cache requires CLOSED_LOOP_COMPACT_V2 evidence")
    runtime_inp = Path(str(meta["inp_path"]))
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
    elif strategy in {"all_open", "all_closed"}:
        expected = 1.0 if strategy == "all_open" else 0.0
        expected_source = "ALL_OPEN" if strategy == "all_open" else "ALL_CLOSED"
        if controls or not controller_present or not decisions:
            raise ValueError(f"{strategy} must use Python writes on a controls-disabled runtime")
        for row in decisions:
            settings = row.get("settings")
            if row.get("source") != expected_source or not isinstance(settings, dict) or not settings:
                raise ValueError(f"{strategy} decision log does not prove the requested strategy")
            if any(abs(float(v) - expected) > 1e-9 for v in settings.values()):
                raise ValueError(f"{strategy} did not command every actuator to {expected}")
    elif strategy in {"auto_rbc", "efd"}:
        expected_source = AUTO_RBC_SOURCE if strategy == "auto_rbc" else EFD_SOURCE
        expected_contract = AUTO_RBC_CONTRACT if strategy == "auto_rbc" else EFD_CONTRACT
        if controls or not controller_present or not decisions:
            raise ValueError(f"{strategy} requires causal Python decisions on controls-disabled SWMM")
        for row in decisions:
            settings = row.get("settings")
            diagnostics = row.get("diagnostics")
            if row.get("source") != expected_source or not isinstance(settings, dict) or not settings:
                raise ValueError(f"{strategy} decision source/settings are invalid")
            if any(not 0.0 <= float(v) <= 1.0 for v in settings.values()):
                raise ValueError(f"{strategy} emitted an out-of-range setting")
            if not isinstance(diagnostics, dict) or diagnostics.get("rule_contract") != expected_contract:
                raise ValueError(f"{strategy} decision lacks the frozen rule contract")
    elif strategy == "hold":
        if controls or not controller_present or not decisions:
            raise ValueError("Hold must use Python writes on a controls-disabled runtime")
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


def _run_job(job: dict[str, object]) -> dict[str, object]:
    strategy = canonical_baseline_id(str(job["strategy"]))
    source_inp = str(job["source_inp"])
    sensors = baseline_sensor_nodes(strategy, source_inp)
    controller = fixed_baseline_controller(
        strategy,
        inp_path=source_inp,
        max_delta_per_update=(
            None if job.get("rule_max_delta") is None else float(job["rule_max_delta"])
        ),
    )
    result = run_authoritative_closed_loop(
        inp_path=str(job["runtime_inp"]),
        output_dir=str(job["event_dir"]),
        run_id=str(job["run_id"]),
        sensor_nodes=sensors,
        controller=controller,
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
        "source_inp": str(Path(source_inp).resolve()),
        "source_inp_sha256": str(job["source_inp_sha256"]),
        "physical_network_sha256": str(job["physical_network_sha256"]),
        "rule_max_setting_delta_per_update": job.get("rule_max_delta"),
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


def parse_strategies(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return FIXED_BASELINE_IDS
    values = tuple(canonical_baseline_id(x) for x in raw.split(",") if x.strip())
    invalid = sorted(set(values) - set(FIXED_BASELINE_IDS))
    if invalid:
        raise ValueError(f"unsupported fixed baselines: {invalid}")
    if len(values) != len(set(values)):
        raise ValueError("duplicate strategies requested after alias canonicalization")
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
    cfg = legacy._config_values(config_path)
    rule_max_delta = _rule_max_delta(config_path)
    strategies = tuple(canonical_baseline_id(x) for x in strategies)
    invalid = sorted(set(strategies) - set(FIXED_BASELINE_IDS))
    if invalid:
        raise ValueError(f"unsupported fixed baselines: {invalid}")

    frame = event_registry.copy()
    frame["scientific_split"] = frame["scientific_split"].astype(str)
    frame = frame[frame["scientific_split"] != "final"].copy() if stage == "prelock" else frame[frame["scientific_split"] == "final"].copy()
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
            runtime = legacy._runtime_inp(
                source=source,
                runtime_dir=runtime_dir,
                strategy=strategy,
                swmm_threads=swmm_threads_per_process,
            )
            key = _cache_key(
                source_inp_sha256=source_sha,
                physical_network_sha256=physical_sha,
                strategy=strategy,
                config_values=cfg,
                rule_max_delta=rule_max_delta,
                swmm_threads_per_process=swmm_threads_per_process,
            )
            event_dir = out / event_id / strategy
            event_dir.mkdir(parents=True, exist_ok=True)
            run_id = f"{event_id}__{strategy}"
            sidecar_path = event_dir / f"{run_id}.baseline_cache.json"
            cached = None if force else legacy._cache_complete(
                sidecar_path, key, require_formal=formalize_final
            )
            if cached is not None:
                evidence = cached["evidence"]
                assert isinstance(evidence, dict)
                results.append({
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
                })
                continue
            jobs.append({
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
                "rule_max_delta": rule_max_delta,
                **cfg,
            })
    if jobs:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            futures = [pool.submit(_run_job, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())
    return pd.DataFrame(results).sort_values(["event_id", "strategy"]).reset_index(drop=True)


def write_views(frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    return legacy._write_views(frame, output_dir)
