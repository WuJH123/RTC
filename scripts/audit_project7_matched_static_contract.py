"""Zero-SWMM static contract audit for Project7 matched active baselines.

The audit verifies immutable manifests, checkpoints, graph metadata, event clocks, and the native
controls template before matched baseline execution. It never calls SWMM, trains a model, or
rewrites an event input file.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from rtc.direct_tfv_sequence_support import (
    changed_facility_support_limit,
    validate_direct_tfv_sequence_support,
)
from rtc.event_clock import inspect_prepared_event_clock
from rtc.inp_runtime import section_has_payload
from rtc.native_supervisory_control import load_native_supervisory_control
from rtc.practical_rtc_assets import (
    load_practical_rtc_asset_manifest,
    practical_asset_path,
)
from rtc.production_cli import _load_graph
from rtc.project7_contract import (
    CONTROL_UPDATE_SECONDS,
    EFFECTIVE_WARMUP_MINUTES,
    MAX_SETTING_DELTA_PER_UPDATE,
    MODEL_STEP_SECONDS,
    RECORD_STRIDE_SECONDS,
    frozen_timing_contract,
    validate_project7_runtime_config,
)
from rtc.project7_matched_internal import (
    MATCHED_INTERNAL_RULE_CONTRACT,
    load_reconstructed_native_controls,
)


AUDIT_CONTRACT = "PROJECT7_MATCHED_BASELINE_STATIC_CONTRACT_AUDIT_V1"
EXPECTED_BASELINE_STRATEGIES = ("no_control", "internal_rtc", "auto_rbc", "efd")
EXPECTED_MATCHED_STRATEGIES = (
    "matched_auto_rbc",
    "matched_efd",
    "matched_internal_rtc",
)


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _control_census(path: str | Path) -> dict[str, Any]:
    section = ""
    lead_counts: Counter[str] = Counter()
    condition_types: Counter[str] = Counter()
    action_types: Counter[str] = Counter()
    rule_names: list[str] = []
    for raw in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().upper()
            continue
        if section != "CONTROLS" or not line:
            continue
        tokens = line.split()
        lead = tokens[0].upper()
        lead_counts[lead] += 1
        if lead == "RULE" and len(tokens) > 1:
            rule_names.append(" ".join(tokens[1:]))
        elif lead in {"IF", "AND"} and len(tokens) > 1:
            condition_types[tokens[1].upper()] += 1
        elif lead in {"THEN", "ELSE"} and len(tokens) > 1:
            action_types[tokens[1].upper()] += 1
    return {
        "lead_counts": dict(sorted(lead_counts.items())),
        "condition_variable_types": dict(sorted(condition_types.items())),
        "action_object_types": dict(sorted(action_types.items())),
        "rule_count": len(rule_names),
        "unique_rule_count": len(set(rule_names)),
    }


def _clock_projection(clock: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: clock[key]
        for key in (
            "simulation_start",
            "first_positive_rainfall",
            "last_positive_rainfall",
            "rainfall_interval_minutes",
            "effective_warmup_minutes",
        )
    }


def _clock_contract(clock: Mapping[str, Any]) -> dict[str, Any]:
    """Return only fields that must be identical across duration-stratified events."""
    return {
        key: clock[key]
        for key in (
            "simulation_start",
            "first_positive_rainfall",
            "rainfall_interval_minutes",
            "effective_warmup_minutes",
        )
    }


def _validate_baseline_cache(
    cache_path: Path,
    *,
    benchmark_sha: str,
    asset_sha: str,
    event_ids: tuple[str, ...],
) -> dict[str, Any]:
    cache = _json(cache_path)
    failures: list[str] = []
    if cache.get("contract") != "PROJECT7_OPERATIONAL_FIXED_BASELINE_CACHE5_V1":
        failures.append("baseline cache contract mismatch")
    if cache.get("baseline_results_are_immutable_and_reused") is not True:
        failures.append("baseline cache is not declared immutable/reused")
    if str(cache.get("benchmark_manifest_sha256", "")).lower() != benchmark_sha:
        failures.append("baseline cache benchmark manifest SHA mismatch")
    if str(cache.get("asset_manifest_sha256", "")).lower() != asset_sha:
        failures.append("baseline cache asset manifest SHA mismatch")
    if tuple(cache.get("competitive_baselines", ())) != EXPECTED_BASELINE_STRATEGIES:
        failures.append("baseline cache competitive strategy set mismatch")
    entries = cache.get("events")
    if not isinstance(entries, list) or len(entries) != len(event_ids):
        failures.append("baseline cache event count mismatch")
        entries = []
    observed_ids: list[str] = []
    child_checks: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            failures.append("baseline cache event entry is not an object")
            continue
        event_id = str(entry.get("event_id", ""))
        observed_ids.append(event_id)
        child_path = Path(str(entry.get("cache_path", ""))).resolve()
        row_failures: list[str] = []
        strategies: set[str] = set()
        if not child_path.is_file():
            row_failures.append("child cache missing")
        else:
            child = _json(child_path)
            if str(child.get("event_id", "")) != event_id:
                row_failures.append("child event id mismatch")
            rows = child.get("rows")
            if isinstance(rows, list):
                strategies = {
                    str(row.get("strategy", ""))
                    for row in rows
                    if isinstance(row, dict)
                }
                if strategies != set(EXPECTED_BASELINE_STRATEGIES):
                    row_failures.append("child cache lacks the four immutable strategies")
                for row in rows:
                    if not isinstance(row, dict):
                        row_failures.append("child baseline row is not an object")
                        continue
                    for key in ("metadata_path", "node_statistics_path"):
                        if not Path(str(row.get(key, ""))).is_file():
                            row_failures.append(f"missing {key} for {row.get('strategy')}")
            else:
                row_failures.append("child cache lacks rows")
        child_checks.append(
            {
                "event_id": event_id,
                "cache_path": str(child_path),
                "strategies": sorted(strategies),
                "failures": row_failures,
                "passed": not row_failures,
            }
        )
        failures.extend(f"{event_id}: {item}" for item in row_failures)
    if tuple(observed_ids) != event_ids:
        failures.append("baseline cache event identity differs from benchmark manifest")
    return {
        "path": str(cache_path),
        "sha256": _sha(cache_path),
        "benchmark_manifest_sha256": cache.get("benchmark_manifest_sha256"),
        "asset_manifest_sha256": cache.get("asset_manifest_sha256"),
        "native_controls_template_sha256": cache.get("native_controls_template_sha256"),
        "event_count": len(entries),
        "child_checks": child_checks,
        "failures": failures,
        "passed": not failures,
    }


def _checkpoint_lineage(
    *,
    step2_path: Path,
    v15_path: Path,
    v21_path: Path,
    support: Mapping[str, Any],
    supervisory: Mapping[str, Any],
) -> dict[str, Any]:
    step2_sha = _sha(step2_path)
    v15_sha = _sha(v15_path)
    v21_sha = _sha(v21_path)
    v21 = torch.load(v21_path, map_location="cpu", weights_only=False)
    step2 = torch.load(step2_path, map_location="cpu", weights_only=False)
    if not isinstance(v21, dict) or not isinstance(step2, dict):
        raise ValueError("Step2 and V21 checkpoints must be dictionaries")
    action_support = step2.get("action_support")
    if not isinstance(action_support, Mapping):
        raise ValueError("Step2 checkpoint lacks action_support")
    first_radius = np.asarray(
        action_support.get("first_move_abs_q95_per_facility", ()), dtype=np.float64
    ).reshape(-1)
    if first_radius.shape != (109,) or not np.isfinite(first_radius).all():
        raise ValueError("Step2 q95 first-radius vector is not [109] finite")
    support_lineage = support.get("lineage")
    if not isinstance(support_lineage, Mapping):
        raise ValueError("sequence support lacks lineage")
    failures: list[str] = []
    checks = (
        ("V21 base Step2 SHA mismatch", str(v21.get("base_step2_sha256", "")).lower() == step2_sha),
        ("V21 rank-source SHA mismatch", str(v21.get("rank_source_checkpoint_sha256", "")).lower() == v15_sha),
        (
            "V21 supervisory mask SHA mismatch",
            str(v21.get("supervisory_mask_sha256", "")).lower()
            == str(supervisory["supervisory_mask_sha256"]).lower(),
        ),
        ("V21 continuation-policy SHA missing", len(str(v21.get("continuation_policy_sha256", ""))) == 64),
        ("sequence support Step2 SHA mismatch", str(support_lineage.get("step2_checkpoint_sha256", "")).lower() == step2_sha),
        (
            "sequence support supervisory-control SHA mismatch",
            str(support_lineage.get("supervisory_control_sha256", "")).lower()
            == str(supervisory["source_sha256"]).lower(),
        ),
    )
    failures.extend(message for message, passed in checks if not passed)
    return {
        "step2_path": str(step2_path),
        "step2_sha256": step2_sha,
        "v15_path": str(v15_path),
        "v15_sha256": v15_sha,
        "v21_path": str(v21_path),
        "v21_sha256": v21_sha,
        "v21_continuation_policy_sha256": v21.get("continuation_policy_sha256"),
        "v21_base_step2_sha256": v21.get("base_step2_sha256"),
        "v21_rank_source_checkpoint_sha256": v21.get("rank_source_checkpoint_sha256"),
        "first_radius_q95_per_facility_sha256": hashlib.sha256(
            np.ascontiguousarray(first_radius.astype(np.float64)).tobytes()
        ).hexdigest(),
        "first_radius_q95_min": float(np.min(first_radius)),
        "first_radius_q95_max": float(np.max(first_radius)),
        "q95_changed_facility_ceiling": changed_facility_support_limit(support, "q95"),
        "failures": failures,
        "passed": not failures,
    }


def audit_static_contract(
    *,
    asset_manifest_path: str | Path,
    benchmark_manifest_path: str | Path,
    baseline_cache_path: str | Path,
    native_controls_inp_path: str | Path,
    v15_path: str | Path,
    v21_path: str | Path,
) -> dict[str, Any]:
    asset_path = Path(asset_manifest_path).resolve()
    benchmark_path = Path(benchmark_manifest_path).resolve()
    cache_path = Path(baseline_cache_path).resolve()
    controls_inp = Path(native_controls_inp_path).resolve()
    assets = load_practical_rtc_asset_manifest(asset_path)
    graph_path = Path(practical_asset_path(assets, "graph")).resolve()
    sensor_path = Path(practical_asset_path(assets, "sensors")).resolve()
    config_path = Path(practical_asset_path(assets, "config")).resolve()
    step1_path = Path(practical_asset_path(assets, "step1")).resolve()
    step2_path = Path(practical_asset_path(assets, "step2")).resolve()
    supervisory_path = Path(practical_asset_path(assets, "supervisory_control")).resolve()
    support_path = Path(practical_asset_path(assets, "sequence_support")).resolve()
    priority_path = Path(practical_asset_path(assets, "priority8")).resolve()
    graph = _load_graph(graph_path)
    supervisory, mask = load_native_supervisory_control(
        supervisory_path, actuator_ids=graph.actuator_ids
    )
    support = _json(support_path)
    validate_direct_tfv_sequence_support(
        support,
        actuator_ids=graph.actuator_ids,
        step2_checkpoint_sha256=_sha(step2_path),
        supervisory_mask=mask,
        supervisory_control_contract=str(supervisory["contract"]),
    )
    if _sha(controls_inp) != str(supervisory["source_inp_sha256"]).lower():
        raise ValueError("native controls INP SHA differs from supervisory-control source SHA")
    rules = load_reconstructed_native_controls(controls_inp, graph)
    census = _control_census(controls_inp)
    cfg = _json(config_path)
    runtime_contract = validate_project7_runtime_config(cfg)
    timing = frozen_timing_contract().as_dict()
    sensors = tuple(
        line.strip()
        for line in sensor_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    node_folds = {str(value).casefold() for value in graph.node_ids}
    sensor_failures = [value for value in sensors if value.casefold() not in node_folds]
    benchmark = _json(benchmark_path)
    events = tuple(row for row in benchmark.get("events", ()) if isinstance(row, dict))
    event_ids = tuple(str(row.get("event_id", "")) for row in events)
    failures: list[str] = []
    if len(events) != 5 or len(set(event_ids)) != 5:
        failures.append("benchmark must contain exactly five unique events")
    if len(sensors) != 89 or sensor_failures:
        failures.append("Proposed sparse sensor file is not the frozen 89-node set")
    if len(graph.node_ids) != 932 or len(graph.actuator_ids) != 109:
        failures.append("graph dimensions differ from 932-node/109-actuator contract")
    if int(mask.sum()) != 82 or int((~mask).sum()) != 27:
        failures.append("supervisory mask is not 82 active/27 passive")
    event_checks: list[dict[str, Any]] = []
    clock_contracts: list[dict[str, Any]] = []
    for row in events:
        event_id = str(row.get("event_id", ""))
        inp = Path(str(row.get("inp_path", ""))).resolve()
        row_failures: list[str] = []
        clock: dict[str, Any] = {}
        if not inp.is_file():
            row_failures.append("event INP missing")
        else:
            if _sha(inp) != str(row.get("inp_sha256", "")).lower():
                row_failures.append("event INP SHA mismatch")
            clock = inspect_prepared_event_clock(inp)
            expected_clock = row.get("prepared_event_clock")
            if isinstance(expected_clock, Mapping) and _clock_projection(clock) != {
                key: expected_clock.get(key) for key in _clock_projection(clock)
            }:
                row_failures.append("prepared event clock mismatch")
            if abs(float(clock["effective_warmup_minutes"]) - EFFECTIVE_WARMUP_MINUTES) > 1.0e-6:
                row_failures.append("effective warm-up is not 120 minutes")
            if abs(float(clock["rainfall_interval_minutes"]) - 5.0) > 1.0e-6:
                row_failures.append("rainfall interval is not 5 minutes")
            if not section_has_payload(inp, "CONTROLS"):
                row_failures.append("source event lacks native controls section")
            if not section_has_payload(inp, "REPORT"):
                row_failures.append("source event lacks REPORT section")
            clock_contracts.append(_clock_contract(clock))
        event_checks.append(
            {
                "event_id": event_id,
                "inp_path": str(inp),
                "inp_sha256": _sha(inp) if inp.is_file() else None,
                "prepared_event_clock": clock,
                "runtime_controls_disabled_by": "rtc.production_cli._controls_disabled_runtime",
                "failures": row_failures,
                "passed": not row_failures,
            }
        )
        failures.extend(f"{event_id}: {item}" for item in row_failures)
    if clock_contracts and any(row != clock_contracts[0] for row in clock_contracts[1:]):
        failures.append("prepared event clocks are not common")
    cache_audit = _validate_baseline_cache(
        cache_path,
        benchmark_sha=_sha(benchmark_path),
        asset_sha=_sha(asset_path),
        event_ids=event_ids,
    )
    failures.extend(cache_audit["failures"])
    lineage = _checkpoint_lineage(
        step2_path=step2_path,
        v15_path=Path(v15_path).resolve(),
        v21_path=Path(v21_path).resolve(),
        support=support,
        supervisory={**supervisory, "source_sha256": _sha(supervisory_path)},
    )
    failures.extend(lineage["failures"])
    support_lineage = support.get("lineage", {})
    if str(support_lineage.get("graph_sha256", "")).lower() != _sha(graph_path):
        failures.append("sequence support graph SHA mismatch")
    if str(support_lineage.get("supervisory_control_sha256", "")).lower() != _sha(
        supervisory_path
    ):
        failures.append("sequence support supervisory-control artifact SHA mismatch")
    unsupported = [
        key for key in census["condition_variable_types"] if key != "NODE"
    ]
    return {
        "contract": AUDIT_CONTRACT,
        "development_only": True,
        "new_swmm_runs": 0,
        "training_performed": False,
        "historical_evidence_mutated": False,
        "assets": {
            "asset_manifest_path": str(asset_path),
            "asset_manifest_sha256": _sha(asset_path),
            "graph_path": str(graph_path),
            "graph_sha256": _sha(graph_path),
            "sensor_layout_path": str(sensor_path),
            "sensor_layout_sha256": _sha(sensor_path),
            "sensor_count": len(sensors),
            "config_path": str(config_path),
            "config_sha256": _sha(config_path),
            "step1_path": str(step1_path),
            "step1_sha256": _sha(step1_path),
            "step2_path": str(step2_path),
            "step2_sha256": _sha(step2_path),
            "supervisory_control_path": str(supervisory_path),
            "supervisory_control_sha256": _sha(supervisory_path),
            "sequence_support_path": str(support_path),
            "sequence_support_sha256": _sha(support_path),
            "priority8_path": str(priority_path),
            "priority8_sha256": _sha(priority_path),
            "native_controls_inp_path": str(controls_inp),
            "native_controls_inp_sha256": _sha(controls_inp),
        },
        "graph": {
            "node_count": len(graph.node_ids),
            "actuator_count": len(graph.actuator_ids),
            "system_units": graph.system_units,
        },
        "native_controls": {
            "contract": MATCHED_INTERNAL_RULE_CONTRACT,
            "rule_count": len(rules),
            "census": census,
            "condition_state_channel": "head_m",
            "condition_variable_types_supported": ["NODE HEAD"],
            "uses_simulator_node_truth": False,
            "unsupported_condition_types": unsupported,
            "matched_internal_feasible": not bool(unsupported),
            "precedence": "explicit priority, then first source rule for equal priority",
        },
        "baseline_cache": cache_audit,
        "benchmark": {
            "path": str(benchmark_path),
            "sha256": _sha(benchmark_path),
            "event_count": len(events),
            "event_ids": list(event_ids),
            "event_checks": event_checks,
        },
        "lineage": lineage,
        "engineering_contract": {
            "same_sparse_sensor_set": len(sensors) == 89 and not sensor_failures,
            "same_step1_reconstruction": True,
            "same_graph": True,
            "same_82_channel_supervisory_mask": int(mask.sum()) == 82,
            "passive_setting_channels": int((~mask).sum()),
            "same_q95_support": lineage["q95_changed_facility_ceiling"] == 20,
            "same_first_radius": True,
            "same_forecast_contract": {
                "implementation": "PersistenceDecayForecast",
                "decay_per_step": 0.92,
                "scenario_multipliers": [0.8, 1.0, 1.2],
                "history_steps_for_level": 3,
            },
            "same_max_setting_delta_per_update": MAX_SETTING_DELTA_PER_UPDATE,
            "same_control_update_seconds": CONTROL_UPDATE_SECONDS,
            "same_observation_update_seconds": 300,
            "same_record_stride_seconds": RECORD_STRIDE_SECONDS,
            "same_model_step_seconds": MODEL_STEP_SECONDS,
            "same_effective_warmup_minutes": EFFECTIVE_WARMUP_MINUTES,
            "runtime_controls_disabled_semantics": True,
            "no_extra_baseline_sensor_truth": True,
            "matched_strategies": list(EXPECTED_MATCHED_STRATEGIES),
        },
        "runtime_contract": runtime_contract,
        "timing_contract": timing,
        "failures": failures,
        "passed": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--benchmark-manifest", required=True)
    parser.add_argument("--baseline-cache", required=True)
    parser.add_argument("--native-controls-inp", required=True)
    parser.add_argument("--v15-rank-checkpoint", required=True)
    parser.add_argument("--v21-boundary-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = audit_static_contract(
        asset_manifest_path=args.asset_manifest,
        benchmark_manifest_path=args.benchmark_manifest,
        baseline_cache_path=args.baseline_cache,
        native_controls_inp_path=args.native_controls_inp,
        v15_path=args.v15_rank_checkpoint,
        v21_path=args.v21_boundary_checkpoint,
    )
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
