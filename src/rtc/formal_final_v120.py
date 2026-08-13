"""Compile the complete locked seven-strategy Final for Project7 V120."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import load_priority_nodes
from .control_lineage import section_payload_sha256
from .formal_final_v5 import (
    COMPETITIVE_BASELINES,
    DIAGNOSTIC_EXTREMES,
    EXPECTED_STRATEGIES,
    _group_detail,
    _strategy_role,
)
from .formal_run_verify import read_json as _json
from .formal_run_verify import verify_formal_run_v4
from .inp_lineage import physical_contract_sha256, scientific_event_contract_sha256
from .policy_lock_v120 import POLICY_LOCK_V120_CONTRACT
from .step2_v120_contract import v120_runtime_contract_sha256


def _sha(path: str | Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verified_lock(path: str | Path) -> dict[str, object]:
    lock = _json(path)
    if lock.get("contract") != POLICY_LOCK_V120_CONTRACT:
        raise ValueError("V120 Final requires the V120 TFV-only Policy Lock")
    if str(lock.get("v120_runtime_contract_sha256", "")) != v120_runtime_contract_sha256():
        raise ValueError("V120 implementation changed after Policy Lock")
    if lock.get("primary_objective") != "whole_system_cumulative_TFV_m3":
        raise ValueError("V120 Policy Lock primary objective drift")
    if lock.get("priority_is_hard_constraint") is not False:
        raise ValueError("V120 priority flooding must not be a hard constraint")
    if lock.get("priority_role") != "report_only" or lock.get("global_peak_role") != "report_only":
        raise ValueError("V120 priority/global peak must remain report-only")
    if lock.get("hydraulic_surrogate_required") is not False or lock.get("hydraulic_gradient_gate_required") is not False:
        raise ValueError("V120 Final cannot re-introduce Hydraulic gates")
    if lock.get("formal_metric_aggregation") != "equal_weight_per_independent_rainfall_group":
        raise ValueError("V120 Final requires rainfall-group-balanced aggregation")
    artefacts = lock.get("artefacts")
    hashes = lock.get("sha256")
    if not isinstance(artefacts, dict) or not isinstance(hashes, dict):
        raise ValueError("V120 Policy Lock lacks artifact/hash maps")
    for name, raw in artefacts.items():
        file = Path(str(raw))
        if not file.is_file() or _sha(file) != str(hashes.get(name, "")):
            raise RuntimeError(f"locked V120 artifact disappeared/changed: {name}: {file}")
    model_contracts = lock.get("model_contracts")
    if not isinstance(model_contracts, dict) or not str(model_contracts.get("swmm_engine_version", "")).strip():
        raise ValueError("V120 Policy Lock lacks SWMM engine identity")
    return lock


def compile_final_v120(
    *, policy_lock_path: str | Path, run_index_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    lock = _verified_lock(policy_lock_path)
    artefacts = lock["artefacts"]
    hashes = lock["sha256"]
    model_contracts = lock["model_contracts"]
    assert isinstance(artefacts, dict) and isinstance(hashes, dict) and isinstance(model_contracts, dict)

    frozen_inp = str(artefacts["frozen_inp"])
    physical_sha = physical_contract_sha256(frozen_inp)
    native_controls_sha = section_payload_sha256(frozen_inp, "CONTROLS")
    priority = load_priority_nodes(str(artefacts["priority_nodes"]))
    plan = _json(str(artefacts["baseline_plan"]))
    strategies = tuple(str(x) for x in plan.get("strategies", []))
    if strategies != EXPECTED_STRATEGIES:
        raise ValueError(f"V120 Final baseline plan must be exactly {list(EXPECTED_STRATEGIES)}")
    controller = _json(str(artefacts["controller_config"]))
    if controller.get("v120_contract") != "PROJECT7_V120_TFV_ONLY_CAUSAL_CONTROLLER_V1":
        raise ValueError("V120 Final controller config mismatch")
    model_step = int(controller["model_step_seconds"])
    control_update = int(controller["control_update_seconds"])
    locked_engine = str(model_contracts["swmm_engine_version"])
    proposed_hashes = {
        name: str(hashes[name])
        for name in ("controller_config", "graph_schema", "step1_model", "step2_model")
    }

    split = pd.read_csv(str(artefacts["split_registry"]), keep_default_na=False)
    required_split = {"event_id", "rainfall_group", "inp_path", "scientific_split"}
    missing_split = sorted(required_split - set(split.columns))
    if missing_split:
        raise ValueError(f"locked split registry lacks Final lineage: {missing_split}")
    for column in ("event_id", "rainfall_group", "scientific_split"):
        split[column] = split[column].astype(str)
    final_registry = split[split["scientific_split"] == "final"].copy()
    if len(final_registry) != 6:
        raise ValueError(f"V120 Final requires exactly six untouched Final events, got {len(final_registry)}")
    event_group = final_registry.set_index("event_id")["rainfall_group"].to_dict()
    event_sha = {
        str(row["event_id"]): scientific_event_contract_sha256(str(row["inp_path"]))
        for _, row in final_registry.iterrows()
    }

    index = pd.read_csv(run_index_path)
    needed = {"event_id", "rainfall_group", "strategy", "formal_manifest_path"}
    missing = sorted(needed - set(index.columns))
    if missing:
        raise ValueError(f"V120 Final run index missing columns: {missing}")
    for column in ("event_id", "rainfall_group", "strategy"):
        index[column] = index[column].astype(str)
    if index.duplicated(["event_id", "strategy"]).any():
        raise ValueError("duplicate event/strategy V120 Final run")
    expected_events = set(final_registry["event_id"])
    present_events = set(index["event_id"])
    if present_events != expected_events:
        raise ValueError(
            "V120 Final must contain every and only locked Final event; "
            f"missing={sorted(expected_events-present_events)}, extra={sorted(present_events-expected_events)}"
        )
    for _, item in index.iterrows():
        event_id = str(item["event_id"])
        if str(item["rainfall_group"]) != str(event_group[event_id]):
            raise ValueError(f"V120 Final event {event_id} rainfall group differs from locked registry")
    for event, group in index.groupby("event_id", sort=False):
        present = set(group["strategy"].astype(str))
        if present != set(strategies):
            raise ValueError(f"incomplete seven-strategy V120 Final matrix at {event}: {sorted(present)}")

    rows: list[dict[str, object]] = []
    for _, item in index.iterrows():
        event_id = str(item["event_id"])
        result = verify_formal_run_v4(
            str(item["formal_manifest_path"]),
            priority=priority,
            physical_sha=physical_sha,
            model_step_seconds=model_step,
            control_update_seconds=control_update,
            expected_event_sha256=event_sha[event_id],
            expected_swmm_engine_version=locked_engine,
            expected_proposed_artifact_sha256=proposed_hashes,
            expected_native_controls_payload_sha256=native_controls_sha,
        )
        for key in ("event_id", "rainfall_group", "strategy"):
            if str(result[key]) != str(item[key]):
                raise ValueError(f"V120 Final index {key} differs from bound formal run")
        result["strategy_role"] = _strategy_role(str(result["strategy"]))
        rows.append(result)

    detail = pd.DataFrame(rows)
    metric_cols = [
        "tfv_m3",
        "priority_flood_volume_m3",
        "global_peak_flood_rate_m3s",
        "main_flow_routing_error_pct",
        "peak_replay_flow_routing_error_pct",
    ]
    grouped = _group_detail(detail, metric_cols)
    summary = (
        grouped.groupby("strategy", as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values("strategy")
        .reset_index(drop=True)
    )
    summary["strategy_role"] = summary["strategy"].map(_strategy_role)
    summary["independent_rainfall_groups"] = int(grouped["rainfall_group"].nunique())
    summary["final_events"] = int(len(expected_events))
    summary["swmm_engine_version"] = locked_engine
    summary["aggregation"] = "equal_weight_per_rainfall_group"

    pairs: dict[str, pd.DataFrame] = {}
    proposed = grouped[grouped["strategy"] == "proposed"].set_index("rainfall_group")
    for reference in strategies:
        if reference == "proposed":
            continue
        base = grouped[grouped["strategy"] == reference].set_index("rainfall_group")
        if set(proposed.index) != set(base.index):
            raise ValueError(f"unpaired V120 Final groups for proposed vs {reference}")
        records: list[dict[str, object]] = []
        for rainfall_group in sorted(proposed.index):
            row: dict[str, object] = {
                "rainfall_group": rainfall_group,
                "reference": reference,
                "reference_role": _strategy_role(reference),
            }
            for metric in ("tfv_m3", "priority_flood_volume_m3", "global_peak_flood_rate_m3s"):
                p = float(proposed.loc[rainfall_group, metric])
                b = float(base.loc[rainfall_group, metric])
                row[f"delta_{metric}"] = p - b
                row[f"reduction_{metric}_pct"] = 100.0 * (b - p) / b if abs(b) > 1e-12 else np.nan
            records.append(row)
        pairs[reference] = pd.DataFrame(records)
    return detail, summary, pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile locked seven-strategy V120 Final")
    parser.add_argument("--policy-lock", required=True)
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    detail, summary, pairs = compile_final_v120(
        policy_lock_path=args.policy_lock,
        run_index_path=args.run_index,
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "v120_final_detail.csv", index=False)
    metric_cols = [
        "tfv_m3",
        "priority_flood_volume_m3",
        "global_peak_flood_rate_m3s",
        "main_flow_routing_error_pct",
        "peak_replay_flow_routing_error_pct",
    ]
    _group_detail(detail, metric_cols).to_csv(out / "v120_final_group_detail.csv", index=False)
    summary.to_csv(out / "v120_final_summary.csv", index=False)
    for reference, frame in pairs.items():
        frame.to_csv(out / f"v120_proposed_vs_{reference}.csv", index=False)
    print(json.dumps({
        "contract": "PROJECT7_V120_FINAL_TFV_PRIMARY_REPORT_ONLY_AUXILIARIES_V1",
        "events": int(detail["event_id"].nunique()),
        "independent_rainfall_groups": int(detail["rainfall_group"].nunique()),
        "aggregation": "equal_weight_per_rainfall_group",
        "strategies": list(EXPECTED_STRATEGIES),
        "competitive_baselines": sorted(COMPETITIVE_BASELINES),
        "diagnostic_extremes": sorted(DIAGNOSTIC_EXTREMES),
        "primary": "whole_system_cumulative_TFV_m3",
        "priority_is_hard_gate": False,
        "hydraulic_surrogate_gate": False,
        "summary": str(out / "v120_final_summary.csv"),
    }, indent=2))


if __name__ == "__main__":
    main()
