from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .code_contract import rtc_source_tree_sha256
from .contracts import load_priority_nodes
from .formal_run_verify import read_json as _json
from .formal_run_verify import verify_formal_run_v4
from .inp_lineage import physical_contract_sha256
from .tfv_pipeline import sha256_file


POLICY_LOCK_CONTRACT = "WUHAN_RTC_TFV_FIRST_POLICY_LOCK_V4_CODE_TIME_DATA_BOUND"


def _verified_lock(path: str | Path) -> dict[str, object]:
    lock = _json(path)
    if lock.get("contract") != POLICY_LOCK_CONTRACT:
        raise ValueError("TFV-first Final requires Policy Lock V4")
    if lock.get("rtc_source_tree_sha256") != rtc_source_tree_sha256():
        raise ValueError(
            "current scientific implementation contract differs from Policy Lock"
        )
    if lock.get("priority_is_hard_constraint") is not False:
        raise ValueError("Policy Lock violates TFV-first soft-priority contract")
    if lock.get("formal_metric_aggregation") != "equal_weight_per_independent_rainfall_group":
        raise ValueError("Policy Lock lacks the rainfall-group-balanced metric contract")
    artefacts, hashes = lock.get("artefacts"), lock.get("sha256")
    if not isinstance(artefacts, dict) or not isinstance(hashes, dict):
        raise ValueError("Policy Lock lacks artefact/hash maps")
    for name, raw in artefacts.items():
        p = Path(str(raw))
        if not p.is_file() or sha256_file(p) != str(hashes.get(name, "")):
            raise RuntimeError(f"locked artefact disappeared/changed: {name}: {p}")
    rainfall_design = lock.get("rainfall_design")
    if not isinstance(rainfall_design, dict) or rainfall_design.get("required_invariants_passed") is not True:
        raise ValueError("Final requires a valid locked rainfall-group split design")
    if int(rainfall_design.get("role_group_counts", {}).get("final", 0)) < 1:  # type: ignore[union-attr]
        raise ValueError("Final requires at least one untouched rainfall group")
    causal_timing = lock.get("causal_timing")
    if (
        not isinstance(causal_timing, dict)
        or causal_timing.get("initial_observation_elapsed_seconds") != 0
    ):
        raise ValueError("Final requires the t=0-included causal timing contract")
    return lock


def _group_detail(detail: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    return (
        detail.groupby(["rainfall_group", "strategy"], as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .sort_values(["rainfall_group", "strategy"])
        .reset_index(drop=True)
    )


def compile_final_v4(
    *, policy_lock_path: str | Path, run_index_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    lock = _verified_lock(policy_lock_path)
    artefacts = lock["artefacts"]
    assert isinstance(artefacts, dict)
    physical_sha = physical_contract_sha256(str(artefacts["frozen_inp"]))
    priority = load_priority_nodes(str(artefacts["priority_nodes"]))
    plan = _json(str(artefacts["baseline_plan"]))
    strategies = tuple(str(x) for x in plan.get("strategies", []))
    expected_strategies = (
        "proposed",
        "no_control",
        "internal_rtc",
        "all_open",
        "all_closed",
    )
    if strategies != expected_strategies:
        raise ValueError(
            "Formal baseline plan must be exactly proposed/no_control/internal_rtc/all_open/all_closed"
        )
    controller = _json(str(artefacts["controller_config"]))
    model_step = int(controller["model_step_seconds"])
    control_update = int(controller["control_update_seconds"])

    split = pd.read_csv(str(artefacts["split_registry"]))
    roles = split[["rainfall_group", "scientific_split"]].drop_duplicates().copy()
    roles["rainfall_group"] = roles["rainfall_group"].astype(str)
    if roles.groupby("rainfall_group")["scientific_split"].nunique().max() != 1:
        raise ValueError("locked split registry contains rainfall-group leakage")
    role_map = (
        roles.set_index("rainfall_group")["scientific_split"].astype(str).to_dict()
    )

    index = pd.read_csv(run_index_path)
    needed = {"event_id", "rainfall_group", "strategy", "formal_manifest_path"}
    missing = sorted(needed - set(index.columns))
    if missing:
        raise ValueError(f"Final run index missing columns: {missing}")
    if index.duplicated(["event_id", "strategy"]).any():
        raise ValueError("duplicate event/strategy Final run")
    if any(
        role_map.get(g) != "final"
        for g in set(index["rainfall_group"].astype(str))
    ):
        raise ValueError("Final index contains a non-final/unknown rainfall group")
    expected = set(strategies)
    for event, group in index.groupby("event_id", sort=False):
        present = set(group["strategy"].astype(str))
        if present != expected:
            raise ValueError(
                f"incomplete Final strategy matrix at {event}: {sorted(present)}"
            )
        if group["rainfall_group"].astype(str).nunique() != 1:
            raise ValueError(f"Final event {event} maps to multiple rainfall groups")

    rows: list[dict[str, object]] = []
    for _, item in index.iterrows():
        result = verify_formal_run_v4(
            str(item["formal_manifest_path"]),
            priority=priority,
            physical_sha=physical_sha,
            model_step_seconds=model_step,
            control_update_seconds=control_update,
        )
        for key in ("event_id", "rainfall_group", "strategy"):
            if str(result[key]) != str(item[key]):
                raise ValueError(f"Final index {key} differs from bound formal run")
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
    summary["independent_rainfall_groups"] = int(
        grouped["rainfall_group"].nunique()
    )
    summary["aggregation"] = "equal_weight_per_rainfall_group"

    pairs: dict[str, pd.DataFrame] = {}
    proposed = grouped[grouped["strategy"] == "proposed"].set_index(
        "rainfall_group"
    )
    for reference in strategies:
        if reference == "proposed":
            continue
        base = grouped[grouped["strategy"] == reference].set_index("rainfall_group")
        if set(proposed.index) != set(base.index):
            raise ValueError(
                f"unpaired Final rainfall groups for proposed vs {reference}"
            )
        records: list[dict[str, object]] = []
        for rainfall_group in sorted(proposed.index):
            row: dict[str, object] = {
                "rainfall_group": rainfall_group,
                "reference": reference,
            }
            for metric in (
                "tfv_m3",
                "priority_flood_volume_m3",
                "global_peak_flood_rate_m3s",
            ):
                p = float(proposed.loc[rainfall_group, metric])
                b = float(base.loc[rainfall_group, metric])
                row[f"delta_{metric}"] = p - b
                row[f"reduction_{metric}_pct"] = (
                    100.0 * (b - p) / b if abs(b) > 1e-12 else np.nan
                )
            records.append(row)
        pairs[reference] = pd.DataFrame(records)
    return detail, summary, pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compile locked rainfall-group-balanced TFV-first Formal Final"
    )
    parser.add_argument("--policy-lock", required=True)
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    detail, summary, pairs = compile_final_v4(
        policy_lock_path=args.policy_lock, run_index_path=args.run_index
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "formal_final_detail.csv", index=False)
    metric_cols = [
        "tfv_m3",
        "priority_flood_volume_m3",
        "global_peak_flood_rate_m3s",
        "main_flow_routing_error_pct",
        "peak_replay_flow_routing_error_pct",
    ]
    grouped = _group_detail(detail, metric_cols)
    grouped.to_csv(out / "formal_final_group_detail.csv", index=False)
    summary.to_csv(out / "formal_final_summary.csv", index=False)
    for reference, frame in pairs.items():
        frame.to_csv(out / f"proposed_vs_{reference}.csv", index=False)
    print(
        json.dumps(
            {
                "contract": "TFV_PRIMARY__PRIORITY_PFV_SOFT_SECONDARY_V1",
                "events": int(detail["event_id"].nunique()),
                "independent_rainfall_groups": int(
                    detail["rainfall_group"].nunique()
                ),
                "aggregation": "equal_weight_per_rainfall_group",
                "strategies": sorted(detail["strategy"].unique().tolist()),
                "priority_pfv_is_hard_gate": False,
                "detail": str(out / "formal_final_detail.csv"),
                "group_detail": str(out / "formal_final_group_detail.csv"),
                "summary": str(out / "formal_final_summary.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
