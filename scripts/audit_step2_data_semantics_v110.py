"""Mine mechanism structure from the existing Train-only D2/D3 V11 assets.

This is a read-only data audit. It does not train, run SWMM, access Validation or
Final, or change any acceptance threshold. Future SWMM outcome descriptors are
written only as offline labels/strata and are explicitly forbidden as runtime
inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from rtc.graph import GraphSchema
from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_control_response_v110 import build_actuator_node_relations_v110
from rtc.step2_data_semantics_v110 import (
    mechanism_records_v110,
    summarize_mechanism_records_v110,
)
from rtc.step2_hydraulic_objective_v110 import derive_effect_scales_v110
from rtc.step2_shards_v60 import validate_v60_cache_lineage
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60


def _load_graph(path: str | Path) -> GraphSchema:
    with np.load(path, allow_pickle=False) as raw:
        return GraphSchema(
            node_ids=tuple(raw["node_ids"].astype(str).tolist()),
            edge_index=raw["edge_index"].astype(np.int64),
            static_node_features=raw["static_node_features"].astype(np.float32),
            static_node_feature_names=tuple(raw["static_node_feature_names"].astype(str).tolist()),
            actuator_ids=tuple(raw["actuator_ids"].astype(str).tolist()),
            actuator_upstream=raw["actuator_upstream"].astype(np.int64),
            actuator_downstream=raw["actuator_downstream"].astype(np.int64),
            actuator_physics=raw["actuator_physics"].astype(np.float32),
            actuator_physics_feature_names=tuple(raw["actuator_physics_feature_names"].astype(str).tolist()),
            system_units=str(raw["system_units"].item()),
        )


def _bin_counts(values, edges, labels):
    x = np.asarray(values, dtype=np.float64)
    bins = np.digitize(x, np.asarray(edges, dtype=np.float64), right=False)
    return {label: int((bins == index).sum()) for index, label in enumerate(labels)}


def _source_summary(records):
    result = {}
    for source in sorted({record.source_kind for record in records}):
        subset = [record for record in records if record.source_kind == source]
        onset = [record.onset_min for record in subset]
        remote = [record.remote_effect_fraction_gt8 for record in subset]
        changed = [record.changed_actuator_count for record in subset]
        tfv = np.asarray([record.delta_tfv_m3 for record in subset], dtype=np.float64)
        result[source] = {
            "records": len(subset),
            "groups": len({record.group_name for record in subset}),
            "events": len({record.event_id for record in subset}),
            "onset_bins_min": _bin_counts(onset, [1e-9, 15, 30, 60, 90], ["zero", "0_15", "15_30", "30_60", "60_90", "90_plus"]),
            "remote_effect_fraction_bins": _bin_counts(remote, [1e-9, 0.10, 0.25, 0.50, 0.75], ["zero", "0_10pct", "10_25pct", "25_50pct", "50_75pct", "75pct_plus"]),
            "changed_actuator_count_bins": _bin_counts(changed, [1, 2, 5, 10, 20, 50, 100], ["zero", "1", "2_4", "5_9", "10_19", "20_49", "50_99", "100_plus"]),
            "delta_tfv_abs_quantiles_m3": {
                "q10": float(np.quantile(np.abs(tfv), 0.10)),
                "q50": float(np.quantile(np.abs(tfv), 0.50)),
                "q90": float(np.quantile(np.abs(tfv), 0.90)),
            },
        }
    return result


def _group_contrast(records):
    by_group = {}
    for record in records:
        by_group.setdefault(record.group_name, []).append(record)
    rows = []
    for name, group in sorted(by_group.items()):
        tfv = np.asarray([r.delta_tfv_m3 for r in group], dtype=np.float64)
        energy = np.asarray([r.hydraulic_effect_energy for r in group], dtype=np.float64)
        remote = np.asarray([r.remote_effect_fraction_gt8 for r in group], dtype=np.float64)
        rows.append({
            "group_name": name,
            "source_kind": group[0].source_kind,
            "event_id": group[0].event_id,
            "rainfall_group": group[0].rainfall_group,
            "checkpoint_id": group[0].checkpoint_id,
            "candidate_count": len(group),
            "delta_tfv_spread_m3": float(tfv.max() - tfv.min()),
            "delta_tfv_best_m3": float(tfv.min()),
            "delta_tfv_worst_m3": float(tfv.max()),
            "hydraulic_energy_spread": float(energy.max() - energy.min()),
            "remote_fraction_spread": float(remote.max() - remote.min()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Project7 V11 D2/D3 mechanism audit")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()

    graph = _load_graph(args.graph)
    if len(graph.node_ids) != 932 or len(graph.actuator_ids) != 109:
        raise ValueError("V11 mechanism audit requires frozen Project7 graph")
    cache = V60TrainCache(args.cache_manifest)
    validate_v60_cache_lineage(args.cache_manifest)
    if cache.legacy_d3_names():
        raise ValueError("V11 mechanism audit refuses legacy dense D3")
    d2 = cache.names("D2")
    d3 = cache.targeted_d3_names()
    fit, holdout = deterministic_rainfall_split_v60(
        cache, names=sorted(d2 + d3), holdout_fraction=float(args.holdout_fraction)
    )
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    fit_d3 = [name for name in fit if name.startswith("D3::")]
    if not fit_d2 or not fit_d3:
        raise ValueError("V11 mechanism audit requires TrainFit D2 and D3")
    fit_events = {cache.entry(name).event_id for name in fit}
    holdout_events = {cache.entry(name).event_id for name in holdout}
    fit_rain = {cache.entry(name).rainfall_group for name in fit}
    holdout_rain = {cache.entry(name).rainfall_group for name in holdout}
    if fit_events & holdout_events or fit_rain & holdout_rain:
        raise ValueError("V11 mechanism audit split leakage detected")

    prepared = prepare_static_v60(graph, "cpu")
    scales = derive_effect_scales_v110(cache, fit_d2, prepared)
    relations = build_actuator_node_relations_v110(graph)
    records = mechanism_records_v110(
        cache, sorted(fit_d2 + fit_d3), prepared, relations, scales
    )
    summary = summarize_mechanism_records_v110(records)
    summary.update(
        {
            "boundary": {
                "trainfit_only": True,
                "holdout_outcomes_used_for_feature_mining": False,
                "validation_accessed": False,
                "final_accessed": False,
                "formal_accessed": False,
                "swmm_run": False,
                "model_trained": False,
            },
            "data": {
                "all_d2_groups": len(d2),
                "all_targeted_d3_groups": len(d3),
                "trainfit_d2_groups": len(fit_d2),
                "trainfit_d3_groups": len(fit_d3),
                "internal_holdout_groups": len(holdout),
                "event_overlap": 0,
                "rainfall_overlap": 0,
            },
            "source_summary": _source_summary(records),
        }
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    record_path = out / "STEP2_V110_MECHANISM_RECORDS_TRAINFIT.csv"
    with record_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    contrast = _group_contrast(records)
    contrast_path = out / "STEP2_V110_GROUP_CONTRAST_TRAINFIT.csv"
    with contrast_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrast[0].keys()))
        writer.writeheader()
        writer.writerows(contrast)
    summary["artifacts"] = {
        "candidate_mechanism_records": str(record_path.resolve()),
        "group_contrast": str(contrast_path.resolve()),
    }
    summary_path = out / "STEP2_V110_DATA_SEMANTICS_AUDIT.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
