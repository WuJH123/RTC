"""Read-only D2/D3 coverage audit relative to the frozen V6/V7 MPC manifold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rtc.graph import GraphSchema
from rtc.step2_control_basis_v60 import build_control_basis_v60
from rtc.step2_manifold_semantics_v110 import (
    ManifoldDescriptorsV110,
    candidate_manifold_descriptors_v110,
    summarize_manifold_descriptors_v110,
)
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


def _concat(items: list[ManifoldDescriptorsV110]) -> ManifoldDescriptorsV110:
    if not items:
        raise ValueError("V11 manifold audit source is empty")
    fields = ManifoldDescriptorsV110.__dataclass_fields__
    return ManifoldDescriptorsV110(**{
        name: np.concatenate([getattr(item, name) for item in items], axis=0)
        for name in fields
    })


def _source(cache, names, basis):
    rows = []
    for name in names:
        entry = cache.entry(name)
        candidates = [i for i in entry.indices if i != entry.reference_index]
        reference = np.asarray(entry.arrays["settings"][entry.reference_index], dtype=np.float64)
        candidate = np.asarray(entry.arrays["settings"][candidates], dtype=np.float64)
        rows.append(candidate_manifold_descriptors_v110(reference, candidate, basis))
    return _concat(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Project7 V11 TrainFit MPC-manifold audit")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()

    graph = _load_graph(args.graph)
    cache = V60TrainCache(args.cache_manifest)
    validate_v60_cache_lineage(args.cache_manifest)
    if cache.legacy_d3_names():
        raise ValueError("V11 manifold audit refuses legacy dense D3")
    selected = sorted(cache.names("D2") + cache.targeted_d3_names())
    fit, holdout = deterministic_rainfall_split_v60(
        cache, names=selected, holdout_fraction=float(args.holdout_fraction)
    )
    fit_events = {cache.entry(name).event_id for name in fit}
    holdout_events = {cache.entry(name).event_id for name in holdout}
    fit_rain = {cache.entry(name).rainfall_group for name in fit}
    holdout_rain = {cache.entry(name).rainfall_group for name in holdout}
    if fit_events & holdout_events or fit_rain & holdout_rain:
        raise ValueError("V11 manifold audit split leakage detected")
    d2 = [name for name in fit if name.startswith("D2::")]
    d3 = [name for name in fit if name.startswith("D3::")]
    if not d2 or not d3:
        raise ValueError("V11 manifold audit requires TrainFit D2 and D3")

    basis = build_control_basis_v60(graph)
    summary = summarize_manifold_descriptors_v110(
        {"D2": _source(cache, d2, basis), "D3": _source(cache, d3, basis)}
    )
    summary["basis"] = {
        "raw_action_dimension": int(len(graph.actuator_ids) * basis.horizon.control_blocks),
        "coefficient_dimension": int(basis.coefficient_dimension),
        "control_groups": int(basis.group_count),
        "temporal_basis_count": int(basis.temporal_basis_count),
    }
    summary["boundary"] = {
        "trainfit_only": True,
        "holdout_outcomes_accessed": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
        "swmm_run": False,
        "model_trained": False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
