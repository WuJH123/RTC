"""Train/evaluate the independent causal-rainfall priority-flood Value model."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from rtc.code_contract import rtc_implementation_contract_sha256
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_control_basis_v60 import build_control_basis_v60
from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_control_response_v70 import ControlValueSurrogateV70
from rtc.step2_optimization_v70 import train_value_event_balanced_v70
from rtc.step2_priority_value_v123 import (
    PriorityValueCacheV123,
    V123_PFV_LABEL_CONTRACT,
    group_pfv_labels_v123,
)
from rtc.step2_train_response_v60 import V60TrainCache, derive_input_normalization_v60, deterministic_rainfall_split_v60
from rtc.step2_train_response_v70 import derive_target_scales_v70, evaluate_value_v70
from rtc.step2_v120_train_helpers import load_graph_v120
from rtc.step2_v70_contract import DirectValueLossContractV70
from rtc.step2_pfv_training_v123 import derive_priority_target_scale_v123


def _read_nodes(path: str | Path) -> tuple[str, ...]:
    nodes = tuple(
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(nodes) != 8 or len(set(nodes)) != len(nodes):
        raise ValueError("V123 PFV requires the frozen unique priority-node list of length 8")
    return nodes


def _metric_aliases(metric: dict) -> dict:
    result = dict(metric)
    if "tfv_mae_m3" in result:
        result["pfv_mae_m3"] = result.pop("tfv_mae_m3")
    for key in ("truth_spread_m3", "predicted_spread_m3", "mean_abs_truth_m3", "mean_abs_prediction_m3"):
        if key in result:
            result[key.replace("truth", "pfv_truth").replace("prediction", "pfv_prediction")] = result[key]
    result["target"] = "PFV"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="V123 causal PFV Value training")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--causal-store", required=True)
    parser.add_argument("--priority-nodes", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if int(args.seed) != 42 or abs(float(args.holdout_fraction) - 0.20) > 1e-12:
        raise ValueError("V123 PFV training is frozen at seed=42 and holdout_fraction=0.20")

    base = V60TrainCache(args.cache_manifest)
    store = load_causal_forecast_store_v123(args.causal_store)
    causal = CausalForecastValueCacheV123(base, store)
    priority_nodes = _read_nodes(args.priority_nodes)
    cache = PriorityValueCacheV123(causal, priority_nodes)
    names = sorted(base.names("D2") + base.targeted_d3_names())
    fit, holdout = deterministic_rainfall_split_v60(base, names=names, holdout_fraction=0.20)
    fit_d2 = [n for n in fit if n.startswith("D2::")]
    fit_d3 = [n for n in fit if n.startswith("D3::")]
    holdout_d2 = [n for n in holdout if n.startswith("D2::")]
    holdout_d3 = [n for n in holdout if n.startswith("D3::")]
    graph = load_graph_v120(args.graph)
    basis = build_control_basis_v60(graph)
    prepared = prepare_static_v60(graph, args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    normalization = derive_input_normalization_v60(base, fit)
    base_scales = derive_target_scales_v70(base, fit)
    pfv_values = [group_pfv_labels_v123(base.entry(name), priority_nodes).delta_pfv_m3 for name in fit]
    pfv_scale = derive_priority_target_scale_v123(pfv_values, minimum_m3=100.0)
    scales = replace(base_scales, direct_tfv_scale_m3=pfv_scale)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    first = base.entry(fit_d2[0]).arrays
    model = ControlValueSurrogateV70(
        state_dim=int(first["initial_state"].shape[-1]),
        rainfall_dim=int(first["rainfall"].shape[-1]),
        physics_dim=int(prepared.actuator_physics.shape[1]),
        actuator_count=len(graph.actuator_ids),
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        tfv_scale_m3=float(pfv_scale),
        hidden_dim=96,
        actuator_embedding_dim=16,
    )
    history = train_value_event_balanced_v70(
        model, cache, fit_d2_names=fit_d2, fit_d3_names=fit_d3,
        normalization=normalization, scales=scales, graph=graph,
        device=device, seed=42, contract=DirectValueLossContractV70(),
    )
    metrics = {}
    for label, subset in (("trainfit_d2", fit_d2), ("trainfit_d3", fit_d3), ("holdout_d2", holdout_d2), ("holdout_d3", holdout_d3)):
        metrics[label] = _metric_aliases(evaluate_value_v70(model, cache, subset, normalization, prepared, device=device))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"target": "PFV", "state_dict": model.state_dict(), "production_compatible": False}, out / "v123_pfv_causal.pt")
    payload = {
        "contract": "PROJECT7_V123_PFV_CAUSAL_TRAINING_V1",
        "label_contract": V123_PFV_LABEL_CONTRACT,
        "target": "signed DeltaPFV m3",
        "priority_nodes": list(priority_nodes),
        "priority_nodes_sha256": hashlib.sha256(Path(args.priority_nodes).read_bytes()).hexdigest(),
        "seed": 42,
        "holdout_fraction": 0.20,
        "device": device,
        "source_tree_sha256": rtc_implementation_contract_sha256(),
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "causal_store": str(Path(args.causal_store).resolve()),
        "causal_store_sha256": hashlib.sha256(Path(args.causal_store).read_bytes()).hexdigest(),
        "rainfall_input": "V123_parent_no_control_causal_forecast",
        "future_realized_rainfall_used_as_model_input": False,
        "target_scale_pfv_m3": float(pfv_scale),
        "split": {"fit_groups": len(fit), "holdout_groups": len(holdout), "fit_d2": len(fit_d2), "fit_d3": len(fit_d3), "holdout_d2": len(holdout_d2), "holdout_d3": len(holdout_d3), "fit_events": sorted({base.entry(n).event_id for n in fit}), "holdout_events": sorted({base.entry(n).event_id for n in holdout})},
        "training_history": history,
        "metrics": metrics,
        "boundary": {"new_swmm": False, "validation_accessed": False, "final_accessed": False, "formal_accessed": False, "production_compatible": False},
    }
    (out / "STEP2_V123_PFV_VALUE_REPORT.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# STEP2 V123 PFV VALUE", "", "| subset | rank | pairwise | sign | top1 | PFV MAE m3 | regret m3 | spread ratio | response ratio |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for subset, metric in metrics.items():
        lines.append(f"| {subset} | {metric['rank']:.4f} | {metric['pairwise']:.4f} | {metric['sign_accuracy']:.4f} | {metric['top1_rate']:.4f} | {metric['pfv_mae_m3']:.1f} | {metric['mean_regret_m3']:.1f} | {metric['spread_ratio']:.4f} | {metric['response_ratio']:.4f} |")
    (out / "STEP2_V123_PFV_VALUE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
