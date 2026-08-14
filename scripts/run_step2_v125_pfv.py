"""V125 PFV data-support ablation using the same D4-FIT anchor neighbourhood."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from rtc.code_contract import rtc_implementation_contract_sha256
from rtc.step2_causal_rainfall_v123 import (
    CausalForecastValueCacheV123,
    derive_causal_input_normalization_v123,
    load_causal_forecast_store_v123,
)
from rtc.step2_control_basis_v60 import build_control_basis_v60
from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_control_response_v70 import ControlValueSurrogateV70
from rtc.step2_d4_cache_v125 import (
    CompositeValueCacheV125,
    D4CausalForecastValueCacheV125,
    D4_SOURCE_KIND,
)
from rtc.step2_optimization_v70 import train_value_event_balanced_v70
from rtc.step2_pfv_training_v123 import derive_priority_target_scale_v123
from rtc.step2_priority_value_v123 import PriorityValueCacheV123, group_pfv_labels_v123
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_train_response_v70 import derive_target_scales_v70, evaluate_value_v70
from rtc.step2_v120_train_helpers import load_graph_v120
from rtc.step2_v70_contract import DirectValueLossContractV70

V125_PFV_TRAINING_CONTRACT = "PROJECT7_V125_PFV_D4_SOFT_PROTECTION_ABLATION_V1"


def _read_nodes(path: str | Path) -> tuple[str, ...]:
    nodes = tuple(
        line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(nodes) != 8 or len(set(nodes)) != 8:
        raise ValueError("V125 PFV requires the frozen unique Priority8 node list")
    return nodes


def _rain(cache, names):
    return {str(cache.entry(name).rainfall_group) for name in names}


def _alias(metric: dict) -> dict:
    out = dict(metric)
    if "tfv_mae_m3" in out:
        out["pfv_mae_m3"] = out.pop("tfv_mae_m3")
    out["target"] = "PFV"
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--d4-fit-cache", required=True)
    p.add_argument("--d4-audit-cache", required=True)
    p.add_argument("--causal-store", required=True)
    p.add_argument("--priority-nodes", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if int(args.seed) != 42:
        raise ValueError("V125 PFV first ablation is frozen at seed=42")

    base = V60TrainCache(args.cache_manifest)
    d4_fit_base = V60TrainCache(args.d4_fit_cache)
    d4_audit_base = V60TrainCache(args.d4_audit_cache)
    store = load_causal_forecast_store_v123(args.causal_store)
    base_causal = CausalForecastValueCacheV123(base, store)
    d4_fit = D4CausalForecastValueCacheV125(d4_fit_base, store)
    d4_audit = D4CausalForecastValueCacheV125(d4_audit_base, store)
    priority_nodes = _read_nodes(args.priority_nodes)

    names = sorted(base.names("D2") + base.targeted_d3_names())
    fit, holdout = deterministic_rainfall_split_v60(base, names=names, holdout_fraction=0.20)
    fit_d2 = [n for n in fit if n.startswith("D2::")]
    fit_d3 = [n for n in fit if n.startswith("D3::")]
    holdout_d2 = [n for n in holdout if n.startswith("D2::")]
    holdout_d3 = [n for n in holdout if n.startswith("D3::")]
    if (len(fit_d2), len(fit_d3), len(holdout_d2), len(holdout_d3)) != (112, 112, 32, 32):
        raise ValueError("V125 PFV base split differs from frozen 112/112/32/32")
    fit_d4 = d4_fit.names(D4_SOURCE_KIND)
    audit_d4 = d4_audit.names(D4_SOURCE_KIND)
    if not fit_d4 or not audit_d4 or (_rain(d4_fit_base, fit_d4) & _rain(d4_audit_base, audit_d4)):
        raise ValueError("V125 PFV requires disjoint non-empty D4 FIT/AUDIT caches")

    graph = load_graph_v120(args.graph)
    basis = build_control_basis_v60(graph)
    normalization = derive_causal_input_normalization_v123(base, store, fit)
    base_scales = derive_target_scales_v70(base, fit)
    # Freeze the PFV scale to the original base TrainFit distribution so this remains
    # a pure action-support experiment rather than a target-rescaling experiment.
    base_pfv = [
        group_pfv_labels_v123(base.entry(name), priority_nodes).delta_pfv_m3
        for name in fit
    ]
    pfv_scale = derive_priority_target_scale_v123(base_pfv, minimum_m3=100.0)
    scales = replace(base_scales, direct_tfv_scale_m3=pfv_scale)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    prepared = prepare_static_v60(graph, device)
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
    training_base = CompositeValueCacheV125((base_causal, d4_fit))
    training = PriorityValueCacheV123(training_base, priority_nodes)
    history = train_value_event_balanced_v70(
        model,
        training,
        fit_d2_names=fit_d2,
        fit_d3_names=fit_d3 + fit_d4,
        normalization=normalization,
        scales=scales,
        graph=graph,
        device=device,
        seed=42,
        contract=DirectValueLossContractV70(),
    )
    base_priority = PriorityValueCacheV123(base_causal, priority_nodes)
    fit_priority = PriorityValueCacheV123(d4_fit, priority_nodes)
    audit_priority = PriorityValueCacheV123(d4_audit, priority_nodes)
    metrics = {
        "base_internal_holdout_d2": _alias(evaluate_value_v70(model, base_priority, holdout_d2, normalization, prepared, device=device)),
        "base_internal_holdout_d3": _alias(evaluate_value_v70(model, base_priority, holdout_d3, normalization, prepared, device=device)),
        "d4_fit_anchor_neighbourhood": _alias(evaluate_value_v70(model, fit_priority, fit_d4, normalization, prepared, device=device)),
        "d4_audit_anchor_neighbourhood": _alias(evaluate_value_v70(model, audit_priority, audit_d4, normalization, prepared, device=device)),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "v125_pfv_d4_causal.pt"
    torch.save(
        {
            "target": "PFV",
            "training_contract": V125_PFV_TRAINING_CONTRACT,
            "state_dict": model.state_dict(),
            "production_compatible": False,
        },
        checkpoint,
    )
    payload = {
        "contract": V125_PFV_TRAINING_CONTRACT,
        "target": "signed anchor-relative-capable DeltaPFV m3",
        "source_tree_sha256": rtc_implementation_contract_sha256(),
        "priority_nodes": list(priority_nodes),
        "priority_nodes_sha256": hashlib.sha256(Path(args.priority_nodes).read_bytes()).hexdigest(),
        "seed": 42,
        "target_scale_pfv_m3": float(pfv_scale),
        "base_cache_manifest": str(Path(args.cache_manifest).resolve()),
        "d4_fit_cache": str(Path(args.d4_fit_cache).resolve()),
        "d4_audit_cache": str(Path(args.d4_audit_cache).resolve()),
        "training_history": history,
        "metrics": metrics,
        "checkpoint": str(checkpoint.resolve()),
        "boundary": {
            "pfv_role": "one_sided_soft_deterioration_protection",
            "pfv_can_buy_worse_tfv": False,
            "d4_audit_used_for_training": False,
            "d4_audit_used_for_scale": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
            "production_compatible": False,
        },
    }
    report = out / "STEP2_V125_PFV_D4_VALUE_REPORT.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
