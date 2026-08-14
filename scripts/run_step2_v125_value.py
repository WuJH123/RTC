"""V125 TFV data-support ablation: fixed V124 architecture + D4-FIT only."""
from __future__ import annotations

import argparse
import hashlib
import json
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
from rtc.step2_control_value_v124 import (
    ControlValueSurrogateV124,
    V124_VALUE_CONTRACT,
    ValueLossContractV124,
    train_value_event_balanced_v124,
)
from rtc.step2_d4_cache_v125 import (
    CompositeValueCacheV125,
    D4CausalForecastValueCacheV125,
    D4_SOURCE_KIND,
)
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_train_response_v70 import derive_target_scales_v70, evaluate_value_v70
from rtc.step2_v120_train_helpers import load_graph_v120

V125_VALUE_TRAINING_CONTRACT = "PROJECT7_V125_TFV_D4_DATA_SUPPORT_ABLATION_V1"


def _rain(cache, names):
    return {str(cache.entry(name).rainfall_group) for name in names}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True, help="frozen V60 D2/D3 cache")
    p.add_argument("--d4-fit-cache", required=True)
    p.add_argument("--d4-audit-cache", required=True)
    p.add_argument("--causal-store", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--holdout-fraction", type=float, default=0.20)
    p.add_argument("--hidden-dim", type=int, default=96)
    p.add_argument("--attention-heads", type=int, default=4)
    p.add_argument("--listwise-weight", type=float, default=0.30)
    args = p.parse_args()
    if (
        int(args.seed) != 42
        or abs(float(args.holdout_fraction) - 0.20) > 1e-12
        or int(args.hidden_dim) != 96
        or int(args.attention_heads) != 4
        or abs(float(args.listwise_weight) - 0.30) > 1e-12
    ):
        raise ValueError("V125 first ablation is frozen at seed42/holdout.20/hidden96/heads4/listwise.30")

    base = V60TrainCache(args.cache_manifest)
    d4_fit_base = V60TrainCache(args.d4_fit_cache)
    d4_audit_base = V60TrainCache(args.d4_audit_cache)
    store = load_causal_forecast_store_v123(args.causal_store)
    base_causal = CausalForecastValueCacheV123(base, store)
    d4_fit = D4CausalForecastValueCacheV125(d4_fit_base, store)
    d4_audit = D4CausalForecastValueCacheV125(d4_audit_base, store)

    names = sorted(base.names("D2") + base.targeted_d3_names())
    fit, holdout = deterministic_rainfall_split_v60(base, names=names, holdout_fraction=0.20)
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    fit_d3 = [name for name in fit if name.startswith("D3::")]
    holdout_d2 = [name for name in holdout if name.startswith("D2::")]
    holdout_d3 = [name for name in holdout if name.startswith("D3::")]
    if (len(fit_d2), len(fit_d3), len(holdout_d2), len(holdout_d3)) != (112, 112, 32, 32):
        raise ValueError("V125 base split differs from frozen V124 112/112/32/32 contract")
    fit_d4 = d4_fit.names(D4_SOURCE_KIND)
    audit_d4 = d4_audit.names(D4_SOURCE_KIND)
    if not fit_d4 or not audit_d4:
        raise ValueError("V125 requires non-empty physically separate D4 FIT and AUDIT caches")
    if _rain(d4_fit_base, fit_d4) & _rain(d4_audit_base, audit_d4):
        raise ValueError("V125 D4 FIT/AUDIT rainfall groups overlap")

    graph = load_graph_v120(args.graph)
    basis = build_control_basis_v60(graph)
    # Frozen V124 normalization/scales: D4 must not move these knobs in the data ablation.
    normalization = derive_causal_input_normalization_v123(base, store, fit)
    scales = derive_target_scales_v70(base, fit)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    prepared = prepare_static_v60(graph, device)
    first = base.entry(fit_d2[0]).arrays
    model = ControlValueSurrogateV124(
        state_dim=int(first["initial_state"].shape[-1]),
        rainfall_dim=int(first["rainfall"].shape[-1]),
        physics_dim=int(prepared.actuator_physics.shape[1]),
        actuator_count=len(graph.actuator_ids),
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        tfv_scale_m3=float(scales.direct_tfv_scale_m3),
        hidden_dim=96,
        actuator_embedding_dim=16,
        attention_heads=4,
    )
    training_cache = CompositeValueCacheV125((base_causal, d4_fit))
    loss_contract = ValueLossContractV124(listwise_weight=0.30)
    history = train_value_event_balanced_v124(
        model,
        training_cache,
        fit_d2_names=fit_d2,
        fit_d3_names=fit_d3 + fit_d4,
        normalization=normalization,
        scales=scales,
        graph=graph,
        device=device,
        seed=42,
        contract=loss_contract,
    )
    metrics = {
        "base_trainfit_d2": evaluate_value_v70(model, base_causal, fit_d2, normalization, prepared, device=device),
        "base_trainfit_d3": evaluate_value_v70(model, base_causal, fit_d3, normalization, prepared, device=device),
        "base_internal_holdout_d2": evaluate_value_v70(model, base_causal, holdout_d2, normalization, prepared, device=device),
        "base_internal_holdout_d3": evaluate_value_v70(model, base_causal, holdout_d3, normalization, prepared, device=device),
        "d4_fit_anchor_neighbourhood": evaluate_value_v70(model, d4_fit, fit_d4, normalization, prepared, device=device),
        "d4_audit_anchor_neighbourhood": evaluate_value_v70(model, d4_audit, audit_d4, normalization, prepared, device=device),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "v125_tfv_d4_causal.pt"
    torch.save(
        {
            # Runtime architecture contract remains V124 so the frozen V123/V125 loader
            # instantiates exactly the same network; scientific training lineage is separate.
            "contract": V124_VALUE_CONTRACT,
            "training_contract": V125_VALUE_TRAINING_CONTRACT,
            "architecture": "interaction_aware_active_actuator_self_attention",
            "state_dict": model.state_dict(),
            "hidden_dim": 96,
            "attention_heads": 4,
            "listwise_weight": 0.30,
            "d4_fit_cache_sha256": hashlib.sha256(Path(args.d4_fit_cache).read_bytes()).hexdigest(),
        },
        checkpoint,
    )
    payload = {
        "contract": V125_VALUE_TRAINING_CONTRACT,
        "runtime_architecture_contract": V124_VALUE_CONTRACT,
        "source_tree_sha256": rtc_implementation_contract_sha256(),
        "base_cache_manifest": str(Path(args.cache_manifest).resolve()),
        "d4_fit_cache": str(Path(args.d4_fit_cache).resolve()),
        "d4_audit_cache": str(Path(args.d4_audit_cache).resolve()),
        "causal_store": str(Path(args.causal_store).resolve()),
        "seed": 42,
        "architecture": {"hidden_dim": 96, "attention_heads": 4, "listwise_weight": 0.30},
        "only_experimental_change": "additional D4-FIT anchor-neighbourhood groups in unchanged V124 multi-action training stage",
        "normalization_and_scale_source": "frozen base V124 TrainFit D2/D3 only",
        "split": {
            "base_fit_d2": len(fit_d2), "base_fit_d3": len(fit_d3),
            "base_holdout_d2": len(holdout_d2), "base_holdout_d3": len(holdout_d3),
            "d4_fit": len(fit_d4), "d4_audit": len(audit_d4),
            "d4_fit_rainfall_groups": sorted(_rain(d4_fit_base, fit_d4)),
            "d4_audit_rainfall_groups": sorted(_rain(d4_audit_base, audit_d4)),
        },
        "target_scale_tfv_m3": float(scales.direct_tfv_scale_m3),
        "training_history": history,
        "metrics": metrics,
        "checkpoint": str(checkpoint.resolve()),
        "boundary": {
            "d4_audit_used_for_training": False,
            "d4_audit_used_for_normalization": False,
            "new_swmm_during_training": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
            "continuous_mpc_authorized": False,
            "d4_audit_is_action_support_holdout_not_independent_state_rainfall_validation": True,
        },
    }
    report = out / "STEP2_V125_TFV_D4_VALUE_REPORT.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    audit = metrics["d4_audit_anchor_neighbourhood"]
    hold = metrics["base_internal_holdout_d3"]
    report.with_suffix(".md").write_text(
        "\n".join([
            "# STEP2 V125 TFV D4 data-support ablation", "",
            f"Base InternalHoldout D3 rank: {hold['rank']:.4f}",
            f"Base InternalHoldout D3 pairwise: {hold['pairwise']:.4f}",
            f"D4-AUDIT anchor-neighbourhood rank: {audit['rank']:.4f}",
            f"D4-AUDIT pairwise: {audit['pairwise']:.4f}",
            f"D4-AUDIT top1: {audit['top1_rate']:.4f}",
            f"D4-AUDIT mean regret: {audit['mean_regret_m3']:.1f} m3", "",
            "D4-AUDIT is an action-support holdout at known TrainFit rainfall/state groups; generic InternalHoldout remains the independent rainfall/state generalization diagnostic.",
            "Continuous MPC remains blocked.",
        ]) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
