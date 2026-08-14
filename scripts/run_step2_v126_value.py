"""Train Project7 V126 TFV Value with a source-aware D2 -> D3 -> D4 curriculum.

This is deliberately not a new architecture sweep. The V124 interaction-aware model and
seed remain fixed. The scientific change is training semantics: broad D2/targeted-D3
representation learning is completed first; D4-FIT then receives a dedicated local
anchor-advantage fine-tune loss and scale. D4-AUDIT and InternalHoldout remain read-only.
"""
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
from rtc.step2_curriculum_v126 import (
    AnchorFineTuneContractV126,
    V126_CURRICULUM_CONTRACT,
    derive_anchor_tfv_scale_v126,
    train_anchor_advantage_finetune_v126,
)
from rtc.step2_d4_cache_v125 import D4CausalForecastValueCacheV125, D4_SOURCE_KIND
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_train_response_v70 import derive_target_scales_v70, evaluate_value_v70
from rtc.step2_v120_train_helpers import load_graph_v120

V126_VALUE_TRAINING_CONTRACT = "PROJECT7_V126_TFV_SOURCE_AWARE_CURRICULUM_V1"


def _rain(cache, names):
    return {str(cache.entry(name).rainfall_group) for name in names}


def _branches(cache, names) -> int:
    return int(sum(len(cache.entry(name).indices) for name in names))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True, help="frozen canonical V60 D2/targeted-D3 cache")
    p.add_argument("--d4-fit-cache", required=True)
    p.add_argument("--d4-audit-cache", required=True)
    p.add_argument("--causal-store", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

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
        raise ValueError("V126 base split differs from frozen 112/112/32/32 contract")
    fit_d4 = d4_fit.names(D4_SOURCE_KIND)
    audit_d4 = d4_audit.names(D4_SOURCE_KIND)
    if not fit_d4 or not audit_d4:
        raise ValueError("V126 requires non-empty physically separate D4 FIT/AUDIT caches")
    if _rain(d4_fit_base, fit_d4) & _rain(d4_audit_base, audit_d4):
        raise ValueError("V126 D4 FIT/AUDIT rainfall groups overlap")

    graph = load_graph_v120(args.graph)
    basis = build_control_basis_v60(graph)
    normalization = derive_causal_input_normalization_v123(base, store, fit)
    broad_scales = derive_target_scales_v70(base, fit)
    runtime_tfv_scale = float(broad_scales.direct_tfv_scale_m3)
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
        tfv_scale_m3=runtime_tfv_scale,
        hidden_dim=96,
        actuator_embedding_dim=16,
        attention_heads=4,
    )

    # Stage A/B: broad causal D2 + targeted D3 only. D4 is intentionally absent here.
    broad_history = train_value_event_balanced_v124(
        model,
        base_causal,
        fit_d2_names=fit_d2,
        fit_d3_names=fit_d3,
        normalization=normalization,
        scales=broad_scales,
        graph=graph,
        device=device,
        seed=42,
        contract=ValueLossContractV124(listwise_weight=0.30),
    )
    before = {
        "base_internal_holdout_d3": evaluate_value_v70(
            model, base_causal, holdout_d3, normalization, prepared, device=device
        ),
        "d4_fit_anchor_neighbourhood": evaluate_value_v70(
            model, d4_fit, fit_d4, normalization, prepared, device=device
        ),
        "d4_audit_anchor_neighbourhood": evaluate_value_v70(
            model, d4_audit, audit_d4, normalization, prepared, device=device
        ),
    }

    # Stage C: decision-task fine-tune. This local loss scale does not alter the model's
    # physical output transform; runtime must keep using the broad V124 TFV scale.
    local_scale = derive_anchor_tfv_scale_v126(d4_fit, fit_d4)
    fine_contract = AnchorFineTuneContractV126()
    d4_history = train_anchor_advantage_finetune_v126(
        model,
        d4_fit,
        fit_d4_names=fit_d4,
        normalization=normalization,
        prepared=prepared,
        device=device,
        scale_m3=local_scale,
        seed=42,
        contract=fine_contract,
    )
    after = {
        "base_trainfit_d2": evaluate_value_v70(
            model, base_causal, fit_d2, normalization, prepared, device=device
        ),
        "base_trainfit_d3": evaluate_value_v70(
            model, base_causal, fit_d3, normalization, prepared, device=device
        ),
        "base_internal_holdout_d2": evaluate_value_v70(
            model, base_causal, holdout_d2, normalization, prepared, device=device
        ),
        "base_internal_holdout_d3": evaluate_value_v70(
            model, base_causal, holdout_d3, normalization, prepared, device=device
        ),
        "d4_fit_anchor_neighbourhood": evaluate_value_v70(
            model, d4_fit, fit_d4, normalization, prepared, device=device
        ),
        "d4_audit_anchor_neighbourhood": evaluate_value_v70(
            model, d4_audit, audit_d4, normalization, prepared, device=device
        ),
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "v126_tfv_source_aware_causal.pt"
    torch.save(
        {
            # Runtime topology is unchanged: existing V124 loader remains valid.
            "contract": V124_VALUE_CONTRACT,
            "training_contract": V126_VALUE_TRAINING_CONTRACT,
            "curriculum_contract": V126_CURRICULUM_CONTRACT,
            "architecture": "V124_interaction_attention_source_aware_finetuned",
            "state_dict": model.state_dict(),
            "hidden_dim": 96,
            "attention_heads": 4,
            "listwise_weight": 0.30,
            "target_scale_tfv_m3": runtime_tfv_scale,
            "broad_tfv_scale_m3": runtime_tfv_scale,
            "d4_local_loss_scale_m3": float(local_scale),
            "d4_fit_cache_sha256": hashlib.sha256(Path(args.d4_fit_cache).read_bytes()).hexdigest(),
        },
        checkpoint,
    )
    payload = {
        "contract": V126_VALUE_TRAINING_CONTRACT,
        "curriculum_contract": V126_CURRICULUM_CONTRACT,
        "runtime_architecture_contract": V124_VALUE_CONTRACT,
        "source_tree_sha256": rtc_implementation_contract_sha256(),
        "seed": 42,
        "scientific_change": "source-aware sequential curriculum; no architecture/seed sweep",
        "broad_stage": "D2 then targeted D3 using frozen V124 objective",
        "decision_stage": "D4-FIT-only anchor-advantage fine-tune",
        "target_scale_tfv_m3": runtime_tfv_scale,
        "broad_tfv_scale_m3": runtime_tfv_scale,
        "d4_local_loss_scale_m3": float(local_scale),
        "fit_counts": {
            "d2_groups": len(fit_d2),
            "d2_branches": _branches(base, fit_d2),
            "targeted_d3_groups": len(fit_d3),
            "targeted_d3_branches": _branches(base, fit_d3),
            "d4_groups": len(fit_d4),
            "d4_branches": _branches(d4_fit_base, fit_d4),
        },
        "holdout_counts": {
            "d2_groups": len(holdout_d2),
            "d2_branches": _branches(base, holdout_d2),
            "targeted_d3_groups": len(holdout_d3),
            "targeted_d3_branches": _branches(base, holdout_d3),
            "d4_audit_groups": len(audit_d4),
            "d4_audit_branches": _branches(d4_audit_base, audit_d4),
        },
        "d4_fit_rainfall_groups": sorted(_rain(d4_fit_base, fit_d4)),
        "d4_audit_rainfall_groups": sorted(_rain(d4_audit_base, audit_d4)),
        "broad_history": broad_history,
        "d4_finetune_history": d4_history,
        "metrics_before_d4_finetune": before,
        "metrics_after_d4_finetune": after,
        "checkpoint": str(checkpoint.resolve()),
        "boundary": {
            "d4_audit_used_for_training": False,
            "internal_holdout_used_for_training": False,
            "legacy_d3_used_for_training": False,
            "development_validation_d2_used_for_training": False,
            "new_swmm": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
            "continuous_mpc_authorized": False,
        },
    }
    report = out / "STEP2_V126_SOURCE_AWARE_VALUE_REPORT.json"
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    audit_before = before["d4_audit_anchor_neighbourhood"]
    audit_after = after["d4_audit_anchor_neighbourhood"]
    generic_after = after["base_internal_holdout_d3"]
    report.with_suffix(".md").write_text(
        "\n".join(
            [
                "# Project7 Step2 V126 source-aware curriculum",
                "",
                f"Runtime TFV transform scale: {runtime_tfv_scale:.3f} m3",
                f"D4 local loss scale (FIT-only): {local_scale:.3f} m3",
                f"D4-AUDIT rank before fine-tune: {audit_before['rank']:.4f}",
                f"D4-AUDIT rank after fine-tune: {audit_after['rank']:.4f}",
                f"D4-AUDIT pairwise after fine-tune: {audit_after['pairwise']:.4f}",
                f"D4-AUDIT sign after fine-tune: {audit_after['sign_accuracy']:.4f}",
                f"D4-AUDIT mean regret after fine-tune: {audit_after['mean_regret_m3']:.1f} m3",
                f"InternalHoldout D3 rank after fine-tune: {generic_after['rank']:.4f}",
                "",
                "D4-AUDIT and InternalHoldout are read-only diagnostics. No continuous MPC is authorized.",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
