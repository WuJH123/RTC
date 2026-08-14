"""Train/evaluate V124 interaction-aware TFV Value on the frozen causal D2/D3 split."""
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
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_train_response_v70 import derive_target_scales_v70, evaluate_value_v70
from rtc.step2_v120_train_helpers import load_graph_v120


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--causal-store", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--listwise-weight", type=float, default=0.30)
    args = parser.parse_args()
    if int(args.seed) != 42 or abs(float(args.holdout_fraction) - 0.20) > 1.0e-12:
        raise ValueError("V124 comparison is frozen at seed=42 and holdout_fraction=0.20")

    base = V60TrainCache(args.cache_manifest)
    store = load_causal_forecast_store_v123(args.causal_store)
    cache = CausalForecastValueCacheV123(base, store)
    names = sorted(base.names("D2") + base.targeted_d3_names())
    fit, holdout = deterministic_rainfall_split_v60(
        base, names=names, holdout_fraction=float(args.holdout_fraction)
    )
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    fit_d3 = [name for name in fit if name.startswith("D3::")]
    holdout_d2 = [name for name in holdout if name.startswith("D2::")]
    holdout_d3 = [name for name in holdout if name.startswith("D3::")]
    if (len(fit_d2), len(fit_d3), len(holdout_d2), len(holdout_d3)) != (112, 112, 32, 32):
        raise ValueError("V124 split differs from frozen 112/112/32/32 contract")

    graph = load_graph_v120(args.graph)
    basis = build_control_basis_v60(graph)
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
        hidden_dim=int(args.hidden_dim),
        actuator_embedding_dim=16,
        attention_heads=int(args.attention_heads),
    )
    loss_contract = ValueLossContractV124(listwise_weight=float(args.listwise_weight))
    history = train_value_event_balanced_v124(
        model,
        cache,
        fit_d2_names=fit_d2,
        fit_d3_names=fit_d3,
        normalization=normalization,
        scales=scales,
        graph=graph,
        device=device,
        seed=42,
        contract=loss_contract,
    )
    metrics = {
        "trainfit_d2": evaluate_value_v70(
            model, cache, fit_d2, normalization, prepared, device=device
        ),
        "trainfit_d3": evaluate_value_v70(
            model, cache, fit_d3, normalization, prepared, device=device
        ),
        "holdout_d2": evaluate_value_v70(
            model, cache, holdout_d2, normalization, prepared, device=device
        ),
        "holdout_d3": evaluate_value_v70(
            model, cache, holdout_d3, normalization, prepared, device=device
        ),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "v124_tfv_causal.pt"
    torch.save(
        {
            "contract": V124_VALUE_CONTRACT,
            "architecture": "interaction_aware_active_actuator_self_attention",
            "state_dict": model.state_dict(),
            "hidden_dim": int(args.hidden_dim),
            "attention_heads": int(args.attention_heads),
            "listwise_weight": float(args.listwise_weight),
        },
        checkpoint,
    )
    payload = {
        "contract": V124_VALUE_CONTRACT,
        "source_tree_sha256": rtc_implementation_contract_sha256(),
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "causal_store": str(Path(args.causal_store).resolve()),
        "causal_store_sha256": hashlib.sha256(Path(args.causal_store).read_bytes()).hexdigest(),
        "seed": 42,
        "split": {
            "fit_d2": len(fit_d2),
            "fit_d3": len(fit_d3),
            "holdout_d2": len(holdout_d2),
            "holdout_d3": len(holdout_d3),
        },
        "architecture": {
            "hidden_dim": int(args.hidden_dim),
            "attention_heads": int(args.attention_heads),
            "listwise_weight": float(args.listwise_weight),
        },
        "target_scale_tfv_m3": float(scales.direct_tfv_scale_m3),
        "training_history": history,
        "metrics": metrics,
        "checkpoint": str(checkpoint.resolve()),
        "boundary": {
            "new_swmm": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
            "continuous_mpc_authorized": False,
        },
    }
    report = out / "STEP2_V124_INTERACTION_VALUE.json"
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    hold = metrics["holdout_d3"]
    report.with_suffix(".md").write_text(
        "\n".join(
            [
                "# STEP2 V124 interaction-aware causal Value",
                "",
                f"Holdout D3 rank: {hold['rank']:.4f}",
                f"Holdout D3 pairwise: {hold['pairwise']:.4f}",
                f"Holdout D3 sign: {hold['sign_accuracy']:.4f}",
                f"Holdout D3 top1: {hold['top1_rate']:.4f}",
                f"Holdout D3 MAE: {hold['tfv_mae_m3']:.2f} m3",
                "",
                "Continuous MPC remains blocked until the frozen gradient/value gate passes.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
