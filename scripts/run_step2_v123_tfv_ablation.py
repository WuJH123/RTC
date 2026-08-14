"""Train/evaluate the frozen V7 TFV architecture with oracle vs causal rainfall."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
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
from rtc.step2_train_response_v60 import (
    InputNormalizationV60,
    V60TrainCache,
    deterministic_rainfall_split_v60,
    derive_input_normalization_v60,
)
from rtc.step2_train_response_v70 import derive_target_scales_v70, evaluate_value_v70
from rtc.step2_optimization_v70 import train_value_event_balanced_v70
from rtc.step2_v120_train_helpers import load_graph_v120
from rtc.step2_v70_contract import DirectValueLossContractV70


def _pre_rain_diagnostic(model, cache, names, normalization, prepared, device, store) -> dict[str, float | int]:
    indices = store.index()
    records = []
    model.eval()
    with torch.no_grad():
        for name in names:
            if float(store.forecast_mmhr[indices[name]].max()) > 1e-8:
                continue
            batch = cache.batch(name, normalization, device)
            pred = model(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
            ).delta_tfv_m3[0].detach().cpu().numpy().astype(np.float64)
            truth = batch.true_delta_tfv_m3[0].detach().cpu().numpy().astype(np.float64)
            records.append({
                "predicted_mean_abs": float(np.mean(np.abs(pred))),
                "truth_mean_abs": float(np.mean(np.abs(truth))),
                "predicted_spread": float(np.ptp(pred)),
                "truth_spread": float(np.ptp(truth)),
                "mae": float(np.mean(np.abs(pred - truth))),
            })
    if not records:
        return {"groups": 0, "mean_abs_prediction_m3": float("nan"), "mean_abs_truth_m3": float("nan"), "spread_m3": float("nan"), "mae_m3": float("nan")}
    return {
        "groups": len(records),
        "mean_abs_prediction_m3": float(np.mean([r["predicted_mean_abs"] for r in records])),
        "mean_abs_truth_m3": float(np.mean([r["truth_mean_abs"] for r in records])),
        "spread_m3": float(np.mean([r["predicted_spread"] for r in records])),
        "truth_spread_m3": float(np.mean([r["truth_spread"] for r in records])),
        "mae_m3": float(np.mean([r["mae"] for r in records])),
    }


def _train_arm(
    *, arm: str, cache, base: V60TrainCache, fit_d2: list[str], fit_d3: list[str], holdout_d2: list[str], holdout_d3: list[str], graph, basis, scales, normalization, prepared, device: str, seed: int, store,
) -> dict[str, Any]:
    first = base.entry(fit_d2[0]).arrays
    model = ControlValueSurrogateV70(
        state_dim=int(first["initial_state"].shape[-1]),
        rainfall_dim=int(first["rainfall"].shape[-1]),
        physics_dim=int(prepared.actuator_physics.shape[1]),
        actuator_count=len(graph.actuator_ids),
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        tfv_scale_m3=float(scales.direct_tfv_scale_m3),
        hidden_dim=96,
        actuator_embedding_dim=16,
    )
    history = train_value_event_balanced_v70(
        model, cache, fit_d2_names=fit_d2, fit_d3_names=fit_d3,
        normalization=normalization, scales=scales, graph=graph,
        device=device, seed=seed, contract=DirectValueLossContractV70(),
    )
    metrics = {
        "trainfit_d2": evaluate_value_v70(model, cache, fit_d2, normalization, prepared, device=device),
        "trainfit_d3": evaluate_value_v70(model, cache, fit_d3, normalization, prepared, device=device),
        "holdout_d2": evaluate_value_v70(model, cache, holdout_d2, normalization, prepared, device=device),
        "holdout_d3": evaluate_value_v70(model, cache, holdout_d3, normalization, prepared, device=device),
    }
    return {
        "arm": arm,
        "rainfall_input": "oracle_realised_cache_trajectory" if arm == "A_ORACLE" else "V123_parent_no_control_causal_forecast",
        "training_history": history,
        "metrics": metrics,
        "pre_rain_zero_forecast": _pre_rain_diagnostic(model, cache, fit_d2 + fit_d3, normalization, prepared, device, store),
        "state_dict": model.state_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="V123 TFV oracle-vs-causal rainfall ablation")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--causal-store", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if int(args.seed) != 42 or abs(float(args.holdout_fraction) - 0.20) > 1e-12:
        raise ValueError("V123 ablation is frozen at seed=42 and holdout_fraction=0.20")
    base = V60TrainCache(args.cache_manifest)
    store = load_causal_forecast_store_v123(args.causal_store)
    causal_cache = CausalForecastValueCacheV123(base, store)
    names = sorted(base.names("D2") + base.targeted_d3_names())
    fit, holdout = deterministic_rainfall_split_v60(base, names=names, holdout_fraction=0.20)
    fit_d2 = [n for n in fit if n.startswith("D2::")]; fit_d3 = [n for n in fit if n.startswith("D3::")]
    holdout_d2 = [n for n in holdout if n.startswith("D2::")]; holdout_d3 = [n for n in holdout if n.startswith("D3::")]
    graph = load_graph_v120(args.graph)
    basis = build_control_basis_v60(graph)
    prepared_cpu = prepare_static_v60(graph, "cpu")
    scales = derive_target_scales_v70(base, fit)
    norm_oracle = derive_input_normalization_v60(base, fit)
    # P0 hardening: rainfall normalization must be computed from the causal
    # forecast store, never from realised future rainfall in the base cache.
    norm_causal = derive_causal_input_normalization_v123(base, store, fit)
    device = args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    prepared = prepare_static_v60(graph, device)
    arm_a = _train_arm(
        arm="A_ORACLE", cache=base, base=base, fit_d2=fit_d2, fit_d3=fit_d3,
        holdout_d2=holdout_d2, holdout_d3=holdout_d3, graph=graph, basis=basis,
        scales=scales, normalization=norm_oracle, prepared=prepared, device=device,
        seed=42, store=store,
    )
    arm_b = _train_arm(
        arm="B_CAUSAL", cache=causal_cache, base=base, fit_d2=fit_d2, fit_d3=fit_d3,
        holdout_d2=holdout_d2, holdout_d3=holdout_d3, graph=graph, basis=basis,
        scales=scales, normalization=norm_causal, prepared=prepared, device=device,
        seed=42, store=store,
    )
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"arm": "A_ORACLE", "state_dict": arm_a.pop("state_dict")}, out / "v123_tfv_oracle.pt")
    torch.save({"arm": "B_CAUSAL", "state_dict": arm_b.pop("state_dict")}, out / "v123_tfv_causal.pt")
    payload = {
        "contract": "PROJECT7_V123_TFV_CAUSAL_ABLATION_V1",
        "seed": 42, "holdout_fraction": 0.20, "device": device,
        "source_tree_sha256": rtc_implementation_contract_sha256(),
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "causal_store": str(Path(args.causal_store).resolve()),
        "causal_store_sha256": hashlib.sha256(Path(args.causal_store).read_bytes()).hexdigest(),
        "split": {"fit_groups": len(fit), "holdout_groups": len(holdout), "fit_events": sorted({base.entry(n).event_id for n in fit}), "holdout_events": sorted({base.entry(n).event_id for n in holdout})},
        "target_scales": {"direct_tfv_scale_m3": float(scales.direct_tfv_scale_m3)},
        "arms": {"A_ORACLE": arm_a, "B_CAUSAL": arm_b},
        "future_realized_rainfall_used_as_model_input": {"A_ORACLE": True, "B_CAUSAL": False},
        "boundary": {"new_swmm": False, "validation_accessed": False, "final_accessed": False, "formal_accessed": False},
    }
    (out / "STEP2_V123_TFV_CAUSAL_ABLATION.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# STEP2 V123 TFV CAUSAL ABLATION", "", "| arm | subset | rank | pairwise | sign | top1 | MAE m3 | spread ratio | response ratio |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm_name, arm in (("A_ORACLE", arm_a), ("B_CAUSAL", arm_b)):
        for subset, m in arm["metrics"].items():
            lines.append(f"| {arm_name} | {subset} | {m['rank']:.4f} | {m['pairwise']:.4f} | {m['sign_accuracy']:.4f} | {m['top1_rate']:.4f} | {m['tfv_mae_m3']:.1f} | {m['spread_ratio']:.4f} | {m['response_ratio']:.4f} |")
    (out / "STEP2_V123_TFV_CAUSAL_ABLATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
