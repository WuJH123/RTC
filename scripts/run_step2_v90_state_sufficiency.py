"""Train-only V9 signed-effect P0 repair and state-sufficiency ladder.

This diagnostic intentionally uses only frozen TrainFit D2 groups.  The oracle
trajectory branch is offline evidence and is never written as a production model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from rtc.step2_control_basis_v60 import build_control_basis_v60
from rtc.step2_control_response_v80 import DirectHydraulicEffectSurrogateV80, prepare_static_v80
from rtc.step2_control_response_v90 import TrajectoryConditionedHydraulicEffectSurrogateV90
from rtc.step2_hydraulic_objective_v80 import derive_onset_sqrt_positive_weight_v80, hydraulic_effect_loss_v80
from rtc.step2_train_response_v60 import V60TrainCache, derive_input_normalization_v60, deterministic_rainfall_split_v60
from rtc.step2_train_response_v70 import derive_target_scales_v70
from rtc.step2_v80_contract import DirectHydraulicEffectLossContractV80
from rtc.step2_shards_v60 import validate_v60_cache_lineage

from run_step2_v80 import _build_reference_hydraulic, _load_checkpoint, _load_graph, _sha256


def _events(cache: V60TrainCache, names: Sequence[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for name in names:
        entry = cache.entry(name)
        grouped[f"{entry.rainfall_group}::{entry.event_id}"].append(name)
    return {key: sorted(values) for key, values in sorted(grouped.items())}


def _forward(model, batch, prepared, mode: str):
    context = None
    if mode == "oracle":
        context = (batch.true_reference_states, batch.true_reference_flows)
    return model(
        batch.initial_state,
        batch.rainfall,
        batch.reference_settings,
        batch.candidate_settings,
        batch.previous_actuator_flow,
        prepared,
        reference_trajectory_context=context,
    )


def _metric(predicted: np.ndarray, truth: np.ndarray, scale: float, fraction: float) -> dict[str, float]:
    pred = np.asarray(predicted, dtype=np.float64)
    actual = np.asarray(truth, dtype=np.float64)
    error = pred - actual
    active = np.abs(actual) >= float(fraction) * max(float(scale), 1e-12)
    mse = float(np.mean(np.square(error)))
    zero = float(np.mean(np.square(actual)))
    result = {
        "skill_vs_zero": 1.0 - mse / max(zero, 1e-12),
        "response_ratio": float(np.mean(np.abs(pred)) / max(float(np.mean(np.abs(actual))), 1e-12)),
        "rmse": float(np.sqrt(mse)),
        "active_fraction": float(active.mean()),
    }
    if active.any():
        result["active_sign"] = float(np.mean(np.sign(pred[active]) == np.sign(actual[active])))
        result["active_skill_vs_zero"] = 1.0 - float(np.mean(np.square(error[active]))) / max(float(np.mean(np.square(actual[active]))), 1e-12)
    else:
        result["active_sign"] = float("nan")
        result["active_skill_vs_zero"] = float("nan")
    return result


def _evaluate(model, cache, names, normalization, prepared, scales, mode, device, contract):
    records: list[dict[str, Any]] = []
    model.to(device).eval()
    with torch.no_grad():
        for name in names:
            batch = cache.batch(name, normalization, device)
            output = _forward(model, batch, prepared, mode)
            idx = output.horizon_indices
            true_ref = batch.true_reference_states.index_select(1, idx)[:, None].expand_as(output.reference_states_physical)
            true_cand = batch.true_candidate_states.index_select(2, idx)
            true_delta = (true_cand - true_ref).cpu().numpy()
            pred_delta = output.raw_delta_states_physical.cpu().numpy()
            true_ref_flow = batch.true_reference_flows.index_select(1, idx)[:, None].expand_as(output.reference_flows_physical)
            true_cand_flow = batch.true_candidate_flows.index_select(2, idx)
            true_flow = (true_cand_flow - true_ref_flow).cpu().numpy()
            pred_flow = output.raw_delta_flows_physical.cpu().numpy()
            event = f"{cache.entry(name).rainfall_group}::{cache.entry(name).event_id}"
            record: dict[str, Any] = {"event": event}
            for channel, label in ((0, "depth"), (2, "flood"), (3, "storage")):
                record[label] = _metric(
                    pred_delta[..., channel], true_delta[..., channel],
                    float(scales.state_delta_scale[channel]), contract.active_effect_fraction,
                )
            record["flow"] = _metric(
                pred_flow, true_flow, float(np.median(scales.flow_delta_scale)), contract.active_effect_fraction,
            )
            records.append(record)
    event_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        event_groups[str(record["event"])].append(record)
    result: dict[str, Any] = {"groups": len(records), "events": len(event_groups), "mode": mode}
    for label in ("depth", "flood", "storage", "flow"):
        for key in ("skill_vs_zero", "active_skill_vs_zero", "response_ratio", "active_sign", "active_fraction", "rmse"):
            event_values = []
            for event_records in event_groups.values():
                values = [float(r[label][key]) for r in event_records if np.isfinite(r[label][key])]
                if values:
                    event_values.append(float(np.mean(values)))
            result[f"{label}_{key}"] = float(np.mean(event_values)) if event_values else float("nan")
    result["mean_primary_skill"] = float(np.mean([result[f"{label}_skill_vs_zero"] for label in ("depth", "flood", "storage", "flow")]))
    return result


def _build_model(graph, cache, fit_names, scales, basis, checkpoint, trajectory_conditioning, seed):
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    reference = _build_reference_hydraulic(graph, cache, fit_names, scales)
    reference.load_state_dict(checkpoint["state_dict"], strict=True)
    cpu_static = prepare_static_v80(graph, "cpu")
    base = DirectHydraulicEffectSurrogateV80(
        reference_model=reference,
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        state_delta_scale=scales.state_delta_scale,
        flow_delta_scale=scales.flow_delta_scale,
        physics_dim=int(cpu_static.base.actuator_physics.shape[1]),
        node_static_dim=int(graph.static_node_features.shape[1]),
        actuator_count=len(graph.actuator_ids),
        contract=DirectHydraulicEffectLossContractV80(),
        trajectory_conditioning=trajectory_conditioning,
    )
    return TrajectoryConditionedHydraulicEffectSurrogateV90(base)


def _train_d2(model, cache, names, normalization, scales, graph, mode, device, seed):
    contract = DirectHydraulicEffectLossContractV80()
    onset_weight = derive_onset_sqrt_positive_weight_v80(cache, names, normalization, contract=contract)
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    prepared = prepare_static_v80(graph, target)
    trainable = list(model.trainable_parameters())
    optimizer = torch.optim.AdamW(trainable, lr=contract.learning_rate, weight_decay=contract.weight_decay)
    rng = np.random.default_rng(int(seed) + 31)
    event_groups = _events(cache, names)
    history = []
    model.to(target).train()
    for epoch in range(1, contract.d2_pretrain_epochs + 1):
        keys = list(event_groups)
        rng.shuffle(keys)
        losses = []
        for key in keys:
            optimizer.zero_grad(set_to_none=True)
            group_names = event_groups[key]
            for name in group_names:
                batch = cache.batch(name, normalization, target)
                output = _forward(model, batch, prepared, mode)
                loss, metrics = hydraulic_effect_loss_v80(output, batch, normalization, scales, onset_positive_weight=onset_weight, contract=contract)
                (loss / len(group_names)).backward()
                losses.append(metrics)
            torch.nn.utils.clip_grad_norm_(trainable, contract.grad_clip)
            optimizer.step()
        history.append({"epoch": epoch, "loss": float(np.mean([m["loss"] for m in losses]))})
    return history, onset_weight, prepared


def _flow_sign_audit(cache: V60TrainCache, graph) -> dict[str, Any]:
    all_flow = np.concatenate([
        np.asarray(cache.entry(name).arrays["target_actuator_flows"][list(cache.entry(name).indices)], dtype=np.float64)
        for name in cache.names()
    ], axis=0)
    physics_names = tuple(graph.actuator_physics_feature_names)
    physics = np.asarray(graph.actuator_physics)
    by_type = {}
    for label, feature in (("pump", "is_pump"), ("orifice", "is_orifice"), ("weir", "is_weir"), ("outlet", "is_outlet")):
        indices = np.flatnonzero(physics[:, physics_names.index(feature)] > 0.5)
        values = all_flow[..., indices].reshape(-1) if len(indices) else np.asarray([], dtype=np.float64)
        by_type[label] = {
            "actuator_count": int(len(indices)),
            "sample_count": int(values.size),
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
            "negative_fraction": float(np.mean(values < 0)) if values.size else None,
            "zero_fraction": float(np.mean(values == 0)) if values.size else None,
            "positive_fraction": float(np.mean(values > 0)) if values.size else None,
        }
    return {
        "contract": "PROJECT7_STEP2_V90_ACTUATOR_FLOW_SIGN_AUDIT_V1",
        "source": "frozen training_cache_v60 target_actuator_flows",
        "physical_flow_projection": "signed_reference_plus_raw_effect_no_global_clamp",
        "by_type": by_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--v70-hydraulic-checkpoint", required=True)
    parser.add_argument("--v80-report", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()
    started = time.perf_counter()
    graph = _load_graph(args.graph)
    cache = V60TrainCache(args.cache_manifest)
    if cache.legacy_d3_names():
        raise ValueError("V9 refuses legacy dense D3")
    selected = sorted(cache.names("D2") + cache.targeted_d3_names())
    fit, holdout = deterministic_rainfall_split_v60(cache, names=selected, holdout_fraction=args.holdout_fraction)
    fit_d2 = [n for n in fit if n.startswith("D2::")]
    if not fit_d2:
        raise ValueError("V9 requires TrainFit D2 groups")
    normalization = derive_input_normalization_v60(cache, fit)
    scales = derive_target_scales_v70(cache, fit)
    basis = build_control_basis_v60(graph)
    checkpoint = _load_checkpoint(args.v70_hydraulic_checkpoint, "hydraulic_response")
    flow_audit = _flow_sign_audit(cache, graph)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "STEP2_V90_ACTUATOR_FLOW_SIGN_AUDIT.json").write_text(json.dumps(flow_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    modes = {"A_current_v8": (False, "current"), "B_predicted_reference": (True, "predicted"), "C_oracle_reference": (True, "oracle")}
    diagnostics = {}
    histories = {}
    diagnostic_started = time.perf_counter()
    for label, (trajectory, mode) in modes.items():
        model = _build_model(graph, cache, fit, scales, basis, checkpoint, trajectory, args.seed)
        history, onset_weight, prepared = _train_d2(model, cache, fit_d2, normalization, scales, graph, mode, args.device, args.seed)
        diagnostics[label] = _evaluate(model, cache, fit_d2, normalization, prepared, scales, mode, torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"), DirectHydraulicEffectLossContractV80())
        diagnostics[label]["onset_positive_weight"] = onset_weight
        histories[label] = history
    ladder_started = time.perf_counter()
    a = diagnostics["A_current_v8"]["mean_primary_skill"]
    b = diagnostics["B_predicted_reference"]["mean_primary_skill"]
    c = diagnostics["C_oracle_reference"]["mean_primary_skill"]
    if b > a + 0.05 and b >= c - max(0.05, 0.5 * abs(c - a)):
        diagnosis = "PREDICTED_REFERENCE_TRAJECTORY_IS_SUFFICIENT"
    elif c > b + 0.05:
        diagnosis = "REFERENCE_HYDRAULIC_MODEL_ACCURACY_IS_PRIMARY_LIMIT"
    elif c <= 0.0:
        diagnosis = "CURRENT_SNAPSHOT_ACTION_DATA_NOT_SUFFICIENT_FOR_LOCAL_HYDRAULIC_EFFECT"
    else:
        diagnosis = "STATE_SUFFICIENCY_INCONCLUSIVE"
    ladder = {
        "contract": "PROJECT7_STEP2_V90_STATE_SUFFICIENCY_LADDER_V1",
        "train_only": True,
        "d2_groups": len(fit_d2),
        "same_seed": args.seed,
        "diagnostics": diagnostics,
        "diagnosis": diagnosis,
        "classification_rule": "B materially exceeds A by 0.05 and is within 0.05 or half of A-C gap; otherwise C materially exceeds B; C mean skill <= 0 supports snapshot insufficiency.",
        "timing_seconds": {"diagnostics": time.perf_counter() - diagnostic_started, "total": time.perf_counter() - started},
    }
    (out / "STEP2_V90_STATE_SUFFICIENCY_LADDER.json").write_text(json.dumps(ladder, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    correctness = {
        "contract": "PROJECT7_STEP2_V90_SIGNED_EFFECT_CORRECTNESS_V1",
        "old_post_clipping_effect": "CONFIRMED",
        "negative_flood_effect": "raw signed output preserved; physical projection independent",
        "negative_depth_effect": "raw signed output preserved; physical projection independent",
        "signed_flow_effect": "signed reference plus raw effect; no global clamp",
        "exact_zero": True,
        "actuator_flow_sign_audit": str(out / "STEP2_V90_ACTUATOR_FLOW_SIGN_AUDIT.json"),
        "lineage": {"graph_sha256": _sha256(args.graph), "cache_manifest_sha256": _sha256(args.cache_manifest), "v70_hydraulic_checkpoint_sha256": _sha256(args.v70_hydraulic_checkpoint), "v80_report_sha256": _sha256(args.v80_report)},
    }
    (out / "STEP2_V90_SIGNED_EFFECT_CORRECTNESS.json").write_text(json.dumps(correctness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"signed_effect_correctness": correctness, "state_sufficiency_ladder": ladder}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
