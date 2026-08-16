"""Audit managed-flow action effects for a current repaired smoke/dev stage checkpoint.

This is a cheap Development gate intended immediately after Stage A.  For the deterministic
held-out D2 groups it teacher-forces each reference/candidate branch with its own authoritative
previous hydraulic state/managed flow, predicts the next managed flow with the checkpoint
actuator submodel, and compares candidate-minus-reference flow effects with authoritative SWMM.

The audit is read-only.  It never changes the FIT-only hybrid flow scale, never trains, and
never accesses Validation/Final/Formal/Policy Lock.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from rtc.development_profile_v128 import apply_profile_to_design, get_execution_profile, profile_groups
from rtc.production_cli import _load_graph
from rtc.stage_checkpoint_v128 import load_stage_checkpoint_v128
from rtc.step2_causal_rainfall_v123 import load_causal_forecast_store_v123
from rtc.step2_current_dev_context_v128 import (
    build_current_action_identifiable_model,
    extend_action_identifiable_stage_lineage,
)
from rtc.step2_d4_cache_v125 import D4_SOURCE_KIND
from rtc.step2_state_store_v127 import load_causal_state_store_v127
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_train_v127 import _static
from rtc.step2_train_v127_streaming import V127StreamingMemoryDesign
from rtc.v128_control_profile import build_v128_control_training_design, configure_v128_cuda_matmul_precision

CONTRACT = "PROJECT7_CURRENT_ACTION_IDENTIFIABLE_ACTUATOR_FLOW_EFFECT_AUDIT_V2"


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rain(cache: V60TrainCache, names: list[str]) -> set[str]:
    return {str(cache.entry(name).rainfall_group) for name in names}


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom > 0 else float("nan")


def _metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    mask = np.isfinite(truth) & np.isfinite(pred) & (np.abs(truth) > 1.0e-8)
    t = truth[mask].astype(np.float64, copy=False)
    p = pred[mask].astype(np.float64, copy=False)
    if t.size == 0:
        raise RuntimeError("action-to-flow audit contains no informative SWMM flow effects")
    ratios = np.abs(p) / np.maximum(np.abs(t), 1.0e-8)
    l2_truth = float(np.linalg.norm(t))
    return {
        "informative_cases": int(t.size),
        "flow_effect_sign_accuracy": float(np.mean(np.sign(t) == np.sign(p))),
        "flow_effect_cosine": _cosine(t, p),
        "flow_effect_mae_m3s": float(np.mean(np.abs(p - t))),
        "predicted_to_true_flow_effect_l2_ratio": (
            float(np.linalg.norm(p) / l2_truth) if l2_truth > 0 else float("nan")
        ),
        "median_abs_predicted_to_true_flow_effect_ratio": float(np.median(ratios)),
        "q10_abs_predicted_to_true_flow_effect_ratio": float(np.quantile(ratios, 0.10)),
        "q90_abs_predicted_to_true_flow_effect_ratio": float(np.quantile(ratios, 0.90)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("smoke", "dev"), required=True)
    p.add_argument("--stage", choices=("stage_a", "stage_b0", "objective"), required=True)
    p.add_argument("--stage-checkpoint", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--edge-physics", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--d4-fit-cache", required=True)
    p.add_argument("--d4-audit-cache", required=True)
    p.add_argument("--causal-store", required=True)
    p.add_argument("--causal-state-store", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    configure_v128_cuda_matmul_precision()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    profile = get_execution_profile(args.profile)
    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != 109:
        raise ValueError("current action-to-flow audit requires exactly 109 actuators")

    base = V60TrainCache(args.cache_manifest)
    d4_fit = V60TrainCache(args.d4_fit_cache)
    d4_audit = V60TrainCache(args.d4_audit_cache)
    rain_store = load_causal_forecast_store_v123(args.causal_store)
    state_store = load_causal_state_store_v127(args.causal_state_store)
    rain_store.validate()
    state_store.validate()
    fit, holdout = deterministic_rainfall_split_v60(
        base,
        names=sorted(base.names("D2") + base.targeted_d3_names()),
        holdout_fraction=0.20,
    )
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    fit_d3 = [name for name in fit if name.startswith("D3::")]
    hold_d2 = [name for name in holdout if name.startswith("D2::")]
    hold_d3 = [name for name in holdout if name.startswith("D3::")]
    if tuple(map(len, (fit_d2, fit_d3, hold_d2, hold_d3))) != (112, 112, 32, 32):
        raise ValueError("canonical D2/D3 split differs from 112/112/32/32")
    d4_fit_names = d4_fit.names(D4_SOURCE_KIND)
    d4_audit_names = d4_audit.names(D4_SOURCE_KIND)
    if (len(d4_fit_names), len(d4_audit_names)) != (33, 15):
        raise ValueError("canonical D4 FIT/AUDIT census differs from 33/15")
    if _rain(d4_fit, d4_fit_names) & _rain(d4_audit, d4_audit_names):
        raise ValueError("D4 FIT/AUDIT rainfall leakage")
    selected = profile_groups(
        profile,
        fit_d2=fit_d2,
        fit_d3=fit_d3,
        hold_d2=hold_d2,
        hold_d3=hold_d3,
        d4_fit=d4_fit_names,
        d4_audit=d4_audit_names,
    )

    first_state = state_store.state_for(base.entry(selected["fit_d2"][0]))
    model = build_current_action_identifiable_model(
        graph,
        edge_physics_path=args.edge_physics,
        state_dim=int(first_state.shape[-1]),
        rainfall_dim=int(rain_store.forecast_mmhr.shape[-1]),
        delta_state_scale=np.ones(int(first_state.shape[-1]), dtype=np.float32),
        delta_flow_scale=np.ones(len(graph.actuator_ids), dtype=np.float32),
    )
    memory = V127StreamingMemoryDesign(
        hydraulic_branch_chunk=4,
        rollout_candidates_per_group=2,
        objective_candidate_chunk=2,
        evaluation_branch_chunk=4,
        residual_sample_rows=131072,
    )
    design = apply_profile_to_design(
        build_v128_control_training_design(
            hydraulic_branch_chunk=memory.hydraulic_branch_chunk,
            rollout_candidates_per_group=memory.rollout_candidates_per_group,
            objective_candidate_chunk=memory.objective_candidate_chunk,
            evaluation_branch_chunk=memory.evaluation_branch_chunk,
        ),
        profile,
    )
    manifest = json.loads(Path(args.cache_manifest).read_text(encoding="utf-8"))
    lineage = {
        "graph_sha256": _sha(args.graph),
        "base_cache_sha256": _sha(args.cache_manifest),
        "d4_fit_cache_sha256": _sha(args.d4_fit_cache),
        "d4_audit_cache_sha256": _sha(args.d4_audit_cache),
        "causal_rainfall_sha256": _sha(args.causal_store),
        "causal_state_store_sha256": _sha(args.causal_state_store),
        "causal_state_step1_file_sha256": str(state_store.step1_sha256),
        "causal_state_sensor_file_sha256": str(state_store.sensor_sha256),
        "causal_state_graph_file_sha256": str(state_store.graph_sha256),
        "causal_state_step1_model_semantic_sha256": str(state_store.step1_model_semantic_sha256),
        "causal_state_sensor_layout_semantic_sha256": str(state_store.sensor_layout_semantic_sha256),
        "causal_rainfall_forecast_contract": str(rain_store.forecast_contract),
        "swmm_engine_version": str(manifest["swmm_engine_version"]),
    }
    lineage = extend_action_identifiable_stage_lineage(
        lineage, edge_physics_path=args.edge_physics
    )
    payload = load_stage_checkpoint_v128(
        args.stage_checkpoint,
        model=model,
        expected_profile=profile.name,
        graph_path=args.graph,
        expected_lineage=lineage,
        expected_training_design=asdict(design),
    )
    if str(payload.get("completed_stage")) != args.stage:
        raise ValueError(
            f"requested {args.stage} but checkpoint contains {payload.get('completed_stage')}"
        )

    model = model.to(device).eval()
    static = _static(graph, device)
    truth_values: list[float] = []
    pred_values: list[float] = []
    rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for name in selected["hold_d2"]:
            entry = base.entry(name)
            arrays = entry.arrays
            ref = int(entry.reference_index)
            ref_setting = np.asarray(arrays["settings"][ref], dtype=np.float32)
            ref_states = np.asarray(arrays["target_states"][ref], dtype=np.float32)
            ref_flows = np.asarray(arrays["target_actuator_flows"][ref], dtype=np.float32)
            ref_initial = np.asarray(arrays["initial_state"][ref], dtype=np.float32)
            ref_prev_flow0 = np.asarray(arrays["previous_actuator_flow"][ref], dtype=np.float32)
            for candidate_index in entry.indices:
                if int(candidate_index) == ref:
                    continue
                cand_setting = np.asarray(arrays["settings"][candidate_index], dtype=np.float32)
                changed = np.flatnonzero(np.any(np.abs(cand_setting - ref_setting) > 1.0e-6, axis=0))
                if changed.size != 1:
                    raise ValueError(
                        f"{name}: held-out D2 candidate must change exactly one actuator, got {changed.size}"
                    )
                actuator_index = int(changed[0])
                cand_states = np.asarray(arrays["target_states"][candidate_index], dtype=np.float32)
                cand_flows = np.asarray(arrays["target_actuator_flows"][candidate_index], dtype=np.float32)
                cand_initial = np.asarray(arrays["initial_state"][candidate_index], dtype=np.float32)
                cand_prev_flow0 = np.asarray(
                    arrays["previous_actuator_flow"][candidate_index], dtype=np.float32
                )
                horizon = int(ref_setting.shape[0])
                physics_norm, identity = model.actuator.prepare_static(
                    static["physics"], batch_size=2
                )
                for k in range(horizon):
                    ref_prev_state = ref_initial if k == 0 else ref_states[k - 1]
                    cand_prev_state = cand_initial if k == 0 else cand_states[k - 1]
                    ref_prev_flow = ref_prev_flow0 if k == 0 else ref_flows[k - 1]
                    cand_prev_flow = cand_prev_flow0 if k == 0 else cand_flows[k - 1]
                    branch_state = torch.as_tensor(
                        np.stack((ref_prev_state, cand_prev_state)),
                        dtype=torch.float32,
                        device=device,
                    )
                    branch_flow = torch.as_tensor(
                        np.stack((ref_prev_flow, cand_prev_flow)),
                        dtype=torch.float32,
                        device=device,
                    )
                    branch_setting = torch.as_tensor(
                        np.stack((ref_setting[k], cand_setting[k])),
                        dtype=torch.float32,
                        device=device,
                    )
                    q, _ = model.actuator.forward_prepared(
                        branch_state[:, static["up"]],
                        branch_state[:, static["down"]],
                        branch_setting,
                        branch_flow,
                        physics_norm,
                        identity,
                    )
                    truth = float(cand_flows[k, actuator_index] - ref_flows[k, actuator_index])
                    pred = float(q[1, actuator_index] - q[0, actuator_index])
                    truth_values.append(truth)
                    pred_values.append(pred)
                    rows.append(
                        {
                            "group": name,
                            "candidate_index": int(candidate_index),
                            "step": k,
                            "actuator_index": actuator_index,
                            "actuator_id": str(graph.actuator_ids[actuator_index]),
                            "setting_delta": float(
                                cand_setting[k, actuator_index] - ref_setting[k, actuator_index]
                            ),
                            "true_flow_effect_m3s": truth,
                            "predicted_flow_effect_m3s": pred,
                            "delta_flow_scale_m3s": float(
                                model.actuator.delta_flow_scale[actuator_index].detach().cpu()
                            ),
                        }
                    )

    truth_array = np.asarray(truth_values, dtype=np.float64)
    pred_array = np.asarray(pred_values, dtype=np.float64)
    metrics = _metrics(truth_array, pred_array)
    result = {
        "contract": CONTRACT,
        "profile": profile.name,
        "completed_stage": args.stage,
        "scientific_split": "development",
        "scientific_claim_allowed": False,
        "stage_checkpoint_sha256": _sha(args.stage_checkpoint),
        "edge_physics_sha256": _sha(args.edge_physics),
        "selected_holdout_d2_groups": selected["hold_d2"],
        "single_actuator_action_contract_verified": True,
        "teacher_forced_authoritative_previous_state_and_flow": True,
        "used_for_training": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
        "flow_effect_metrics": metrics,
    }
    for value in metrics.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError("action-to-flow audit produced a non-finite metric")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ACTION_TO_FLOW_DETAIL.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "ACTION_TO_FLOW_METRICS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
