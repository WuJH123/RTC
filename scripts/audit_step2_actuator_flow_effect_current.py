"""Audit direct and feedback managed-flow action effects for current Project7 Step2.

The primary Development gate is the *direct same-prefix* response: reference/candidate settings
are evaluated from one common authoritative pre-action hydraulic state and managed flow at the
first setting-divergence transition. This isolates setting -> managed flow.

A secondary diagnostic retains the historical all-H72 teacher-forced comparison where each branch
uses its own authoritative previous state/flow. That quantity includes network feedback and must
not be interpreted as local actuator `dq/du`. Micro and actuator-balanced summaries are both
reported so one high-flow asset cannot silently determine promotion/rejection.
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
from rtc.step2_counterfactual_first_v128 import first_direct_response_spec_numpy
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

CONTRACT = "PROJECT7_CURRENT_COUNTERFACTUAL_FIRST_ACTUATOR_FLOW_EFFECT_AUDIT_V4"


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


def _macro_by_actuator(
    rows: list[dict[str, object]],
) -> tuple[dict[str, float | int], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["actuator_id"]), []).append(row)
    per: list[dict[str, object]] = []
    truth_energy: dict[str, float] = {}
    for actuator, values in sorted(grouped.items()):
        truth = np.asarray([float(row["true_flow_effect_m3s"]) for row in values], dtype=np.float64)
        pred = np.asarray([float(row["predicted_flow_effect_m3s"]) for row in values], dtype=np.float64)
        mask = np.abs(truth) > 1.0e-8
        if not mask.any():
            continue
        metrics = _metrics(truth, pred)
        energy = float(np.dot(truth[mask], truth[mask]))
        truth_energy[actuator] = energy
        per.append({"actuator_id": actuator, "truth_l2": float(np.sqrt(energy)), **metrics})
    if not per:
        raise RuntimeError("action-to-flow audit produced no actuator-level metrics")
    cosines = np.asarray([float(row["flow_effect_cosine"]) for row in per], dtype=np.float64)
    signs = np.asarray([float(row["flow_effect_sign_accuracy"]) for row in per], dtype=np.float64)
    ratios = np.asarray(
        [float(row["predicted_to_true_flow_effect_l2_ratio"]) for row in per], dtype=np.float64
    )
    total_energy = float(sum(truth_energy.values()))
    ranked = sorted(per, key=lambda row: float(row["truth_l2"]), reverse=True)
    for row in ranked:
        energy = truth_energy[str(row["actuator_id"])]
        row["truth_l2_energy_fraction"] = energy / total_energy if total_energy > 0 else 0.0
    macro = {
        "informative_actuators": len(per),
        "mean_actuator_cosine": float(np.nanmean(cosines)),
        "median_actuator_cosine": float(np.nanmedian(cosines)),
        "mean_actuator_sign_accuracy": float(np.nanmean(signs)),
        "median_actuator_sign_accuracy": float(np.nanmedian(signs)),
        "mean_actuator_l2_ratio": float(np.nanmean(ratios)),
        "median_actuator_l2_ratio": float(np.nanmedian(ratios)),
        "largest_actuator_truth_l2_energy_fraction": float(ranked[0]["truth_l2_energy_fraction"]),
    }
    return macro, ranked[:10]


def _predict_flow(
    model,
    *,
    state: np.ndarray,
    previous_flow: np.ndarray,
    settings: np.ndarray,
    static: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    state_t = torch.as_tensor(state, dtype=torch.float32, device=device)
    flow_t = torch.as_tensor(previous_flow, dtype=torch.float32, device=device)
    settings_t = torch.as_tensor(settings, dtype=torch.float32, device=device)
    physics_norm, identity = model.actuator.prepare_static(
        static["physics"], batch_size=int(state_t.shape[0])
    )
    q, _ = model.actuator.forward_prepared(
        state_t[:, static["up"]],
        state_t[:, static["down"]],
        settings_t,
        flow_t,
        physics_norm,
        identity,
    )
    return q


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
        direct_action_flow_scale=np.ones(len(graph.actuator_ids), dtype=np.float32),
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
    direct_rows: list[dict[str, object]] = []
    feedback_rows: list[dict[str, object]] = []
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
                candidate = int(candidate_index)
                if candidate == ref:
                    continue
                cand_setting = np.asarray(arrays["settings"][candidate], dtype=np.float32)
                changed = np.flatnonzero(
                    np.any(np.abs(cand_setting - ref_setting) > 1.0e-6, axis=0)
                )
                if changed.size != 1:
                    raise ValueError(
                        f"{name}: held-out D2 candidate must change exactly one actuator"
                    )
                actuator = int(changed[0])
                cand_states = np.asarray(arrays["target_states"][candidate], dtype=np.float32)
                cand_flows = np.asarray(arrays["target_actuator_flows"][candidate], dtype=np.float32)
                cand_initial = np.asarray(arrays["initial_state"][candidate], dtype=np.float32)
                cand_prev_flow0 = np.asarray(
                    arrays["previous_actuator_flow"][candidate], dtype=np.float32
                )

                spec = first_direct_response_spec_numpy(
                    arrays, reference=ref, candidate=candidate, require_single_actuator=True
                )
                if spec is not None:
                    k = int(spec["step"])
                    if k == 0:
                        prefix_state = ref_initial
                        prefix_flow = ref_prev_flow0
                    else:
                        prefix_state = ref_states[k - 1]
                        prefix_flow = ref_flows[k - 1]
                    q = _predict_flow(
                        model,
                        state=np.stack((prefix_state, prefix_state)),
                        previous_flow=np.stack((prefix_flow, prefix_flow)),
                        settings=np.stack((ref_setting[k], cand_setting[k])),
                        static=static,
                        device=device,
                    )
                    truth = float(cand_flows[k, actuator] - ref_flows[k, actuator])
                    pred = float(q[1, actuator] - q[0, actuator])
                    direct_rows.append(
                        {
                            "group": name,
                            "candidate_index": candidate,
                            "step": k,
                            "actuator_index": actuator,
                            "actuator_id": str(graph.actuator_ids[actuator]),
                            "setting_delta": float(
                                cand_setting[k, actuator] - ref_setting[k, actuator]
                            ),
                            "true_flow_effect_m3s": truth,
                            "predicted_flow_effect_m3s": pred,
                            "temporal_flow_scale_m3s": float(
                                model.actuator.delta_flow_scale[actuator].cpu()
                            ),
                            "direct_action_flow_scale_m3s": float(
                                model.actuator.direct_action_flow_scale[actuator].cpu()
                            ),
                            "prefix_state_max_abs": float(spec["prefix_state_max_abs"]),
                            "prefix_flow_max_abs": float(spec["prefix_flow_max_abs"]),
                        }
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
                    state = torch.as_tensor(
                        np.stack((ref_prev_state, cand_prev_state)),
                        dtype=torch.float32,
                        device=device,
                    )
                    prev_flow = torch.as_tensor(
                        np.stack((ref_prev_flow, cand_prev_flow)),
                        dtype=torch.float32,
                        device=device,
                    )
                    setting = torch.as_tensor(
                        np.stack((ref_setting[k], cand_setting[k])),
                        dtype=torch.float32,
                        device=device,
                    )
                    q, _ = model.actuator.forward_prepared(
                        state[:, static["up"]],
                        state[:, static["down"]],
                        setting,
                        prev_flow,
                        physics_norm,
                        identity,
                    )
                    feedback_rows.append(
                        {
                            "group": name,
                            "candidate_index": candidate,
                            "step": k,
                            "actuator_index": actuator,
                            "actuator_id": str(graph.actuator_ids[actuator]),
                            "setting_delta": float(
                                cand_setting[k, actuator] - ref_setting[k, actuator]
                            ),
                            "true_flow_effect_m3s": float(
                                cand_flows[k, actuator] - ref_flows[k, actuator]
                            ),
                            "predicted_flow_effect_m3s": float(
                                q[1, actuator] - q[0, actuator]
                            ),
                        }
                    )

    if not direct_rows or not feedback_rows:
        raise RuntimeError("current flow audit produced no direct or feedback rows")
    direct_truth = np.asarray([float(r["true_flow_effect_m3s"]) for r in direct_rows])
    direct_pred = np.asarray([float(r["predicted_flow_effect_m3s"]) for r in direct_rows])
    feedback_truth = np.asarray([float(r["true_flow_effect_m3s"]) for r in feedback_rows])
    feedback_pred = np.asarray([float(r["predicted_flow_effect_m3s"]) for r in feedback_rows])
    direct_metrics = _metrics(direct_truth, direct_pred)
    feedback_metrics = _metrics(feedback_truth, feedback_pred)
    direct_macro, direct_top = _macro_by_actuator(direct_rows)
    feedback_macro, feedback_top = _macro_by_actuator(feedback_rows)

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
        "direct_same_prefix_semantics": (
            "first setting-divergence only; identical authoritative previous state/flow supplied "
            "to reference and candidate"
        ),
        "feedback_full_horizon_semantics": (
            "branch-specific authoritative previous state/flow; includes hydraulic feedback and "
            "is not a local dq/du"
        ),
        "flow_effect_metrics": direct_metrics,
        "direct_same_prefix_metrics": direct_metrics,
        "direct_actuator_balanced_metrics": direct_macro,
        "direct_top_truth_l2_contributors": direct_top,
        "feedback_full_horizon_metrics": feedback_metrics,
        "feedback_actuator_balanced_metrics": feedback_macro,
        "feedback_top_truth_l2_contributors": feedback_top,
        "promotion_metric": (
            "direct_same_prefix plus actuator-balanced macro; never feedback micro cosine alone"
        ),
        "used_for_training": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
    }
    for section in (direct_metrics, feedback_metrics, direct_macro, feedback_macro):
        for value in section.values():
            if isinstance(value, float) and not math.isfinite(value):
                raise RuntimeError("action-to-flow audit produced a non-finite metric")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("ACTION_TO_FLOW_DIRECT_DETAIL.csv", direct_rows),
        ("ACTION_TO_FLOW_FEEDBACK_DETAIL.csv", feedback_rows),
    ):
        with (out / filename).open("w", newline="", encoding="utf-8") as fh:
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
