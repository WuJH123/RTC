"""Development-only same-prefix normal-vs-oracle hydraulic action-effect audit.

Each held-out D2 pair is evaluated only at its first setting-divergence transition. Reference and
candidate share one authoritative pre-action hydraulic state/managed-flow prefix. The normal
branch uses the candidate/reference settings and predicted managed flow. The oracle branch blocks
the branch-varying setting side channel inside the hydraulic transition and substitutes stored
authoritative SWMM managed flow, so its direct effect is a strict ``managed flow -> hydraulics``
isolation. Rainfall comes from the frozen causal forecast store, never realised future rainfall.
This is Development diagnostic evidence only and is never an online/training/Final truth source.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
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
from rtc.step2_oracle_isolation_v128 import (
    ORACLE_FLOW_ISOLATION_CONTRACT,
    oracle_flow_transition_prediction,
    shared_reference_setting,
)
from rtc.step2_state_store_v127 import load_causal_state_store_v127
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_train_v127 import _static
from rtc.step2_train_v127_streaming import V127StreamingMemoryDesign
from rtc.v128_control_profile import build_v128_control_training_design, configure_v128_cuda_matmul_precision

CONTRACT = "PROJECT7_CURRENT_DIRECT_SAME_PREFIX_NORMAL_VS_ORACLE_HYDRAULIC_AUDIT_V4_Q_ONLY"


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rain_groups(cache: V60TrainCache, names: list[str]) -> set[str]:
    return {str(cache.entry(name).rainfall_group) for name in names}


def _effect_metrics(
    truth: list[np.ndarray] | list[float], pred: list[np.ndarray] | list[float]
) -> dict[str, float | int]:
    t = np.asarray(truth, dtype=np.float64).reshape(-1)
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(t) & np.isfinite(p) & (np.abs(t) > 1.0e-8)
    t, p = t[mask], p[mask]
    if t.size == 0:
        return {
            "informative_count": 0,
            "sign_accuracy": 0.0,
            "effect_mae": 0.0,
            "relative_mae": 0.0,
            "cosine": 0.0,
        }
    denom = float(np.linalg.norm(t) * np.linalg.norm(p))
    return {
        "informative_count": int(t.size),
        "sign_accuracy": float(np.mean(np.sign(t) == np.sign(p))),
        "effect_mae": float(np.mean(np.abs(p - t))),
        "relative_mae": float(np.mean(np.abs(p - t) / np.maximum(np.abs(t), 1.0e-8))),
        "cosine": float(np.dot(t, p) / denom) if denom > 1.0e-12 else 0.0,
    }


def _normal_transition(model, *, prev_state, prev_flow, setting, rainfall, static):
    batch = int(prev_state.shape[0])
    physics_norm, identity = model.actuator.prepare_static(static["physics"], batch_size=batch)
    static_norm, edges, inv = model.transition.prepare_static(
        static["static"], static["edges"], batch_size=batch, dtype=prev_state.dtype
    )
    q, response = model.actuator.forward_prepared(
        prev_state[:, static["up"]],
        prev_state[:, static["down"]],
        setting,
        prev_flow,
        physics_norm,
        identity,
    )
    injection = torch.zeros(
        batch, prev_state.shape[1], 1, device=prev_state.device, dtype=prev_state.dtype
    )
    injection = injection.index_add(1, static["up"], -q[..., None]).index_add(
        1, static["down"], q[..., None]
    )
    context = model._typed_action_context(
        state=prev_state,
        setting=setting,
        previous_flow=prev_flow,
        predicted_flow=q,
        responsiveness=response,
        upstream=static["up"],
        downstream=static["down"],
        physics_norm=physics_norm,
        identity_embedding=identity,
    )
    state = model.transition.forward_prepared(
        prev_state, rainfall, static_norm, injection, edges, inv, context
    )
    return state, q


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
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--depth-index", type=int, default=0)
    p.add_argument("--flood-rate-index", type=int, default=2)
    args = p.parse_args()

    configure_v128_cuda_matmul_precision()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    profile = get_execution_profile(args.profile)
    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != 109:
        raise ValueError("direct hydraulic audit requires the frozen 109-actuator graph")
    base = V60TrainCache(args.cache_manifest)
    d4_fit = V60TrainCache(args.d4_fit_cache)
    d4_audit = V60TrainCache(args.d4_audit_cache)
    rain_store = load_causal_forecast_store_v123(args.causal_store)
    state_store = load_causal_state_store_v127(args.causal_state_store)
    rain_store.validate()
    state_store.validate()
    rain_index = rain_store.index()

    fit, holdout = deterministic_rainfall_split_v60(
        base,
        names=sorted(base.names("D2") + base.targeted_d3_names()),
        holdout_fraction=0.20,
    )
    fit_d2 = [x for x in fit if x.startswith("D2::")]
    fit_d3 = [x for x in fit if x.startswith("D3::")]
    hold_d2 = [x for x in holdout if x.startswith("D2::")]
    hold_d3 = [x for x in holdout if x.startswith("D3::")]
    d4_fit_names = d4_fit.names(D4_SOURCE_KIND)
    d4_audit_names = d4_audit.names(D4_SOURCE_KIND)
    if tuple(map(len, (fit_d2, fit_d3, hold_d2, hold_d3))) != (112, 112, 32, 32):
        raise ValueError("canonical D2/D3 split differs from 112/112/32/32")
    if (len(d4_fit_names), len(d4_audit_names)) != (33, 15):
        raise ValueError("canonical D4 FIT/AUDIT census differs from 33/15")
    if _rain_groups(d4_fit, d4_fit_names) & _rain_groups(d4_audit, d4_audit_names):
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
    lineage = extend_action_identifiable_stage_lineage(
        {
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
        },
        edge_physics_path=args.edge_physics,
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
        raise ValueError("requested stage differs from checkpoint")

    model = model.to(device).eval()
    static = _static(graph, device)
    true_depth: list[np.ndarray] = []
    normal_depth: list[np.ndarray] = []
    oracle_depth: list[np.ndarray] = []
    true_flood: list[np.ndarray] = []
    normal_flood: list[np.ndarray] = []
    oracle_flood: list[np.ndarray] = []
    true_flow_effect: list[float] = []
    normal_flow_effect: list[float] = []
    pairs = 0

    with torch.inference_mode():
        for name in selected["hold_d2"]:
            entry = base.entry(name)
            arrays = entry.arrays
            ref = int(entry.reference_index)
            for candidate_raw in entry.indices:
                candidate = int(candidate_raw)
                if candidate == ref:
                    continue
                spec = first_direct_response_spec_numpy(
                    arrays, reference=ref, candidate=candidate, require_single_actuator=True
                )
                if spec is None:
                    continue
                k, actuator = int(spec["step"]), int(spec["actuator_index"])
                if k == 0:
                    prefix_state = np.asarray(arrays["initial_state"][ref], dtype=np.float32)
                    prefix_flow = np.asarray(arrays["previous_actuator_flow"][ref], dtype=np.float32)
                else:
                    prefix_state = np.asarray(arrays["target_states"][ref][k - 1], dtype=np.float32)
                    prefix_flow = np.asarray(arrays["target_actuator_flows"][ref][k - 1], dtype=np.float32)
                prev_state = torch.as_tensor(
                    np.stack((prefix_state, prefix_state)), dtype=torch.float32, device=device
                )
                prev_flow = torch.as_tensor(
                    np.stack((prefix_flow, prefix_flow)), dtype=torch.float32, device=device
                )
                setting = torch.as_tensor(
                    np.stack((arrays["settings"][ref][k], arrays["settings"][candidate][k])),
                    dtype=torch.float32,
                    device=device,
                )
                causal_rain = np.asarray(rain_store.forecast_mmhr[rain_index[name], k], dtype=np.float32)
                rainfall = torch.as_tensor(
                    np.stack((causal_rain, causal_rain)), dtype=torch.float32, device=device
                )
                normal_state, normal_q = _normal_transition(
                    model,
                    prev_state=prev_state,
                    prev_flow=prev_flow,
                    setting=setting,
                    rainfall=rainfall,
                    static=static,
                )
                true_q = torch.as_tensor(
                    np.stack(
                        (
                            arrays["target_actuator_flows"][ref][k],
                            arrays["target_actuator_flows"][candidate][k],
                        )
                    ),
                    dtype=torch.float32,
                    device=device,
                )
                oracle_state = oracle_flow_transition_prediction(
                    model,
                    prev_state=prev_state,
                    previous_flow=prev_flow,
                    setting=setting,
                    oracle_flow=true_q,
                    rainfall=rainfall,
                    static=static,
                    action_context_setting=shared_reference_setting(setting),
                )
                truth_state = np.stack(
                    (arrays["target_states"][ref][k], arrays["target_states"][candidate][k])
                ).astype(np.float32)
                truth_delta = truth_state[1] - truth_state[0]
                normal_delta = (normal_state[1] - normal_state[0]).cpu().numpy()
                oracle_delta = (oracle_state[1] - oracle_state[0]).cpu().numpy()
                true_depth.append(truth_delta[:, args.depth_index])
                normal_depth.append(normal_delta[:, args.depth_index])
                oracle_depth.append(oracle_delta[:, args.depth_index])
                true_flood.append(truth_delta[:, args.flood_rate_index])
                normal_flood.append(normal_delta[:, args.flood_rate_index])
                oracle_flood.append(oracle_delta[:, args.flood_rate_index])
                true_flow_effect.append(float(true_q[1, actuator] - true_q[0, actuator]))
                normal_flow_effect.append(float(normal_q[1, actuator] - normal_q[0, actuator]))
                pairs += 1

    if pairs == 0:
        raise RuntimeError("direct hydraulic audit found no same-prefix held-out pairs")
    result = {
        "contract": CONTRACT,
        "oracle_flow_isolation_contract": ORACLE_FLOW_ISOLATION_CONTRACT,
        "profile": profile.name,
        "completed_stage": args.stage,
        "scientific_split": "development",
        "scientific_claim_allowed": False,
        "same_prefix_pairs": int(pairs),
        "causal_rainfall": True,
        "normal_flow_effect": _effect_metrics(true_flow_effect, normal_flow_effect),
        "depth": {
            "normal_vs_swmm": _effect_metrics(true_depth, normal_depth),
            "oracle_flow_only_vs_swmm": _effect_metrics(true_depth, oracle_depth),
        },
        "flood_rate": {
            "normal_vs_swmm": _effect_metrics(true_flood, normal_flood),
            "oracle_flow_only_vs_swmm": _effect_metrics(true_flood, oracle_flood),
        },
        "oracle_flow_role": "diagnostic q-to-hydraulics isolation only; not used online",
        "oracle_direct_pair_setting_bypass_blocked": True,
        "oracle_branch_varying_control_signal": "authoritative_managed_flow_only",
        "gradient_label_used": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
    }
    Path(args.out).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
