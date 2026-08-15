"""Audit current V128 Step2 action effects by actuator-to-node graph distance.

Designed for the smoke/dev funnel: point it at ``stage_objective.pt`` after a staged run.
It reconstructs the identical deterministic profile/data design, verifies the NONFINAL stage
checkpoint lineage, and evaluates held-out D2 same-prefix SWMM action effects.  It never
promotes a smoke/dev checkpoint or touches Validation/Final/Formal.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from rtc.development_profile_v128 import apply_profile_to_design, get_execution_profile, profile_groups
from rtc.production_cli import _load_graph
from rtc.stage_checkpoint_v128 import load_stage_checkpoint_v128
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_d4_cache_v125 import D4_SOURCE_KIND
from rtc.step2_differentiable_v128 import build_v128_model_from_graph
from rtc.step2_spatial_audit_v128 import evaluate_d2_spatial_action_effect_v128
from rtc.step2_state_store_v127 import CausalStep1StateCacheV127, derive_v127_input_normalization, load_causal_state_store_v127
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_train_v127 import configure_model_normalization_v127
from rtc.step2_train_v127_streaming import V127StreamingMemoryDesign, derive_residual_scales_streaming_v127
from rtc.v128_control_profile import build_v128_control_training_design, configure_v128_cuda_matmul_precision

CONTRACT = "PROJECT7_CURRENT_STEP2_SPATIAL_ACTION_EFFECT_AUDIT_V1_DEVELOPMENT_ONLY"


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rain(cache: V60TrainCache, names: list[str]) -> set[str]:
    return {str(cache.entry(name).rainfall_group) for name in names}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=("smoke", "dev"), required=True)
    p.add_argument("--stage-checkpoint", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--d4-fit-cache", required=True)
    p.add_argument("--d4-audit-cache", required=True)
    p.add_argument("--causal-store", required=True)
    p.add_argument("--causal-state-store", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--flood-rate-index", type=int, default=2)
    p.add_argument("--hydraulic-branch-chunk", type=int, default=4)
    p.add_argument("--rollout-candidates-per-group", type=int, default=2)
    p.add_argument("--objective-candidate-chunk", type=int, default=2)
    p.add_argument("--evaluation-branch-chunk", type=int, default=4)
    p.add_argument("--residual-sample-rows", type=int, default=131072)
    args = p.parse_args()

    configure_v128_cuda_matmul_precision()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    profile = get_execution_profile(args.profile)
    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != 109:
        raise ValueError("current spatial audit requires exactly 109 actuators")

    base = V60TrainCache(args.cache_manifest)
    d4_fit_raw = V60TrainCache(args.d4_fit_cache)
    d4_audit_raw = V60TrainCache(args.d4_audit_cache)
    rain_store = load_causal_forecast_store_v123(args.causal_store)
    state_store = load_causal_state_store_v127(args.causal_state_store)
    fit, holdout = deterministic_rainfall_split_v60(
        base,
        names=sorted(base.names("D2") + base.targeted_d3_names()),
        holdout_fraction=0.20,
    )
    fit_d2 = [x for x in fit if x.startswith("D2::")]
    fit_d3 = [x for x in fit if x.startswith("D3::")]
    hold_d2 = [x for x in holdout if x.startswith("D2::")]
    hold_d3 = [x for x in holdout if x.startswith("D3::")]
    if tuple(map(len, (fit_d2, fit_d3, hold_d2, hold_d3))) != (112, 112, 32, 32):
        raise ValueError("canonical D2/D3 split differs from 112/112/32/32")
    d4_fit_names = d4_fit_raw.names(D4_SOURCE_KIND)
    d4_audit_names = d4_audit_raw.names(D4_SOURCE_KIND)
    if (len(d4_fit_names), len(d4_audit_names)) != (33, 15):
        raise ValueError("canonical D4 FIT/AUDIT census differs from 33/15")
    if _rain(d4_fit_raw, d4_fit_names) & _rain(d4_audit_raw, d4_audit_names):
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

    base_online = CausalStep1StateCacheV127(CausalForecastValueCacheV123(base, rain_store), state_store)
    normalization = derive_v127_input_normalization(
        base_cache=base,
        causal_rainfall=rain_store,
        causal_state=state_store,
        fit_names=selected["fit_d2"] + selected["fit_d3"],
    )
    sample_rows = min(int(args.residual_sample_rows), 8192 if profile.name == "smoke" else 32768)
    state_scale, flow_scale, _ = derive_residual_scales_streaming_v127(
        ((base, selected["fit_d2"] + selected["fit_d3"]), (d4_fit_raw, selected["d4_fit"])),
        sample_rows=sample_rows,
    )
    first_state = state_store.state_for(base.entry(selected["fit_d2"][0]))
    model = build_v128_model_from_graph(
        graph,
        state_dim=int(first_state.shape[-1]),
        rainfall_dim=int(rain_store.forecast_mmhr.shape[-1]),
        delta_state_scale=state_scale,
        delta_flow_scale=flow_scale,
    )
    configure_model_normalization_v127(
        model,
        normalization=normalization,
        graph=graph,
        state_delta_scale=state_scale,
        flow_delta_scale=flow_scale,
    )

    memory = V127StreamingMemoryDesign(
        hydraulic_branch_chunk=int(args.hydraulic_branch_chunk),
        rollout_candidates_per_group=int(args.rollout_candidates_per_group),
        objective_candidate_chunk=int(args.objective_candidate_chunk),
        evaluation_branch_chunk=int(args.evaluation_branch_chunk),
        residual_sample_rows=int(args.residual_sample_rows),
    )
    design = build_v128_control_training_design(
        hydraulic_branch_chunk=memory.hydraulic_branch_chunk,
        rollout_candidates_per_group=memory.rollout_candidates_per_group,
        objective_candidate_chunk=memory.objective_candidate_chunk,
        evaluation_branch_chunk=memory.evaluation_branch_chunk,
    )
    design = apply_profile_to_design(design, profile)
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
        "swmm_engine_version": str(json.loads(Path(args.cache_manifest).read_text(encoding="utf-8"))["swmm_engine_version"]),
    }
    stage = load_stage_checkpoint_v128(
        args.stage_checkpoint,
        model=model,
        expected_profile=profile.name,
        graph_path=args.graph,
        expected_lineage=lineage,
        expected_training_design=asdict(design),
    )
    if str(stage.get("completed_stage")) != "objective":
        raise ValueError("spatial action-effect audit requires the completed objective stage checkpoint")

    audit = evaluate_d2_spatial_action_effect_v128(
        model,
        cache=base_online,
        names=selected["hold_d2"],
        normalization=normalization,
        graph=graph,
        device=device,
        flood_rate_index=int(args.flood_rate_index),
        branch_chunk=int(args.evaluation_branch_chunk),
        effect_floor_m3=1.0,
    )
    payload = {
        "contract": CONTRACT,
        "profile": profile.name,
        "scientific_claim_allowed": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
        "stage_checkpoint_sha256": _sha(args.stage_checkpoint),
        "stage_completed": stage["completed_stage"],
        "selected_holdout_d2_groups": selected["hold_d2"],
        "audit": audit,
        "promotion_rule": (
            "Use this report to reject spatially weak variants. Do not promote architecture "
            "changes to full until held-out far-field sign/magnitude improves without material "
            "near-field/ranking regression."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
