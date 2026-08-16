"""Fused ranking + H30-H360 hydraulic audit for current Development stage checkpoints.

Unlike ``audit_step2_v128_fast.py``, this entrypoint does not require a promoted production
checkpoint. It rebuilds the current counterfactual-first model/edge/source lineage, restores an
exact NONFINAL ``stage_b0`` or ``objective`` checkpoint with ``strict=True``, and evaluates the
same deterministic smoke/dev split. This makes ranking/horizon evidence an executable gate before
H360 objective and before any production promotion.
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
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
from rtc.step2_current_dev_context_v128 import (
    build_current_action_identifiable_model,
    extend_action_identifiable_stage_lineage,
)
from rtc.step2_d4_cache_v125 import D4CausalForecastValueCacheV125, D4_SOURCE_KIND
from rtc.step2_eval_v127_fast import V127_FAST_EVAL_CONTRACT, evaluate_groups_fast_v127
from rtc.step2_state_store_v127 import CausalStep1StateCacheV127, derive_v127_input_normalization, load_causal_state_store_v127
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_train_v127_streaming import V127StreamingMemoryDesign
from rtc.v128_control_profile import build_v128_control_training_design, configure_v128_cuda_matmul_precision

CONTRACT = "PROJECT7_CURRENT_STAGE_RANKING_HORIZON_AUDIT_V1_NONFINAL_SOURCE_STRICT"


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
    p.add_argument("--stage", choices=("stage_b0", "objective"), required=True)
    p.add_argument("--stage-checkpoint", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--edge-physics", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--d4-fit-cache", required=True)
    p.add_argument("--d4-audit-cache", required=True)
    p.add_argument("--causal-store", required=True)
    p.add_argument("--causal-state-store", required=True)
    p.add_argument("--ranking-out", required=True)
    p.add_argument("--horizon-out", required=True)
    p.add_argument("--telemetry-out", required=True)
    p.add_argument("--depth-index", type=int, default=0)
    p.add_argument("--flood-rate-index", type=int, default=2)
    p.add_argument("--branch-chunk", type=int, default=12)
    p.add_argument("--device", default="cuda")
    p.add_argument("--hydraulic-branch-chunk", type=int, default=4)
    p.add_argument("--rollout-candidates-per-group", type=int, default=2)
    p.add_argument("--objective-candidate-chunk", type=int, default=2)
    p.add_argument("--evaluation-branch-chunk", type=int, default=4)
    p.add_argument("--residual-sample-rows", type=int, default=131072)
    args = p.parse_args()

    if int(args.branch_chunk) <= 0:
        raise ValueError("stage ranking audit branch chunk must be positive")
    configure_v128_cuda_matmul_precision()
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    profile = get_execution_profile(args.profile)
    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != 109:
        raise ValueError("current stage ranking audit requires exactly 109 actuators")

    base = V60TrainCache(args.cache_manifest)
    d4_fit_raw = V60TrainCache(args.d4_fit_cache)
    d4_audit_raw = V60TrainCache(args.d4_audit_cache)
    rain = load_causal_forecast_store_v123(args.causal_store)
    state = load_causal_state_store_v127(args.causal_state_store)
    rain.validate()
    state.validate()

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
    base_online = CausalStep1StateCacheV127(CausalForecastValueCacheV123(base, rain), state)
    d4_fit = CausalStep1StateCacheV127(D4CausalForecastValueCacheV125(d4_fit_raw, rain), state)
    d4_audit = CausalStep1StateCacheV127(D4CausalForecastValueCacheV125(d4_audit_raw, rain), state)
    normalization = derive_v127_input_normalization(
        base_cache=base,
        causal_rainfall=rain,
        causal_state=state,
        fit_names=selected["fit_d2"] + selected["fit_d3"],
    )

    first_state = state.state_for(base.entry(selected["fit_d2"][0]))
    state_dim = int(first_state.shape[-1])
    model = build_current_action_identifiable_model(
        graph,
        edge_physics_path=args.edge_physics,
        state_dim=state_dim,
        rainfall_dim=int(rain.forecast_mmhr.shape[-1]),
        delta_state_scale=np.ones(state_dim, dtype=np.float32),
        delta_flow_scale=np.ones(len(graph.actuator_ids), dtype=np.float32),
        direct_action_flow_scale=np.ones(len(graph.actuator_ids), dtype=np.float32),
    )

    memory = V127StreamingMemoryDesign(
        hydraulic_branch_chunk=int(args.hydraulic_branch_chunk),
        rollout_candidates_per_group=int(args.rollout_candidates_per_group),
        objective_candidate_chunk=int(args.objective_candidate_chunk),
        evaluation_branch_chunk=int(args.evaluation_branch_chunk),
        residual_sample_rows=int(args.residual_sample_rows),
    )
    memory.validate()
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
            "causal_state_step1_file_sha256": str(state.step1_sha256),
            "causal_state_sensor_file_sha256": str(state.sensor_sha256),
            "causal_state_graph_file_sha256": str(state.graph_sha256),
            "causal_state_step1_model_semantic_sha256": str(state.step1_model_semantic_sha256),
            "causal_state_sensor_layout_semantic_sha256": str(state.sensor_layout_semantic_sha256),
            "causal_rainfall_forecast_contract": str(rain.forecast_contract),
            "swmm_engine_version": str(manifest["swmm_engine_version"]),
        },
        edge_physics_path=args.edge_physics,
    )
    stage_payload = load_stage_checkpoint_v128(
        args.stage_checkpoint,
        model=model,
        expected_profile=profile.name,
        graph_path=args.graph,
        expected_lineage=lineage,
        expected_training_design=asdict(design),
    )
    if str(stage_payload.get("completed_stage")) != args.stage:
        raise ValueError(
            f"stage ranking audit requested {args.stage} but checkpoint contains "
            f"{stage_payload.get('completed_stage')}"
        )
    model = model.to(device).eval()

    specs = {
        "trainfit_d2": (base_online, selected["fit_d2"]),
        "trainfit_d3": (base_online, selected["fit_d3"]),
        "internal_holdout_d2": (base_online, selected["hold_d2"]),
        "internal_holdout_d3": (base_online, selected["hold_d3"]),
        "d4_fit": (d4_fit, selected["d4_fit"]),
        "d4_audit": (d4_audit, selected["d4_audit"]),
    }
    checkpoint_sha = _sha(args.stage_checkpoint)
    ranking: dict[str, object] = {
        "contract": CONTRACT + "_RANKING",
        "stage": args.stage,
        "profile": profile.name,
        "stage_checkpoint_sha256": checkpoint_sha,
    }
    horizon: dict[str, object] = {
        "contract": CONTRACT + "_HORIZON",
        "stage": args.stage,
        "profile": profile.name,
        "stage_checkpoint_sha256": checkpoint_sha,
        "results": {},
        "metric_semantics": {
            "hydraulic_truth": "authoritative SWMM target state/managed-flow trajectories at 300 s",
            "initial_input": "causal Step1 reconstruction",
            "purpose": "diagnose autoregressive error growth and control ordering before production promotion",
        },
    }
    telemetry: dict[str, object] = {
        "contract": CONTRACT,
        "execution_contract": V127_FAST_EVAL_CONTRACT,
        "stage": args.stage,
        "profile": profile.name,
        "stage_checkpoint_sha256": checkpoint_sha,
        "splits": {},
    }
    for key, (cache, names) in specs.items():
        r, h, t = evaluate_groups_fast_v127(
            model,
            cache=cache,
            names=names,
            normalization=normalization,
            graph=graph,
            device=device,
            depth_index=int(args.depth_index),
            flood_rate_index=int(args.flood_rate_index),
            requested_branch_chunk=int(args.branch_chunk),
        )
        ranking[key] = r
        horizon["results"][key] = h
        telemetry["splits"][key] = t

    boundary = {
        "read_only": True,
        "scientific_claim_allowed": False,
        "stage_checkpoint_nonfinal": True,
        "internal_holdout_used_for_training": False,
        "d4_audit_used_for_training": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
        "production_checkpoint_required": False,
    }
    common_lineage = {
        "graph_sha256": _sha(args.graph),
        "edge_physics_sha256": _sha(args.edge_physics),
        "cache_manifest_sha256": _sha(args.cache_manifest),
        "d4_fit_cache_sha256": _sha(args.d4_fit_cache),
        "d4_audit_cache_sha256": _sha(args.d4_audit_cache),
        "causal_rainfall_sha256": _sha(args.causal_store),
        "causal_state_store_sha256": _sha(args.causal_state_store),
    }
    ranking["lineage"] = common_lineage
    ranking["boundary"] = boundary
    horizon["lineage"] = common_lineage
    horizon["boundary"] = boundary
    telemetry["lineage"] = common_lineage
    telemetry["boundary"] = boundary

    for path_text, payload_out in (
        (args.ranking_out, ranking),
        (args.horizon_out, horizon),
        (args.telemetry_out, telemetry),
    ):
        path = Path(path_text)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload_out, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(telemetry, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
