"""Compare one source-strict nonfinal V128 stage on the frozen smoke/dev held-out D2 groups.

Development diagnostic only. This entrypoint exists to localize where action-gradient collapse
appears across stage_a -> stage_b0 -> objective. It never trains, never rewrites a checkpoint,
and never accesses Validation/Final/Formal/Policy Lock.
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
from rtc.step2_d4_cache_v125 import D4_SOURCE_KIND
from rtc.step2_differentiable_v128 import build_v128_model_from_graph
from rtc.step2_gradient_audit_v128_dev import evaluate_d2_gradient_v128_development
from rtc.step2_state_store_v127 import CausalStep1StateCacheV127, derive_v127_input_normalization, load_causal_state_store_v127
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_train_v127_streaming import V127StreamingMemoryDesign
from rtc.v128_control_profile import build_v128_control_training_design, configure_v128_cuda_matmul_precision

CONTRACT = "PROJECT7_CURRENT_V128_NONFINAL_STAGE_GRADIENT_DECOMPOSITION_V1"


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
    p.add_argument("--stage", choices=("stage_a", "stage_b0", "objective"), required=True)
    p.add_argument("--stage-checkpoint", required=True)
    p.add_argument("--graph", required=True)
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
        raise ValueError("stage gradient decomposition requires exactly 109 actuators")

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
    online = CausalStep1StateCacheV127(CausalForecastValueCacheV123(base, rain_store), state_store)
    normalization = derive_v127_input_normalization(
        base_cache=base,
        causal_rainfall=rain_store,
        causal_state=state_store,
        fit_names=selected["fit_d2"] + selected["fit_d3"],
    )
    first_state = state_store.state_for(base.entry(selected["fit_d2"][0]))
    model = build_v128_model_from_graph(
        graph,
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
    base_manifest = json.loads(Path(args.cache_manifest).read_text(encoding="utf-8"))
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
        "swmm_engine_version": str(base_manifest["swmm_engine_version"]),
    }
    payload = load_stage_checkpoint_v128(
        args.stage_checkpoint,
        model=model,
        expected_profile=profile.name,
        graph_path=args.graph,
        expected_lineage=lineage,
        expected_training_design=asdict(design),
    )
    actual_stage = str(payload.get("completed_stage"))
    if actual_stage != args.stage:
        raise ValueError(f"requested stage {args.stage} but checkpoint contains {actual_stage}")

    rows, metrics = evaluate_d2_gradient_v128_development(
        model,
        graph=graph,
        base_cache=base,
        online_cache=online,
        names=selected["hold_d2"],
        normalization=normalization,
        device=device,
        flood_rate_index=2,
    )
    metrics.update(
        {
            "outer_contract": CONTRACT,
            "profile": profile.name,
            "completed_stage": actual_stage,
            "stage_checkpoint_sha256": _sha(args.stage_checkpoint),
            "selected_holdout_d2_groups": selected["hold_d2"],
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
        }
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    import csv
    with (out / "D2_DEVELOPMENT_GRADIENT_DETAIL.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "D2_DEVELOPMENT_GRADIENT_METRICS.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
