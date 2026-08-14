"""Train/evaluate the Project7 V127 control-oriented differentiable surrogate.

First run uses only existing D2/D3/D4 authoritative SWMM labels.  It does not launch
SWMM, consume Validation/Final/Formal, or train on D4-AUDIT/InternalHoldout.  All model
inputs are online-equivalent: frozen causal Step1 state + causal rainfall forecast +
current actuator flow + continuous target sequence.  Future SWMM hydraulics/flooding are
labels only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from rtc.checkpoint_v127 import save_step2_v127
from rtc.code_contract import rtc_implementation_contract_sha256
from rtc.production_cli import _load_graph
from rtc.step2_causal_rainfall_v123 import (
    CausalForecastValueCacheV123,
    load_causal_forecast_store_v123,
)
from rtc.step2_d4_cache_v125 import D4CausalForecastValueCacheV125, D4_SOURCE_KIND
from rtc.step2_differentiable_v127 import V127SurrogateDesign, build_v127_model_from_graph
from rtc.step2_state_store_v127 import (
    CausalStep1StateCacheV127,
    derive_v127_input_normalization,
    load_causal_state_store_v127,
)
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_train_v127 import (
    V127TrainingDesign,
    configure_model_normalization_v127,
    derive_residual_scales_v127,
    evaluate_objective_groups_v127,
    train_hydraulic_stage_v127,
    train_objective_stage_v127,
)

V127_RUN_CONTRACT = "PROJECT7_V127_EXISTING_D2_D3_D4_DIFFERENTIABLE_TRAINING_V1"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rain(cache, names: list[str]) -> set[str]:
    return {str(cache.entry(name).rainfall_group) for name in names}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True)
    p.add_argument("--cache-manifest", required=True)
    p.add_argument("--d4-fit-cache", required=True)
    p.add_argument("--d4-audit-cache", required=True)
    p.add_argument("--causal-store", required=True)
    p.add_argument("--causal-state-store", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--depth-index", type=int, default=0)
    p.add_argument("--flood-rate-index", type=int, default=2)
    args = p.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != 109:
        raise ValueError("V127 requires the frozen 109-actuator graph")
    base = V60TrainCache(args.cache_manifest)
    d4_fit_raw = V60TrainCache(args.d4_fit_cache)
    d4_audit_raw = V60TrainCache(args.d4_audit_cache)
    rainfall_store = load_causal_forecast_store_v123(args.causal_store)
    state_store = load_causal_state_store_v127(args.causal_state_store)

    selected = sorted(base.names("D2") + base.targeted_d3_names())
    fit, holdout = deterministic_rainfall_split_v60(base, names=selected, holdout_fraction=0.20)
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    fit_d3 = [name for name in fit if name.startswith("D3::")]
    hold_d2 = [name for name in holdout if name.startswith("D2::")]
    hold_d3 = [name for name in holdout if name.startswith("D3::")]
    if (len(fit_d2), len(fit_d3), len(hold_d2), len(hold_d3)) != (112, 112, 32, 32):
        raise ValueError("V127 canonical D2/D3 rainfall split differs from frozen 112/112/32/32")
    d4_fit_names = d4_fit_raw.names(D4_SOURCE_KIND)
    d4_audit_names = d4_audit_raw.names(D4_SOURCE_KIND)
    if (len(d4_fit_names), len(d4_audit_names)) != (33, 15):
        raise ValueError("V127 expects the completed D4 FIT/AUDIT group census 33/15")
    if _rain(d4_fit_raw, d4_fit_names) & _rain(d4_audit_raw, d4_audit_names):
        raise ValueError("V127 D4 FIT/AUDIT rainfall leakage")

    # First replace future-realised rain, then replace SWMM checkpoint truth with causal
    # Step1 state.  The order is explicit and testable.
    base_causal_rain = CausalForecastValueCacheV123(base, rainfall_store)
    d4_fit_causal_rain = D4CausalForecastValueCacheV125(d4_fit_raw, rainfall_store)
    d4_audit_causal_rain = D4CausalForecastValueCacheV125(d4_audit_raw, rainfall_store)
    base_online = CausalStep1StateCacheV127(base_causal_rain, state_store)
    d4_fit_online = CausalStep1StateCacheV127(d4_fit_causal_rain, state_store)
    d4_audit_online = CausalStep1StateCacheV127(d4_audit_causal_rain, state_store)

    normalization = derive_v127_input_normalization(
        base_cache=base,
        causal_rainfall=rainfall_store,
        causal_state=state_store,
        fit_names=fit,
    )
    state_scale, flow_scale = derive_residual_scales_v127(
        ((base, fit_d2 + fit_d3), (d4_fit_raw, d4_fit_names))
    )
    first = state_store.state_for(base.entry(fit_d2[0]))
    design = V127SurrogateDesign()
    design.validate()
    model = build_v127_model_from_graph(
        graph,
        state_dim=int(first.shape[-1]),
        rainfall_dim=int(rainfall_store.forecast_mmhr.shape[-1]),
        delta_state_scale=state_scale,
        delta_flow_scale=flow_scale,
        design=design,
    )
    configure_model_normalization_v127(
        model,
        normalization=normalization,
        graph=graph,
        state_delta_scale=state_scale,
        flow_delta_scale=flow_scale,
    )
    train_design = V127TrainingDesign()
    source_caches = {"D2": base_online, "D3": base_online, "D4": d4_fit_online}
    source_groups = {"D2": fit_d2, "D3": fit_d3, "D4": d4_fit_names}
    hydraulic_history = train_hydraulic_stage_v127(
        model,
        source_caches=source_caches,
        source_groups=source_groups,
        normalization=normalization,
        graph=graph,
        device=device,
        depth_index=int(args.depth_index),
        flood_rate_index=int(args.flood_rate_index),
        design=train_design,
    )
    objective_history = train_objective_stage_v127(
        model,
        source_caches=source_caches,
        source_groups=source_groups,
        normalization=normalization,
        graph=graph,
        device=device,
        flood_rate_index=int(args.flood_rate_index),
        design=train_design,
    )

    evaluations = {
        "trainfit_d2": evaluate_objective_groups_v127(
            model, cache=base_online, names=fit_d2, normalization=normalization, graph=graph,
            device=device, flood_rate_index=int(args.flood_rate_index),
        ),
        "trainfit_d3": evaluate_objective_groups_v127(
            model, cache=base_online, names=fit_d3, normalization=normalization, graph=graph,
            device=device, flood_rate_index=int(args.flood_rate_index),
        ),
        "internal_holdout_d2": evaluate_objective_groups_v127(
            model, cache=base_online, names=hold_d2, normalization=normalization, graph=graph,
            device=device, flood_rate_index=int(args.flood_rate_index),
        ),
        "internal_holdout_d3": evaluate_objective_groups_v127(
            model, cache=base_online, names=hold_d3, normalization=normalization, graph=graph,
            device=device, flood_rate_index=int(args.flood_rate_index),
        ),
        "d4_fit": evaluate_objective_groups_v127(
            model, cache=d4_fit_online, names=d4_fit_names, normalization=normalization, graph=graph,
            device=device, flood_rate_index=int(args.flood_rate_index),
        ),
        "d4_audit": evaluate_objective_groups_v127(
            model, cache=d4_audit_online, names=d4_audit_names, normalization=normalization, graph=graph,
            device=device, flood_rate_index=int(args.flood_rate_index),
        ),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "contract": V127_RUN_CONTRACT,
        "step2_contract": model.v127_contract,
        "training_contract": "PROJECT7_V127_HYDRAULIC_THEN_FLOOD_OBJECTIVE_CURRICULUM_V1",
        "rtc_implementation_contract_sha256": rtc_implementation_contract_sha256(),
        "scientific_goal": "differentiable hydraulic response and TFV gradients for continuous MPC",
        "data_roles": {
            "D2": "single-actuator hydraulic sensitivity and Jacobian support",
            "targeted_D3": "joint continuous-action nonlinear response",
            "D4_FIT": "local physical-response support; not an anchor-value objective",
            "D4_AUDIT": "read-only local response/ranking audit",
        },
        "counts": {
            "fit_d2_groups": len(fit_d2), "fit_d3_groups": len(fit_d3),
            "holdout_d2_groups": len(hold_d2), "holdout_d3_groups": len(hold_d3),
            "d4_fit_groups": len(d4_fit_names), "d4_audit_groups": len(d4_audit_names),
        },
        "normalization": {
            "state_source": "causal Step1 TrainFit estimates",
            "rainfall_source": "causal persistence-decay TrainFit forecast",
            "future_swmm_truth_used_as_input": False,
        },
        "hydraulic_history": hydraulic_history,
        "objective_history": objective_history,
        "evaluations": evaluations,
        "lineage": {
            "graph_sha256": _sha(args.graph),
            "base_cache_sha256": _sha(args.cache_manifest),
            "d4_fit_cache_sha256": _sha(args.d4_fit_cache),
            "d4_audit_cache_sha256": _sha(args.d4_audit_cache),
            "causal_rainfall_sha256": _sha(args.causal_store),
            "causal_state_store_sha256": _sha(args.causal_state_store),
        },
        "boundary": {
            "new_swmm": False,
            "d4_audit_used_for_training": False,
            "internal_holdout_used_for_training": False,
            "d2_development_validation_used_for_training": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
            "continuous_mpc_authorized_by_training_alone": False,
        },
    }
    report_path = out / "STEP2_V127_EXISTING_DATA_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    checkpoint = save_step2_v127(
        out / "step2_v127_existing_data.pt",
        model=model,
        graph=graph,
        training_report=report,
        lineage=report["lineage"],
    )
    report["checkpoint"] = str(checkpoint.resolve())
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
