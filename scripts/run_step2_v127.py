"""Train/evaluate Project7 V127 Step2 from existing D2/D3/D4; no new SWMM."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from rtc.checkpoint_v127 import save_step2_v127
from rtc.code_contract import rtc_implementation_contract_sha256
from rtc.production_cli import _load_graph
from rtc.step2_causal_rainfall_v123 import CausalForecastValueCacheV123, load_causal_forecast_store_v123
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

V127_RUN_CONTRACT = "PROJECT7_V127_EXISTING_D2_D3_D4_DIFFERENTIABLE_TRAINING_V3_CAUSAL_INPUT"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rain(cache, names) -> set[str]:
    return {str(cache.entry(name).rainfall_group) for name in names}


def _finite_metrics(name: str, metrics: dict[str, float]) -> None:
    required = ("rank", "pairwise", "top1", "tfv_mae_m3", "selected_regret_m3")
    missing = [key for key in required if key not in metrics]
    if missing:
        raise RuntimeError(f"{name}: V127 evaluation lacks {missing}")
    bad = [key for key in required if not math.isfinite(float(metrics[key]))]
    if bad:
        raise RuntimeError(f"{name}: V127 evaluation produced non-finite metrics {bad}")


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
        raise ValueError("V127 requires exactly 109 frozen writable actuators")
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
        raise ValueError("V127 canonical D2/D3 split differs from frozen 112/112/32/32")
    d4_fit_names = d4_fit_raw.names(D4_SOURCE_KIND)
    d4_audit_names = d4_audit_raw.names(D4_SOURCE_KIND)
    if (len(d4_fit_names), len(d4_audit_names)) != (33, 15):
        raise ValueError("V127 requires completed D4 FIT/AUDIT group census 33/15")
    if _rain(d4_fit_raw, d4_fit_names) & _rain(d4_audit_raw, d4_audit_names):
        raise ValueError("V127 D4 FIT/AUDIT rainfall leakage")

    base_online = CausalStep1StateCacheV127(CausalForecastValueCacheV123(base, rain_store), state_store)
    d4_fit_online = CausalStep1StateCacheV127(D4CausalForecastValueCacheV125(d4_fit_raw, rain_store), state_store)
    d4_audit_online = CausalStep1StateCacheV127(D4CausalForecastValueCacheV125(d4_audit_raw, rain_store), state_store)
    normalization = derive_v127_input_normalization(
        base_cache=base,
        causal_rainfall=rain_store,
        causal_state=state_store,
        fit_names=fit,
    )
    state_scale, flow_scale = derive_residual_scales_v127(
        ((base, fit_d2 + fit_d3), (d4_fit_raw, d4_fit_names))
    )
    first_state = state_store.state_for(base.entry(fit_d2[0]))
    model = build_v127_model_from_graph(
        graph,
        state_dim=int(first_state.shape[-1]),
        rainfall_dim=int(rain_store.forecast_mmhr.shape[-1]),
        delta_state_scale=state_scale,
        delta_flow_scale=flow_scale,
        design=V127SurrogateDesign(),
    )
    configure_model_normalization_v127(
        model,
        normalization=normalization,
        graph=graph,
        state_delta_scale=state_scale,
        flow_delta_scale=flow_scale,
    )
    source_caches = {"D2": base_online, "D3": base_online, "D4": d4_fit_online}
    source_groups = {"D2": fit_d2, "D3": fit_d3, "D4": d4_fit_names}
    train_design = V127TrainingDesign()
    hydraulic_history = train_hydraulic_stage_v127(
        model,
        source_caches=source_caches,
        source_groups=source_groups,
        normalization=normalization,
        graph=graph,
        device=device,
        depth_index=args.depth_index,
        flood_rate_index=args.flood_rate_index,
        design=train_design,
    )
    objective_history = train_objective_stage_v127(
        model,
        source_caches=source_caches,
        source_groups=source_groups,
        normalization=normalization,
        graph=graph,
        device=device,
        flood_rate_index=args.flood_rate_index,
        design=train_design,
    )
    specs = {
        "trainfit_d2": (base_online, fit_d2),
        "trainfit_d3": (base_online, fit_d3),
        "internal_holdout_d2": (base_online, hold_d2),
        "internal_holdout_d3": (base_online, hold_d3),
        "d4_fit": (d4_fit_online, d4_fit_names),
        "d4_audit": (d4_audit_online, d4_audit_names),
    }
    evaluations = {}
    for key, (cache, names) in specs.items():
        metrics = evaluate_objective_groups_v127(
            model,
            cache=cache,
            names=names,
            normalization=normalization,
            graph=graph,
            device=device,
            flood_rate_index=args.flood_rate_index,
        )
        _finite_metrics(key, metrics)
        evaluations[key] = metrics

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lineage = {
        "graph_sha256": _sha(args.graph),
        "base_cache_sha256": _sha(args.cache_manifest),
        "d4_fit_cache_sha256": _sha(args.d4_fit_cache),
        "d4_audit_cache_sha256": _sha(args.d4_audit_cache),
        "causal_rainfall_sha256": _sha(args.causal_store),
        "causal_state_store_sha256": _sha(args.causal_state_store),
    }
    report = {
        "contract": V127_RUN_CONTRACT,
        "step2_contract": model.v127_contract,
        "rtc_implementation_contract_sha256": rtc_implementation_contract_sha256(),
        "scientific_goal": "differentiable hydraulic response and trustworthy action gradients for continuous MPC",
        "data_roles": {
            "D2": "single-actuator hydraulic sensitivity/Jacobian support",
            "targeted_D3": "joint multi-actuator nonlinear response",
            "D4_FIT": "local physical-response support, not anchor-Value target",
            "D4_AUDIT": "read-only local response/ranking audit",
        },
        "normalization": {
            "state_source": "causal Step1 TrainFit estimates",
            "rainfall_source": "causal runtime-equivalent TrainFit forecasts",
            "future_swmm_truth_used_as_input": False,
        },
        "hydraulic_history": hydraulic_history,
        "objective_history": objective_history,
        "evaluations": evaluations,
        "lineage": lineage,
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
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    checkpoint = save_step2_v127(
        out / "step2_v127_existing_data.pt",
        model=model,
        graph=graph,
        input_normalization=normalization,
        training_report=report,
        lineage=lineage,
    )
    report["checkpoint"] = str(checkpoint.resolve())
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
