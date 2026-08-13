"""Canonical causal execution-bound Project7 V120 trainer.

The SWMM outcomes in the existing D2/D3 cache are unchanged.  Only the model
rainfall input is replaced by the online-legal checkpoint persistence/decay
forecast, removing future-realised rainfall leakage from Step2 Value learning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from run_step2_v120_final import (
    _branch_count,
    _candidate_count,
    _gate,
    _git_head,
    _load_frozen_train_events,
    _load_graph,
    _sha256,
)
from rtc.code_contract import rtc_implementation_contract_sha256
from rtc.step2_causal_forecast_v120 import (
    CausalForecastCacheV120,
    V120_CAUSAL_RAINFALL_CONTRACT,
    derive_causal_input_normalization_v120,
)
from rtc.step2_control_basis_v60 import basis_manifest_v60, build_control_basis_v60
from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_control_response_v70 import ControlValueSurrogateV70
from rtc.step2_optimization_v70 import train_value_event_balanced_v70
from rtc.step2_policy_v120 import RuntimeNormalizationV120, candidate_coefficients_v120
from rtc.step2_shards_v60 import validate_v60_cache_lineage
from rtc.step2_train_response_v60 import (
    V60TrainCache,
    derive_magnitude_strata_v60,
    deterministic_rainfall_split_v60,
)
from rtc.step2_train_response_v70 import (
    derive_target_scales_v70,
    evaluate_value_strata_v70,
    evaluate_value_v70,
)
from rtc.step2_v120_contract import (
    SOURCE_D2_AUTHORITATIVE_BRANCH_CENSUS,
    TARGETED_D3_AUTHORITATIVE_BRANCH_CENSUS,
    Step2V120Contract,
    V120_BUNDLE_CONTRACT,
    V120_CANDIDATE_POLICY_CONTRACT,
    V120_CONTRACT,
    V120_DATA_ELIGIBILITY_CONTRACT,
    v120_graph_semantic_sha256,
    v120_runtime_contract_sha256,
)
from rtc.step2_v70_contract import DirectValueLossContractV70, V70_CONTRACT

CAUSAL_DECAY_PER_STEP = 0.92


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train causal execution-bound Project7 Step2 V120"
    )
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--swmm-engine-version", required=True)
    parser.add_argument(
        "--split-contract",
        default=str(repo / "configs" / "project7_v069_split_contract.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()

    contract = Step2V120Contract(seed=int(args.seed))
    contract.validate()
    split_payload, frozen_train = _load_frozen_train_events(args.split_contract)
    graph = _load_graph(args.graph)
    if len(graph.actuator_ids) != contract.actuator_count:
        raise ValueError("V120 requires frozen 109 actuators")

    truth_cache = V60TrainCache(args.cache_manifest)
    legacy = truth_cache.legacy_d3_names()
    if legacy:
        raise ValueError(f"V120 rejects legacy dense D3 groups: {legacy[:3]}")
    d2 = truth_cache.names("D2")
    d3 = truth_cache.targeted_d3_names()
    if not d2 or not d3:
        raise ValueError("V120 requires existing D2 plus targeted D3-v2")

    cache_events = {truth_cache.entry(name).event_id for name in sorted(d2 + d3)}
    extra = sorted(cache_events - frozen_train)
    missing = sorted(frozen_train - cache_events)
    if extra:
        raise ValueError(
            "P0 V120 data leakage: cache contains non-Train events: " + ", ".join(extra)
        )
    if missing:
        raise ValueError(
            "V120 canonical cache is incomplete for frozen Train18: " + ", ".join(missing)
        )
    if len(cache_events) != contract.training_event_count:
        raise ValueError("V120 cache must contain exactly frozen Train18")

    d2_branches = _branch_count(truth_cache, d2)
    d3_branches = _branch_count(truth_cache, d3)
    if d3_branches != TARGETED_D3_AUTHORITATIVE_BRANCH_CENSUS:
        raise ValueError(
            f"V120 requires targeted D3=3600 branches; got {d3_branches}"
        )

    fit, holdout = deterministic_rainfall_split_v60(
        truth_cache,
        names=sorted(d2 + d3),
        holdout_fraction=float(args.holdout_fraction),
    )
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    fit_d3 = [name for name in fit if name.startswith("D3::")]
    holdout_d2 = [name for name in holdout if name.startswith("D2::")]
    holdout_d3 = [name for name in holdout if name.startswith("D3::")]
    if not all((fit_d2, fit_d3, holdout_d2, holdout_d3)):
        raise ValueError("V120 internal split must contain D2/D3 on both sides")
    fit_events = sorted({truth_cache.entry(name).event_id for name in fit})
    holdout_events = sorted({truth_cache.entry(name).event_id for name in holdout})
    fit_rain = sorted({truth_cache.entry(name).rainfall_group for name in fit})
    holdout_rain = sorted({truth_cache.entry(name).rainfall_group for name in holdout})
    if set(fit_events) & set(holdout_events) or set(fit_rain) & set(holdout_rain):
        raise ValueError("V120 event/rainfall internal split leakage")
    if len(fit_events) != 14 or len(holdout_events) != 4:
        raise ValueError(
            f"V120 requires frozen 14/4 internal split; got {len(fit_events)}/{len(holdout_events)}"
        )

    # Model inputs use only checkpoint rainfall.  Targets/scales continue to come
    # from the original authoritative SWMM branches.
    norm = derive_causal_input_normalization_v120(
        truth_cache, fit, decay_per_step=CAUSAL_DECAY_PER_STEP
    )
    causal_cache = CausalForecastCacheV120(
        truth_cache, decay_per_step=CAUSAL_DECAY_PER_STEP
    )
    scales = derive_target_scales_v70(truth_cache, fit)
    strata = derive_magnitude_strata_v60(truth_cache, fit_d3)
    basis = build_control_basis_v60(graph)
    first = truth_cache.entry(fit[0])
    state_dim = int(first.arrays["initial_state"].shape[-1])
    rainfall_dim = int(first.arrays["rainfall"].shape[-1])
    if state_dim != 6:
        raise ValueError("V120 requires frozen six-state Step1 interface")
    prepared_cpu = prepare_static_v60(graph, "cpu")
    model = ControlValueSurrogateV70(
        state_dim=state_dim,
        rainfall_dim=rainfall_dim,
        physics_dim=int(prepared_cpu.actuator_physics.shape[1]),
        actuator_count=len(graph.actuator_ids),
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        tfv_scale_m3=float(scales.direct_tfv_scale_m3),
        hidden_dim=96,
        actuator_embedding_dim=16,
    )
    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    history = train_value_event_balanced_v70(
        model,
        causal_cache,  # type: ignore[arg-type]
        fit_d2_names=fit_d2,
        fit_d3_names=fit_d3,
        normalization=norm,
        scales=scales,
        graph=graph,
        device=str(device),
        seed=int(args.seed),
        contract=DirectValueLossContractV70(),
    )
    prepared = prepare_static_v60(graph, device)
    metrics = {
        "fit_d2": evaluate_value_v70(
            model, causal_cache, fit_d2, norm, prepared, device=device  # type: ignore[arg-type]
        ),
        "fit_d3": evaluate_value_v70(
            model, causal_cache, fit_d3, norm, prepared, device=device  # type: ignore[arg-type]
        ),
        "holdout_d2": evaluate_value_v70(
            model, causal_cache, holdout_d2, norm, prepared, device=device  # type: ignore[arg-type]
        ),
        "holdout_d3": evaluate_value_v70(
            model, causal_cache, holdout_d3, norm, prepared, device=device  # type: ignore[arg-type]
        ),
    }
    magnitude = evaluate_value_strata_v70(
        model,
        causal_cache,  # type: ignore[arg-type]
        holdout_d3,
        norm,
        prepared,
        device=device,
        q33_m3=float(strata["q33_m3"]),
        q67_m3=float(strata["q67_m3"]),
    )
    gate_ok, reasons = _gate(metrics["holdout_d3"])
    cache_lineage = validate_v60_cache_lineage(args.cache_manifest)
    normalization = RuntimeNormalizationV120(
        norm.state_mean,
        norm.state_std,
        norm.rainfall_mean,
        norm.rainfall_std,
        norm.flow_mean,
        norm.flow_std,
    )
    engine = str(args.swmm_engine_version).strip()
    if not engine:
        raise ValueError("V120 requires a non-empty SWMM engine version")

    model_config = {
        "state_dim": state_dim,
        "rainfall_dim": rainfall_dim,
        "physics_dim": int(prepared_cpu.actuator_physics.shape[1]),
        "actuator_count": len(graph.actuator_ids),
        "hidden_dim": 96,
        "actuator_embedding_dim": 16,
        "direct_tfv_scale_m3": float(scales.direct_tfv_scale_m3),
        "swmm_engine_version": engine,
    }
    candidate_count = int(candidate_coefficients_v120(basis, contract=contract).shape[0])
    bundle = {
        "bundle_contract": V120_BUNDLE_CONTRACT,
        "step2_contract": V120_CONTRACT,
        "base_value_contract": V70_CONTRACT,
        "kind": "control_value_only_execution_bound_causal_rainfall",
        "state_dict": model.state_dict(),
        "model_config": model_config,
        "input_normalization": normalization.as_payload(),
        "basis_manifest": basis_manifest_v60(basis),
        "rtc_implementation_contract_sha256": rtc_implementation_contract_sha256(),
        "v120_runtime_contract_sha256": v120_runtime_contract_sha256(),
        "lineage": {
            "git_head": _git_head(),
            "graph_sha256": _sha256(args.graph),
            "graph_semantic_sha256": v120_graph_semantic_sha256(graph),
            "cache_manifest_sha256": _sha256(args.cache_manifest),
            "split_contract_sha256": _sha256(args.split_contract),
            "training_entrypoint_sha256": _sha256(Path(__file__)),
            "basis_sha256_from_cache_lineage": str(
                cache_lineage["v60_control_basis_sha256"]
            ),
            "design_sha256_from_cache_lineage": str(
                cache_lineage["v60_design_contract_sha256"]
            ),
        },
        "rainfall_input": {
            "contract": V120_CAUSAL_RAINFALL_CONTRACT,
            "future_realized_rainfall_used_as_model_input": False,
            "checkpoint_rainfall_only": True,
            "decay_per_step": CAUSAL_DECAY_PER_STEP,
            "training_scenario_multiplier": 1.0,
            "runtime_scenario_multipliers": [0.75, 1.0, 1.25],
            "required_runtime_history_steps_for_level": 1,
        },
        "data_census": {
            "contract": V120_DATA_ELIGIBILITY_CONTRACT,
            "source_d2_authoritative_branch_census": SOURCE_D2_AUTHORITATIVE_BRANCH_CENSUS,
            "source_d2_census_role": "upstream intervention census; not pooled across frozen split",
            "eligible_cache_d2_groups": len(d2),
            "eligible_cache_d2_branches": d2_branches,
            "eligible_cache_d2_candidates": _candidate_count(truth_cache, d2),
            "targeted_d3_groups": len(d3),
            "targeted_d3_branches": d3_branches,
            "targeted_d3_candidates": _candidate_count(truth_cache, d3),
            "eligible_event_count": len(cache_events),
            "eligible_events": sorted(cache_events),
            "non_train_events_loaded": [],
        },
        "split": {
            "scientific_split_contract": str(split_payload["contract"]),
            "fit_event_ids": fit_events,
            "holdout_event_ids": holdout_events,
            "fit_rainfall_groups": fit_rain,
            "holdout_rainfall_groups": holdout_rain,
            "event_overlap": 0,
            "rainfall_overlap": 0,
            "validation_events_loaded": 0,
            "final_events_loaded": 0,
        },
        "candidate_policy": {
            "contract": V120_CANDIDATE_POLICY_CONTRACT,
            "candidate_count_including_hold": candidate_count,
            "candidate_banks": contract.candidate_banks,
            "families": [
                "hold",
                "single_group",
                "same_zone",
                "cross_zone",
                "sparse_low_discrepancy",
            ],
            "continuous_gradient_search": False,
            "score_only_executable_sequences": True,
            "first_move_bound_to_current_and_target_readback": True,
        },
        "value_horizon_minutes": 360,
        "control_update_seconds": 600,
        "runtime_compatible": bool(gate_ok),
        "production_compatible": False,
        "value_gate": {"passed": bool(gate_ok), "reasons": reasons},
        "metrics": metrics,
        "magnitude_strata": magnitude,
        "training_history": history,
        "boundary": {
            "new_swmm": False,
            "hydraulic_model_trained": False,
            "nodewise_action_effect_required": False,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
        },
    }

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle_path = out / "step2_v120_execution_bound_causal_bundle.pt"
    torch.save(bundle, bundle_path)
    report = {key: value for key, value in bundle.items() if key != "state_dict"}
    (out / "STEP2_V120_EXECUTION_BOUND_CAUSAL_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not gate_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
