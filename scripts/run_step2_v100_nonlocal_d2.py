"""Run the single pre-registered Project7 V10 nonlocal D2 mechanism test.

This is a structural replacement test, not a hyperparameter search. It uses only
canonical TrainFit/InternalHoldout D2 data, frozen V7 reference Hydraulic, and
frozen-INP regulator-aware actuator/node geometry. No SWMM, D3, Validation,
Final, Formal, oracle reference trajectory, or online all-link flow is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from run_step2_v90_physical_edge_d2 import (
    _assert_full_horizon_causality,
    _build_reference,
    _git_head,
    _load_checkpoint,
    _load_graph,
    _sha256,
    _validate_canonical_schedule,
    _validate_v7_lineage,
)

from rtc.step2_control_basis_v60 import build_control_basis_v60
from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_control_response_v80 import prepare_static_v80
from rtc.step2_control_response_v100 import (
    DirectHydraulicEffectSurrogateV100,
    build_actuator_node_influence_assets_v100,
)
from rtc.step2_hydraulic_eval_v90 import PRIMARY_SKILL_KEYS, evaluate_hydraulic_effect_v90
from rtc.step2_hydraulic_objective_v90 import derive_onset_sqrt_positive_weight_v80
from rtc.step2_optimization_v90 import candidate_batch_chunks_v90
from rtc.step2_optimization_v100 import train_d2_nonlocal_mechanism_v100
from rtc.step2_shards_v60 import validate_v60_cache_lineage
from rtc.step2_train_response_v60 import (
    V60TrainCache,
    derive_input_normalization_v60,
    deterministic_rainfall_split_v60,
)
from rtc.step2_train_response_v70 import derive_target_scales_v70
from rtc.step2_v100_contract import NonlocalHydraulicEffectLossContractV100, V100_CONTRACT


_PRECHECK_CHUNK = 2


def _implementation_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    members = (
        "scripts/run_step2_v100_nonlocal_d2.py",
        "src/rtc/step2_v100_contract.py",
        "src/rtc/step2_control_response_v100.py",
        "src/rtc/step2_optimization_v100.py",
        "src/rtc/step2_hydraulic_objective_v90.py",
        "src/rtc/step2_hydraulic_eval_v90.py",
    )
    digest = hashlib.sha256()
    for relative in members:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _preflight(model, cache, name, normalization, graph, device, assets):
    target = torch.device(device)
    model.to(target).eval()
    prepared = prepare_static_v80(graph, target)
    batch = cache.batch(name, normalization, target)
    with torch.no_grad():
        zero = model(
            batch.initial_state,
            batch.rainfall,
            batch.reference_settings,
            batch.reference_settings[:, None],
            batch.previous_actuator_flow,
            prepared,
        )
    if not torch.equal(zero.raw_delta_states_physical, torch.zeros_like(zero.raw_delta_states_physical)):
        raise RuntimeError("V10 exact-zero state effect failed")
    if not torch.equal(zero.raw_delta_flows_physical, torch.zeros_like(zero.raw_delta_flows_physical)):
        raise RuntimeError("V10 exact-zero flow effect failed")
    if any(parameter.requires_grad for parameter in model.reference_model.parameters()):
        raise RuntimeError("V10 found trainable frozen V7 Hydraulic parameters")
    if assets.uses_future_truth or assets.uses_online_link_flow:
        raise RuntimeError("V10 nonlocal assets violate causal/static contract")
    if assets.regulator_count != len(graph.actuator_ids):
        raise RuntimeError("V10 regulator-aware topology lost one or more controlled links")

    first_chunk = next(candidate_batch_chunks_v90(batch, candidate_chunk_size=_PRECHECK_CHUNK))
    candidate = first_chunk.candidate_settings.detach().clone().requires_grad_(True)
    output = model(
        first_chunk.initial_state,
        first_chunk.rainfall,
        first_chunk.reference_settings,
        candidate,
        first_chunk.previous_actuator_flow,
        prepared,
    )
    score = output.raw_delta_states_physical.square().mean() + output.raw_delta_flows_physical.square().mean()
    (gradient,) = torch.autograd.grad(score, candidate, allow_unused=False)
    if not bool(torch.isfinite(gradient).all()):
        raise RuntimeError("V10 action gradient is non-finite")
    gradient_nonzero = int(torch.count_nonzero(gradient).item())
    if gradient_nonzero == 0:
        raise RuntimeError("V10 action gradient is identically zero")
    _assert_full_horizon_causality(
        candidate_settings=candidate,
        baseline_output=output,
        forward=lambda settings: model(
            first_chunk.initial_state,
            first_chunk.rainfall,
            first_chunk.reference_settings,
            settings,
            first_chunk.previous_actuator_flow,
            prepared,
        ),
    )
    return {
        "signed_state_exact_zero": True,
        "signed_flow_exact_zero": True,
        "reference_frozen": True,
        "future_action_causality": True,
        "action_gradient_nonzero_fraction": float(gradient_nonzero / gradient.numel()),
        "regulators_present_in_relation_geometry": int(assets.regulator_count),
        "reachable_actuator_node_pair_fraction": float(assets.reachable_pair_fraction),
        "finite_hop_cutoff": False,
        "uses_future_truth": False,
        "uses_online_link_flow": False,
    }


def _primary_skills(metrics: dict) -> dict[str, float]:
    overall = metrics["overall"]
    return {key: float(overall[key]) for key in PRIMARY_SKILL_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Project7 V10 nonlocal hydraulic operator D2 mechanism")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--v70-value-checkpoint", required=True)
    parser.add_argument("--v70-hydraulic-checkpoint", required=True)
    parser.add_argument("--frozen-inp", required=True)
    parser.add_argument("--expected-inp-sha256", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()
    _validate_canonical_schedule(seed=args.seed, holdout_fraction=args.holdout_fraction)

    started = time.perf_counter()
    graph = _load_graph(args.graph)
    if len(graph.node_ids) != 932 or len(graph.actuator_ids) != 109:
        raise ValueError("V10 requires the frozen 932-node / 109-actuator Project7 graph")
    cache = V60TrainCache(args.cache_manifest)
    if cache.legacy_d3_names():
        raise ValueError("V10 refuses legacy dense D3")
    d2, d3 = cache.names("D2"), cache.targeted_d3_names()
    fit, holdout = deterministic_rainfall_split_v60(
        cache, names=sorted(d2 + d3), holdout_fraction=float(args.holdout_fraction)
    )
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    holdout_d2 = [name for name in holdout if name.startswith("D2::")]
    if len(fit_d2) != 112 or len(holdout_d2) != 32:
        raise ValueError("V10 requires canonical 112/32 TrainFit/InternalHoldout D2 split")
    fit_events = {cache.entry(name).event_id for name in fit}
    holdout_events = {cache.entry(name).event_id for name in holdout}
    fit_rain = {cache.entry(name).rainfall_group for name in fit}
    holdout_rain = {cache.entry(name).rainfall_group for name in holdout}
    if fit_events & holdout_events or fit_rain & holdout_rain:
        raise ValueError("V10 TrainFit/InternalHoldout leakage detected")

    cache_lineage = validate_v60_cache_lineage(args.cache_manifest)
    normalization = derive_input_normalization_v60(cache, fit)
    scales = derive_target_scales_v70(cache, fit)
    basis = build_control_basis_v60(graph)
    value_checkpoint = _load_checkpoint(args.v70_value_checkpoint, "control_value")
    hydraulic_checkpoint = _load_checkpoint(args.v70_hydraulic_checkpoint, "hydraulic_response")
    v7_lineage = _validate_v7_lineage(
        value_checkpoint=value_checkpoint,
        hydraulic_checkpoint=hydraulic_checkpoint,
        graph_sha256=_sha256(args.graph),
        cache_manifest_sha256=_sha256(args.cache_manifest),
        cache_lineage=cache_lineage,
    )
    influence_assets = build_actuator_node_influence_assets_v100(
        inp_path=args.frozen_inp,
        expected_inp_sha256=args.expected_inp_sha256,
        node_ids=graph.node_ids,
        actuator_ids=graph.actuator_ids,
        actuator_upstream=graph.actuator_upstream,
        actuator_downstream=graph.actuator_downstream,
    )
    contract = NonlocalHydraulicEffectLossContractV100()
    contract.validate()
    onset_positive_weight = derive_onset_sqrt_positive_weight_v80(
        cache, fit, normalization, contract=contract
    )

    reference = _build_reference(graph, cache, fit, scales)
    reference.load_state_dict(hydraulic_checkpoint["state_dict"], strict=True)
    prepared_cpu = prepare_static_v60(graph, "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = DirectHydraulicEffectSurrogateV100(
        reference_model=reference,
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        state_delta_scale=scales.state_delta_scale,
        flow_delta_scale=scales.flow_delta_scale,
        physics_dim=int(prepared_cpu.actuator_physics.shape[1]),
        node_static_dim=int(graph.static_node_features.shape[1]),
        actuator_count=len(graph.actuator_ids),
        influence_assets=influence_assets,
        contract=contract,
    )
    target = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    preflight = _preflight(model, cache, fit_d2[0], normalization, graph, str(target), influence_assets)
    history = train_d2_nonlocal_mechanism_v100(
        model,
        cache,
        fit_d2_names=fit_d2,
        normalization=normalization,
        scales=scales,
        onset_positive_weight=onset_positive_weight,
        graph=graph,
        device=str(target),
        seed=args.seed,
        contract=contract,
    )
    prepared = prepare_static_v80(graph, target)
    fit_metrics = evaluate_hydraulic_effect_v90(
        model, cache, fit_d2, normalization, prepared, scales,
        graph=graph, device=target, contract=contract,
    )
    holdout_metrics = evaluate_hydraulic_effect_v90(
        model, cache, holdout_d2, normalization, prepared, scales,
        graph=graph, device=target, contract=contract,
    )
    holdout_skills = _primary_skills(holdout_metrics)
    mechanism_supported = all(value > 0.0 for value in holdout_skills.values())
    decision = (
        "V100_D2_NONLOCAL_MECHANISM_SUPPORTED"
        if mechanism_supported
        else "V100_D2_NONLOCAL_MECHANISM_NOT_YET_SUPPORTED"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": V100_CONTRACT,
        "development_only": True,
        "production_compatible": False,
        "swmm_run": False,
        "validation_accessed": False,
        "final_accessed": False,
        "formal_accessed": False,
        "d3_training_run": False,
        "oracle_reference_used": False,
        "new_link_flow_used": False,
        "lineage": {
            "git_head": _git_head(),
            "implementation_sha256": _implementation_sha256(),
            "graph_sha256": _sha256(args.graph),
            "cache_manifest_sha256": _sha256(args.cache_manifest),
            "value_checkpoint_sha256": _sha256(args.v70_value_checkpoint),
            "hydraulic_checkpoint_sha256": _sha256(args.v70_hydraulic_checkpoint),
            "frozen_inp_sha256": influence_assets.inp_sha256,
            "cache_lineage": cache_lineage,
            "v7_lineage": v7_lineage,
            "fit_d2_group_count": len(fit_d2),
            "holdout_d2_group_count": len(holdout_d2),
            "fit_d2_group_digest": hashlib.sha256("\n".join(sorted(fit_d2)).encode()).hexdigest(),
            "seed": int(args.seed),
        },
        "operator": {
            "type": "regulator_aware_all_range_actuator_to_node_operator",
            "finite_hop_cutoff": False,
            "pair_feature_names": list(influence_assets.pair_feature_names),
            "pair_count": influence_assets.pair_count,
            "reachable_pair_fraction": influence_assets.reachable_pair_fraction,
            "physical_link_count": influence_assets.physical_link_count,
            "conduit_count": influence_assets.conduit_count,
            "regulator_count": influence_assets.regulator_count,
            "multi_actuator_semantics": "latent source aggregation followed by zero-centred nonlinear node interaction; not sum of D2 output effects",
        },
        "schedule": {
            "source": "TRAINFIT_D2_ONLY",
            "epochs": int(contract.d2_pretrain_epochs),
            "optimizer": "AdamW",
            "learning_rate": float(contract.learning_rate),
            "weight_decay": float(contract.weight_decay),
            "grad_clip": float(contract.grad_clip),
            "candidate_chunk_size": int(contract.candidate_chunk_size),
            "seed": int(args.seed),
            "fp32": True,
            "no_sweep": True,
        },
        "preflight": preflight,
        "training_history": history,
        "metrics": {"TrainFit_D2": fit_metrics, "TrainInternalHoldout_D2": holdout_metrics},
        "holdout_primary_skill": holdout_skills,
        "decision": decision,
        "formal_v10_authorized": False,
        "d3_authorized": bool(mechanism_supported),
        "new_swmm_authorized": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report = out_dir / "STEP2_V100_NONLOCAL_D2_MECHANISM.json"
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report), "decision": decision, "holdout": holdout_skills}, indent=2), flush=True)


if __name__ == "__main__":
    main()
