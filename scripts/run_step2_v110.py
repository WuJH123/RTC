"""Canonical Train-only runner for Project7 Step2 V11.

Stages are explicit and never chained automatically:
- d2: 4-epoch single-actuator mechanism learning/evaluation.
- d3: 10-epoch targeted multi-actuator interaction learning with a 0.25 D2 anchor;
      requires an accepted D2 V11 report/checkpoint.

No SWMM, Validation, Final, Formal or production wiring is touched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import torch

from rtc.graph import GraphSchema
from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_control_response_v70 import HydraulicResponseSurrogateV70
from rtc.step2_control_response_v110 import (
    ActuatorSetHydraulicResponseV110,
    build_actuator_node_relations_v110,
)
from rtc.step2_hydraulic_eval_v110 import evaluate_hydraulic_effect_v110
from rtc.step2_hydraulic_objective_v110 import derive_effect_scales_v110
from rtc.step2_optimization_v110 import train_d2_mechanism_v110, train_d3_interaction_v110
from rtc.step2_shards_v60 import validate_v60_cache_lineage
from rtc.step2_train_response_v60 import (
    V60TrainCache,
    derive_input_normalization_v60,
    deterministic_rainfall_split_v60,
)
from rtc.step2_v60_contract import MultiResolutionHorizonV60
from rtc.step2_v110_contract import (
    ActuatorSetHydraulicContractV110,
    HydraulicHorizonV110,
    V110_CONTRACT,
)


def _load_graph(path: str | Path) -> GraphSchema:
    with np.load(path, allow_pickle=False) as raw:
        return GraphSchema(
            node_ids=tuple(raw["node_ids"].astype(str).tolist()),
            edge_index=raw["edge_index"].astype(np.int64),
            static_node_features=raw["static_node_features"].astype(np.float32),
            static_node_feature_names=tuple(raw["static_node_feature_names"].astype(str).tolist()),
            actuator_ids=tuple(raw["actuator_ids"].astype(str).tolist()),
            actuator_upstream=raw["actuator_upstream"].astype(np.int64),
            actuator_downstream=raw["actuator_downstream"].astype(np.int64),
            actuator_physics=raw["actuator_physics"].astype(np.float32),
            actuator_physics_feature_names=tuple(raw["actuator_physics_feature_names"].astype(str).tolist()),
            system_units=str(raw["system_units"].item()),
        )


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _load_checkpoint(path: str | Path, expected_kind: str) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != expected_kind:
        raise ValueError(f"checkpoint {path} is not V7 {expected_kind}")
    if "state_dict" not in payload:
        raise ValueError(f"checkpoint {path} lacks state_dict")
    return payload


def _build_reference(graph, cache, names, hydraulic_checkpoint):
    entry = cache.entry(names[0])
    state_dim = int(entry.arrays["initial_state"].shape[-1])
    rainfall_dim = int(entry.arrays["rainfall"].shape[-1])
    prepared = prepare_static_v60(graph, "cpu")
    state_dict = hydraulic_checkpoint["state_dict"]
    if "state_scale" not in state_dict or "flow_scale" not in state_dict:
        raise ValueError("V7 hydraulic checkpoint lacks physical scale buffers")
    model = HydraulicResponseSurrogateV70(
        state_scale=state_dict["state_scale"], flow_scale=state_dict["flow_scale"],
        horizon_contract=MultiResolutionHorizonV60(), state_dim=state_dim,
        rainfall_dim=rainfall_dim, node_static_dim=int(graph.static_node_features.shape[1]),
        physics_dim=int(prepared.actuator_physics.shape[1]), actuator_count=len(graph.actuator_ids),
        hidden_dim=64, latent_dim=32, temporal_dim=12,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _preflight(model, cache, name, normalization, prepared, device):
    batch = cache.batch(name, normalization, device)
    model.to(device).eval()
    reference = batch.reference_settings[:, None]
    with torch.no_grad():
        zero = model(batch.initial_state, batch.rainfall, batch.reference_settings, reference,
                     batch.previous_actuator_flow, prepared)
    if not torch.equal(zero.raw_delta_states_physical, torch.zeros_like(zero.raw_delta_states_physical)):
        raise RuntimeError("V11 exact-zero state effect failed")
    if not torch.equal(zero.raw_delta_flows_physical, torch.zeros_like(zero.raw_delta_flows_physical)):
        raise RuntimeError("V11 exact-zero flow effect failed")

    future = reference.clone()
    change_step = 18
    current = future[:, :, change_step:, 0]
    replacement = torch.where(current < 0.5, current + 0.25, current - 0.25).clamp(0.0, 1.0)
    future[:, :, change_step:, 0] = replacement
    with torch.no_grad():
        delayed = model(batch.initial_state, batch.rainfall, batch.reference_settings, future,
                        batch.previous_actuator_flow, prepared)
    early = delayed.horizon_indices < change_step
    if not torch.equal(delayed.raw_delta_states_physical[:, :, early],
                       torch.zeros_like(delayed.raw_delta_states_physical[:, :, early])):
        raise RuntimeError("V11 future action leaked into earlier state response")
    if not torch.equal(delayed.raw_delta_flows_physical[:, :, early],
                       torch.zeros_like(delayed.raw_delta_flows_physical[:, :, early])):
        raise RuntimeError("V11 future action leaked into earlier flow response")

    candidate = batch.candidate_settings[:, :1].detach().clone().requires_grad_(True)
    output = model(batch.initial_state, batch.rainfall, batch.reference_settings, candidate,
                   batch.previous_actuator_flow, prepared)
    score = output.raw_delta_states_physical.square().mean() + output.raw_delta_flows_physical.square().mean()
    gradient = torch.autograd.grad(score, candidate)[0]
    if not bool(torch.isfinite(gradient).all()) or int(torch.count_nonzero(gradient)) == 0:
        raise RuntimeError("V11 action gradient is non-finite or identically zero")
    if any(parameter.requires_grad for parameter in model.reference_model.parameters()):
        raise RuntimeError("V11 frozen V7 reference became trainable")
    return {
        "exact_zero": True, "future_action_causality": True,
        "action_gradient_finite": True, "action_gradient_nonzero": True,
        "reference_frozen": True, "hydraulic_horizon_minutes": 120,
        "long_horizon_value_minutes": 360, "finite_hop_cutoff": False,
        "oracle_future_truth_input": False, "online_all_link_flow_input": False,
    }


def _primary(metrics: dict[str, Any]) -> dict[str, float]:
    overall = metrics["overall"]
    return {
        "depth": float(overall.get("depth_m_skill_vs_zero", float("nan"))),
        "flood": float(overall.get("flood_m3s_skill_vs_zero", float("nan"))),
        "volume": float(overall.get("volume_m3_skill_vs_zero", float("nan"))),
        "managed_flow": float(overall.get("managed_flow_m3s_skill_vs_zero", float("nan"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Project7 Step2 V11 Train-only development")
    parser.add_argument("--stage", choices=("d2", "d3"), required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--v70-value-checkpoint", required=True)
    parser.add_argument("--v70-hydraulic-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--d2-checkpoint")
    parser.add_argument("--d2-report")
    args = parser.parse_args()

    contract = ActuatorSetHydraulicContractV110()
    contract.validate()
    graph = _load_graph(args.graph)
    if len(graph.node_ids) != 932 or len(graph.actuator_ids) != 109:
        raise ValueError("V11 requires frozen Project7 932-node/109-actuator graph")
    cache = V60TrainCache(args.cache_manifest)
    validate_v60_cache_lineage(args.cache_manifest)
    if cache.legacy_d3_names():
        raise ValueError("V11 refuses legacy dense D3")
    d2 = cache.names("D2")
    d3 = cache.targeted_d3_names()
    selected = sorted(d2 + d3)
    fit, holdout = deterministic_rainfall_split_v60(cache, names=selected,
                                                     holdout_fraction=float(args.holdout_fraction))
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    holdout_d2 = [name for name in holdout if name.startswith("D2::")]
    fit_d3 = [name for name in fit if name.startswith("D3::")]
    holdout_d3 = [name for name in holdout if name.startswith("D3::")]
    if not all((fit_d2, holdout_d2, fit_d3, holdout_d3)):
        raise ValueError("V11 split requires D2/D3 on TrainFit/InternalHoldout")
    fit_events = {cache.entry(name).event_id for name in fit}
    holdout_events = {cache.entry(name).event_id for name in holdout}
    fit_rain = {cache.entry(name).rainfall_group for name in fit}
    holdout_rain = {cache.entry(name).rainfall_group for name in holdout}
    if fit_events & holdout_events or fit_rain & holdout_rain:
        raise ValueError("V11 event/rainfall leakage detected")

    value_checkpoint = _load_checkpoint(args.v70_value_checkpoint, "control_value")
    hydraulic_checkpoint = _load_checkpoint(args.v70_hydraulic_checkpoint, "hydraulic_response")
    if value_checkpoint.get("split_manifest_sha256") != hydraulic_checkpoint.get("split_manifest_sha256"):
        raise ValueError("V11 V7 Value/Hydraulic split lineage mismatch")

    normalization = derive_input_normalization_v60(cache, fit)
    prepared_cpu = prepare_static_v60(graph, "cpu")
    effect_scales = derive_effect_scales_v110(cache, fit_d2, prepared_cpu)
    reference = _build_reference(graph, cache, fit, hydraulic_checkpoint)
    relations = build_actuator_node_relations_v110(graph)
    entry = cache.entry(fit_d2[0])
    state_dim = int(entry.arrays["initial_state"].shape[-1])
    rainfall_dim = int(entry.arrays["rainfall"].shape[-1])
    if state_dim != 6:
        raise ValueError("V11 requires six-state node contract")
    model = ActuatorSetHydraulicResponseV110(
        reference_model=reference,
        state_magnitude_scale=effect_scales.state_magnitude_scale,
        flow_magnitude_scale=effect_scales.flow_magnitude_scale,
        node_static_dim=int(graph.static_node_features.shape[1]),
        physics_dim=int(prepared_cpu.actuator_physics.shape[1]), rainfall_dim=rainfall_dim,
        actuator_count=len(graph.actuator_ids), node_count=len(graph.node_ids),
        relations=relations, horizon=HydraulicHorizonV110(), contract=contract,
    )

    if args.stage == "d3":
        if not args.d2_checkpoint or not args.d2_report:
            raise ValueError("V11 D3 requires --d2-checkpoint and --d2-report")
        d2_report = json.loads(Path(args.d2_report).read_text(encoding="utf-8"))
        if d2_report.get("contract") != V110_CONTRACT or not d2_report.get("d3_authorized", False):
            raise ValueError("V11 D3 refused: D2 mechanism report is not accepted")
        d2_payload = torch.load(args.d2_checkpoint, map_location="cpu", weights_only=False)
        if d2_payload.get("contract") != V110_CONTRACT or d2_payload.get("stage") != "d2":
            raise ValueError("V11 D3 checkpoint is not an accepted V11 D2 checkpoint")
        model.load_state_dict(d2_payload["state_dict"], strict=True)

    target = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    prepared = prepare_static_v60(graph, target)
    preflight = _preflight(model, cache, fit_d2[0] if args.stage == "d2" else fit_d3[0],
                           normalization, prepared, target)
    before_names = holdout_d2 if args.stage == "d2" else holdout_d3
    before = evaluate_hydraulic_effect_v110(model, cache, before_names, normalization,
                                             prepared, effect_scales, device=target)
    if args.stage == "d2":
        history = train_d2_mechanism_v110(model, cache, fit_d2, normalization, prepared,
                                           effect_scales, device=target, contract=contract)
        fit_names, holdout_names = fit_d2, holdout_d2
    else:
        history = train_d3_interaction_v110(model, cache, fit_d3, fit_d2, normalization,
                                             prepared, effect_scales, device=target, contract=contract)
        fit_names, holdout_names = fit_d3, holdout_d3

    fit_metrics = evaluate_hydraulic_effect_v110(model, cache, fit_names, normalization,
                                                  prepared, effect_scales, device=target)
    holdout_metrics = evaluate_hydraulic_effect_v110(model, cache, holdout_names, normalization,
                                                      prepared, effect_scales, device=target)
    primary = _primary(holdout_metrics)
    d2_supported = args.stage == "d2" and all(np.isfinite(value) and value > 0.0 for value in primary.values())
    decision = (
        "V110_D2_MECHANISM_SUPPORTED" if d2_supported else
        "V110_D2_MECHANISM_NOT_YET_SUPPORTED" if args.stage == "d2" else
        "V110_D3_INTERACTION_DEVELOPMENT_COMPLETE"
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lineage = {
        "git_head": _git_head(), "graph_sha256": _sha256(args.graph),
        "cache_manifest_sha256": _sha256(args.cache_manifest),
        "v70_value_checkpoint_sha256": _sha256(args.v70_value_checkpoint),
        "v70_hydraulic_checkpoint_sha256": _sha256(args.v70_hydraulic_checkpoint),
        "stage": args.stage, "seed": contract.seed, "fit_group_count": len(fit_names),
        "holdout_group_count": len(holdout_names), "event_overlap": 0, "rainfall_overlap": 0,
    }
    checkpoint_path = out / f"step2_v110_{args.stage}_development.pt"
    torch.save({"contract": V110_CONTRACT, "stage": args.stage, "state_dict": model.state_dict(),
                "lineage": lineage, "production_compatible": False}, checkpoint_path)
    report = {
        "contract": V110_CONTRACT, "stage": args.stage, "development_only": True,
        "boundary": {
            "swmm_run": False, "validation_accessed": False, "final_accessed": False,
            "formal_accessed": False, "production_wiring_modified": False,
            "v7_value_retrained": False, "hydraulic_horizon_minutes": 120,
            "long_horizon_value_minutes": 360,
        },
        "architecture": {
            "variable_length_changed_actuator_set": True,
            "action_prefix_to_response_time": True, "future_action_causality": True,
            "actuator_self_attention": True, "global_node_cross_attention": True,
            "finite_hop_cutoff": False, "active_sign_magnitude_heads": True,
            "active_threshold": "TrainFit node/actuator-local P90x0.25 with physical floors",
            "recurrent_free_run": False,
        },
        "lineage": lineage, "preflight": preflight, "before_holdout": before,
        "training_history": history, "fit_metrics": fit_metrics,
        "holdout_metrics": holdout_metrics, "holdout_primary_skill": primary,
        "decision": decision, "d3_authorized": bool(d2_supported),
        "formal_authorized": False, "production_authorized": False,
        "new_swmm_authorized": False, "checkpoint": str(checkpoint_path),
    }
    report_path = out / f"STEP2_V110_{args.stage.upper()}_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "decision": decision, "primary": primary}, indent=2))


if __name__ == "__main__":
    main()
