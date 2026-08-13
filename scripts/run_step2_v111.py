"""Train-only V11.1 development runner.

Profiles are deliberately finite and pre-registered: ``tiny`` repeats one
TrainFit group, ``micro`` uses a deterministic 12-group TrainFit subset, and
``canonical`` uses all 112 TrainFit D2 groups.  No profile reads Internal
Holdout labels during training and none invokes SWMM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time
from typing import Any

import numpy as np
import torch

from rtc.step2_control_response_v111 import ActuatorSetHydraulicResponseV111
from rtc.step2_control_response_v110 import build_actuator_node_relations_v110
from rtc.step2_control_response_v60 import prepare_static_v60
from rtc.step2_hydraulic_eval_v111 import evaluate_arrays_v111
from rtc.step2_hydraulic_objective_v111 import derive_effect_scales_v111
from rtc.step2_optimization_v111 import backward_group_v111, train_d2_v111
from rtc.step2_shards_v60 import validate_v60_cache_lineage
from rtc.step2_train_response_v60 import (
    V60TrainCache,
    derive_input_normalization_v60,
    deterministic_rainfall_split_v60,
)
from rtc.step2_v110_contract import HydraulicHorizonV110
from rtc.step2_v111_contract import V111LossContract

from run_step2_v110 import _build_reference, _load_checkpoint, _load_graph, _sha256


def _git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _select_micro(cache: V60TrainCache, fit_names: list[str]) -> list[str]:
    """Deterministic mechanism coverage selection without outcome inspection."""
    ordered = sorted(fit_names)
    by_type: dict[str, list[str]] = {"pump": [], "orifice": [], "weir": []}
    for name in ordered:
        entry = cache.entry(name)
        ref = entry.reference_index
        candidates = [i for i in entry.indices if i != ref]
        delta = np.asarray(entry.arrays["settings"][candidates[0]]) - np.asarray(entry.arrays["settings"][ref])
        changed = np.flatnonzero(np.any(np.abs(delta) > 1.0e-7, axis=0))
        if changed.size:
            # The graph actuator catalog is not in the cache; source metadata
            # retains the actuator type in candidate_roles when available.
            role = str(entry.arrays.get("actuator_type", ["unknown"])[ref]).lower()
            if role not in by_type:
                role = ("pump" if int(changed[0]) < 57 else "orifice" if int(changed[0]) < 99 else "weir")
            by_type[role].append(name)
    selected: list[str] = []
    for values in by_type.values():
        selected.extend(values[:3])
    for name in ordered:
        if name not in selected:
            selected.append(name)
        if len(selected) >= 12:
            break
    return sorted(dict.fromkeys(selected))[:12]


def _build_model(graph, cache, fit_names, normalization, hydraulic_checkpoint, prepared_cpu, scales):
    reference = _build_reference(graph, cache, fit_names, hydraulic_checkpoint)
    entry = cache.entry(fit_names[0])
    return ActuatorSetHydraulicResponseV111(
        reference_model=reference,
        state_magnitude_scale=scales.state_magnitude_scale,
        flow_magnitude_scale=scales.flow_magnitude_scale,
        node_static_dim=int(graph.static_node_features.shape[1]),
        physics_dim=int(prepared_cpu.actuator_physics.shape[1]),
        rainfall_dim=int(entry.arrays["rainfall"].shape[-1]),
        actuator_count=len(graph.actuator_ids), node_count=len(graph.node_ids),
        relations=build_actuator_node_relations_v110(graph),
        horizon=HydraulicHorizonV110(),
    )


def _collect(model, cache, names, normalization, prepared, scales, device):
    states_pred, states_true, flows_pred, flows_true, event_ids = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for name in names:
            batch = cache.batch(name, normalization, device)
            output = model(batch.initial_state, batch.rainfall, batch.reference_settings,
                           batch.candidate_settings, batch.previous_actuator_flow, prepared)
            idx = output.horizon_indices
            truth_state = (batch.true_candidate_states.index_select(2, idx)
                           - batch.true_reference_states.index_select(1, idx)[:, None])[..., [0, 2, 3, 4, 5]]
            truth_flow = (batch.true_candidate_flows.index_select(2, idx)
                          - batch.true_reference_flows.index_select(1, idx)[:, None])
            pred_state = output.raw_delta_states_physical[..., [0, 2, 3, 4, 5]]
            pred_flow = output.raw_delta_flows_physical
            states_pred.append(pred_state[0].detach().cpu().numpy())
            states_true.append(truth_state[0].detach().cpu().numpy())
            flows_pred.append(pred_flow[0].detach().cpu().numpy())
            flows_true.append(truth_flow[0].detach().cpu().numpy())
            event_ids.extend([cache.entry(name).event_id] * int(pred_state.shape[1]))
    return evaluate_arrays_v111(
        event_ids=event_ids,
        pred_state=np.concatenate(states_pred, axis=0), true_state=np.concatenate(states_true, axis=0),
        pred_flow=np.concatenate(flows_pred, axis=0), true_flow=np.concatenate(flows_true, axis=0), scales=scales,
    )


def _preflight(model, cache, name, normalization, prepared, device):
    batch = cache.batch(name, normalization, device)
    model.eval()
    reference = batch.reference_settings[:, None]
    with torch.no_grad():
        zero = model(batch.initial_state, batch.rainfall, batch.reference_settings, reference,
                     batch.previous_actuator_flow, prepared)
    if not torch.equal(zero.raw_delta_states_physical, torch.zeros_like(zero.raw_delta_states_physical)):
        raise RuntimeError("V111 exact-zero state preflight failed")
    if not torch.equal(zero.raw_delta_flows_physical, torch.zeros_like(zero.raw_delta_flows_physical)):
        raise RuntimeError("V111 exact-zero flow preflight failed")
    future = reference.clone()
    future[:, :, 18:, 0] = 0.25
    with torch.no_grad():
        delayed = model(batch.initial_state, batch.rainfall, batch.reference_settings, future,
                        batch.previous_actuator_flow, prepared)
    early = delayed.horizon_indices < 18
    if not torch.equal(delayed.raw_delta_states_physical[:, :, early], torch.zeros_like(delayed.raw_delta_states_physical[:, :, early])):
        raise RuntimeError("V111 future action leaked into earlier state output")
    if not torch.equal(delayed.raw_delta_flows_physical[:, :, early], torch.zeros_like(delayed.raw_delta_flows_physical[:, :, early])):
        raise RuntimeError("V111 future action leaked into earlier flow output")
    candidate = batch.candidate_settings[:, :1].detach().clone().requires_grad_(True)
    output = model(batch.initial_state, batch.rainfall, batch.reference_settings, candidate,
                   batch.previous_actuator_flow, prepared)
    score = output.raw_delta_states_physical.square().mean() + output.raw_delta_flows_physical.square().mean()
    grad = torch.autograd.grad(score, candidate)[0]
    if not bool(torch.isfinite(grad).all()) or int(torch.count_nonzero(grad)) == 0:
        raise RuntimeError("V111 action gradient is non-finite or zero")
    return {"exact_zero": True, "future_action_causality": True,
            "action_gradient_finite": True, "action_gradient_nonzero": True,
            "reference_frozen": not any(p.requires_grad for p in model.reference_model.parameters()),
            "no_swmm": True, "validation_accessed": False, "final_accessed": False, "formal_accessed": False}


def _tiny(model, cache, name, normalization, prepared, scales, device, steps=100):
    contract = V111LossContract()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=contract.learning_rate,
                                   weight_decay=contract.weight_decay)
    history = []
    batch = cache.batch(name, normalization, device)
    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        metrics = backward_group_v111(model, batch, prepared, scales, loss_contract=contract)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], contract.grad_clip)
        optimizer.step()
        if step == 1 or step % 5 == 0 or step == steps:
            row = {"step": step, **metrics}
            history.append(row)
            print(f"[V111_TINY] step={step} loss={metrics['loss']:.6g} direct={metrics['direct_signed']:.6g} "
                  f"active={metrics['direct_active']:.6g} inactive={metrics['direct_inactive']:.6g}", flush=True)
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("tiny", "micro", "canonical"), required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--v70-hydraulic-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    start = time.time()
    contract = V111LossContract(); contract.validate()
    torch.manual_seed(contract.seed); np.random.seed(contract.seed); random.seed(contract.seed)
    graph = _load_graph(args.graph)
    cache = V60TrainCache(args.cache_manifest)
    validate_v60_cache_lineage(args.cache_manifest)
    if cache.legacy_d3_names():
        raise ValueError("V111 refuses legacy dense D3")
    d2 = cache.names("D2")
    fit_d2, holdout_d2 = deterministic_rainfall_split_v60(cache, names=d2, holdout_fraction=.20)
    if len(fit_d2) != 112 or len(holdout_d2) != 32:
        raise ValueError(f"unexpected V111 D2 split: {len(fit_d2)}/{len(holdout_d2)}")
    normalization = derive_input_normalization_v60(cache, fit_d2)
    prepared_cpu = prepare_static_v60(graph, "cpu")
    scales = derive_effect_scales_v111(cache, fit_d2, prepared_cpu)
    hydraulic_checkpoint = _load_checkpoint(args.v70_hydraulic_checkpoint, "hydraulic_response")
    target = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    prepared = prepare_static_v60(graph, target)
    if args.profile == "tiny":
        names = [sorted(fit_d2)[0]]
    elif args.profile == "micro":
        names = _select_micro(cache, fit_d2)
    else:
        names = list(fit_d2)
    model = _build_model(graph, cache, fit_d2, normalization, hydraulic_checkpoint, prepared_cpu, scales).to(target)
    preflight = _preflight(model, cache, names[0], normalization, prepared, target)
    if args.profile == "tiny":
        history = _tiny(model, cache, names[0], normalization, prepared, scales, target, steps=contract.tiny_steps)
    elif args.profile == "micro":
        history = train_d2_v111(model, cache, names, normalization, prepared, scales, device=target, epochs=4)
    else:
        history = train_d2_v111(model, cache, names, normalization, prepared, scales, device=target, epochs=contract.canonical_max_epochs)
    fit_metrics = _collect(model, cache, names, normalization, prepared, scales, target)
    holdout_metrics = None if args.profile != "canonical" else _collect(model, cache, holdout_d2, normalization, prepared, scales, target)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    report = {
        "contract": "PROJECT7_STEP2_V111_ZERO_ANCHORED_SIGNED_EFFECT_V1",
        "profile": args.profile, "development_only": True,
        "boundary": {"swmm_run": False, "validation_accessed": False, "final_accessed": False,
                      "formal_accessed": False, "production_wiring_modified": False,
                      "v7_value_retrained": False, "legacy_dense_d3_present": False},
        "lineage": {"git_head": _git_head(), "graph_sha256": _sha256(args.graph),
                    "cache_manifest_sha256": _sha256(args.cache_manifest),
                    "v70_hydraulic_checkpoint_sha256": _sha256(args.v70_hydraulic_checkpoint),
                    "fit_group_count": len(fit_d2), "holdout_group_count": len(holdout_d2),
                    "profile_group_count": len(names), "seed": 42},
        "split": {"fit_d2": len(fit_d2), "holdout_d2": len(holdout_d2),
                  "event_overlap": 0, "rainfall_overlap": 0, "profile_groups": names},
        "preflight": preflight, "scale_evidence": {
            "state_fallback_fraction": scales.state_fallback_fraction,
            "flow_fallback_fraction": scales.flow_fallback_fraction,
            "state_local_active_support_min": int(scales.state_local_active_support.min()),
            "flow_local_active_support_min": int(scales.flow_local_active_support.min()),
            "state_active_quantiles": scales.state_active_quantiles,
            "flow_active_quantiles": scales.flow_active_quantiles,
        },
        "training_history": history, "fit_metrics": fit_metrics, "holdout_metrics": holdout_metrics,
        "elapsed_seconds": time.time() - start,
        "production_compatible": False,
    }
    path = out / f"STEP2_V111_{args.profile.upper()}_REPORT.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(path), "profile": args.profile, "fit": fit_metrics.get("primary_event_pooled_skill_vs_zero"),
                      "holdout": None if holdout_metrics is None else holdout_metrics.get("primary_event_pooled_skill_vs_zero")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
