"""Robust Stage-A curriculum for the counterfactual-first Project7 V128 candidate.

This module intentionally sits above ``step2_counterfactual_first_v128``: the latter defines the
new model/scales and causal pair extraction, while this file owns the training curriculum.  It
makes two contracts explicit:

* `_candidate_permutation` is one-based because it normally indexes branches after reference;
  direct-pair lists are zero-based, so an explicit adapter is mandatory;
* sparse direct hydraulic effects are response-weighted after physical residual scaling instead
  of being diluted by a uniform mean over ~932 nodes.

No SWMM gradient labels are trained.  The curriculum is trajectory/counterfactual-first:
A0 direct setting->flow, A1 oracle managed-flow->hydraulics, A2 joint predicted-flow transition.
"""
from __future__ import annotations

import gc
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_action_identifiable_v128 import _response_weighted_effect_loss, _teacher_step
from .step2_counterfactual_first_v128 import (
    _DIRECT_EFFECT_FLOOR,
    _cpu_direct_specs,
    _direct_effect_loss,
    _oracle_transition_prediction,
    _predict_direct_pair,
)
from .step2_train_response_v60 import InputNormalizationV60
from .step2_train_v127 import _ordered, _static
from .step2_train_v127_control import V127ControlTrainingDesign, _candidate_permutation, _state_weights
from .step2_train_v127_streaming import (
    _cpu_group,
    _cuda_peak,
    _reset_cuda_peak,
    _select_to_device,
)

COUNTERFACTUAL_STAGE_A_V3_CONTRACT = (
    "PROJECT7_V128_COUNTERFACTUAL_FIRST_STAGE_A_V3_DIRECT_ORACLE_JOINT"
)
DIRECT_FLOW_A0_V3_CONTRACT = "PROJECT7_V128_DIRECT_SAME_PREFIX_FLOW_A0_V3"
ORACLE_HYDRAULIC_A1_V3_CONTRACT = "PROJECT7_V128_ORACLE_FLOW_HYDRAULIC_A1_V3"
JOINT_DIRECT_A2_V3_CONTRACT = "PROJECT7_V128_JOINT_DIRECT_HYDRAULIC_A2_V3"


def _zero_based_spec_order(count: int, *, group_name: str, epoch: int, seed: int) -> np.ndarray:
    """Adapt the canonical one-based candidate permutation to a zero-based spec list."""
    values = _candidate_permutation(count, group_name=group_name, epoch=epoch, seed=seed)
    if values.size == 0:
        return values
    result = values - 1
    if int(result.min()) < 0 or int(result.max()) >= count:
        raise RuntimeError("direct-pair permutation escaped zero-based spec range")
    return result


def _direct_state_effect_loss(
    model: Any,
    *,
    predicted_delta: torch.Tensor,
    true_delta: torch.Tensor,
    state_weights: torch.Tensor,
) -> torch.Tensor:
    return _response_weighted_effect_loss(
        predicted_delta * state_weights,
        true_delta * state_weights,
        model.transition.delta_state_scale,
    )


def train_direct_action_flow_a0_v3(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    design: V127ControlTrainingDesign,
) -> dict[str, Any]:
    """A0: absolute q fit plus direct same-prefix magnitude+direction supervision."""
    design.validate()
    torch.manual_seed(design.seed + 3201)
    np.random.seed(design.seed + 3201)
    model.train().to(device)
    static = _static(graph, device)
    optimizer = torch.optim.AdamW(
        model.actuator.parameters(), lr=float(design.learning_rate), weight_decay=float(design.weight_decay)
    )
    absolute_reports: list[float] = []
    effect_reports: list[float] = []
    magnitude_reports: list[float] = []
    direction_reports: list[float] = []
    selected_pairs = 0
    direct_nonzero = 0
    _reset_cuda_peak(device)

    for source, name in _ordered(source_groups, 3201, design.seed):
        cpu = _cpu_group(source_caches[source], name, normalization)
        branches, horizon = cpu["settings"].shape[:2]
        optimizer.zero_grad(set_to_none=True)
        absolute_group = 0.0
        for start in range(0, branches, design.hydraulic_branch_chunk):
            stop = min(start + design.hydraulic_branch_chunk, branches)
            positions = np.arange(start, stop, dtype=np.int64)
            chunk = _select_to_device(cpu, positions, device=device, include_truth=True)
            physics_norm, identity = model.actuator.prepare_static(static["physics"], batch_size=len(positions))
            term = torch.zeros((), device=device)
            for k in range(horizon):
                prev_state = chunk["initial"] if k == 0 else chunk["states"][:, k - 1]
                prev_flow = chunk["previous_flow"] if k == 0 else chunk["flows"][:, k - 1]
                q, _ = model.actuator.forward_prepared(
                    prev_state[:, static["up"]], prev_state[:, static["down"]],
                    chunk["settings"][:, k], prev_flow, physics_norm, identity
                )
                error = (q - chunk["flows"][:, k]) / model.actuator.flow_std
                term = term + F.smooth_l1_loss(error, torch.zeros_like(error), beta=0.5)
            weighted = (term / float(horizon)) * (float(len(positions)) / float(branches))
            if not bool(torch.isfinite(weighted)):
                raise RuntimeError(f"{name}: A0 absolute-flow loss is non-finite")
            weighted.backward()
            absolute_group += float(weighted.detach())
            del chunk, term, weighted

        specs = _cpu_direct_specs(cpu)
        budget = min(8 if str(source).upper() == "D2" else 4, len(specs))
        if budget:
            order = _zero_based_spec_order(
                len(specs), group_name=name, epoch=3201, seed=design.seed
            )[:budget]
            losses: list[torch.Tensor] = []
            magnitudes: list[torch.Tensor] = []
            directions: list[torch.Tensor] = []
            for offset in order.tolist():
                spec = specs[int(offset)]
                q, target, _, _ = _predict_direct_pair(
                    model, cpu=cpu, spec=spec, static=static, device=device
                )
                actuator = int(spec["actuator_index"])
                true_delta = target[1] - target[0]
                loss, magnitude, direction = _direct_effect_loss(
                    q[1] - q[0], true_delta, model.actuator.direct_action_flow_scale[actuator]
                )
                losses.append(loss)
                magnitudes.append(magnitude)
                directions.append(direction)
                selected_pairs += 1
                direct_nonzero += int(abs(float(true_delta.detach().cpu())) > _DIRECT_EFFECT_FLOOR)
            effect = torch.stack(losses).mean()
            if not bool(torch.isfinite(effect)):
                raise RuntimeError(f"{name}: A0 direct-flow effect loss is non-finite")
            effect.backward()
            effect_reports.append(float(effect.detach()))
            magnitude_reports.append(float(torch.stack(magnitudes).mean().detach()))
            direction_reports.append(float(torch.stack(directions).mean().detach()))
        torch.nn.utils.clip_grad_norm_(model.actuator.parameters(), design.grad_clip)
        optimizer.step()
        absolute_reports.append(absolute_group)
        del cpu

    row = {
        "stage": "v128_direct_same_prefix_action_flow_a0",
        "contract": DIRECT_FLOW_A0_V3_CONTRACT,
        "absolute_flow_loss": float(np.mean(absolute_reports)) if absolute_reports else 0.0,
        "direct_flow_effect_loss": float(np.mean(effect_reports)) if effect_reports else 0.0,
        "direct_flow_magnitude_loss": float(np.mean(magnitude_reports)) if magnitude_reports else 0.0,
        "direct_flow_direction_loss": float(np.mean(direction_reports)) if direction_reports else 0.0,
        "selected_direct_counterfactual_pairs": int(selected_pairs),
        "nonzero_direct_counterfactual_pairs": int(direct_nonzero),
        "common_authoritative_prefix_used": True,
        "full_horizon_feedback_used_as_direct_label": False,
        "gradient_label_used": False,
        "fit_only": True,
        **_cuda_peak(device),
    }
    print("[V128_COUNTERFACTUAL_A0] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    gc.collect()
    return row


def train_oracle_hydraulic_a1_v3(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    depth_index: int,
    flood_rate_index: int,
    design: V127ControlTrainingDesign,
) -> dict[str, Any]:
    """A1: isolate managed-flow -> next-state dynamics before joint predicted-flow training."""
    design.validate()
    model.train().to(device)
    static = _static(graph, device)
    state_weights = _state_weights(
        model, depth_index=depth_index, flood_rate_index=flood_rate_index, design=design, device=device
    )
    actuator_flags = [p.requires_grad for p in model.actuator.parameters()]
    for parameter in model.actuator.parameters():
        parameter.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(design.learning_rate), weight_decay=float(design.weight_decay)
    )
    absolute_reports: list[float] = []
    direct_reports: list[float] = []
    selected_pairs = 0
    _reset_cuda_peak(device)
    try:
        for source, name in _ordered(source_groups, 3301, design.seed):
            cpu = _cpu_group(source_caches[source], name, normalization)
            branches, horizon = cpu["settings"].shape[:2]
            optimizer.zero_grad(set_to_none=True)
            absolute_group = 0.0
            for start in range(0, branches, design.hydraulic_branch_chunk):
                stop = min(start + design.hydraulic_branch_chunk, branches)
                positions = np.arange(start, stop, dtype=np.int64)
                chunk = _select_to_device(cpu, positions, device=device, include_truth=True)
                term = torch.zeros((), device=device)
                steps = len(range(0, horizon, design.teacher_stride))
                for k in range(0, horizon, design.teacher_stride):
                    prev_state = chunk["initial"] if k == 0 else chunk["states"][:, k - 1]
                    prev_flow = chunk["previous_flow"] if k == 0 else chunk["flows"][:, k - 1]
                    pred = _oracle_transition_prediction(
                        model,
                        prev_state=prev_state,
                        previous_flow=prev_flow,
                        setting=chunk["settings"][:, k],
                        oracle_flow=chunk["flows"][:, k],
                        rainfall=chunk["rainfall"][:, k],
                        static=static,
                    )
                    error = (pred - chunk["states"][:, k]) / model.transition.state_std
                    term = term + F.smooth_l1_loss(
                        error * state_weights, torch.zeros_like(error), beta=0.5
                    )
                weighted = (term / float(steps)) * (float(len(positions)) / float(branches))
                if not bool(torch.isfinite(weighted)):
                    raise RuntimeError(f"{name}: A1 oracle-flow absolute loss is non-finite")
                weighted.backward()
                absolute_group += float(weighted.detach())
                del chunk, term, weighted

            specs = _cpu_direct_specs(cpu)
            budget = min(4, len(specs))
            if budget:
                order = _zero_based_spec_order(
                    len(specs), group_name=name, epoch=3301, seed=design.seed
                )[:budget]
                losses: list[torch.Tensor] = []
                for offset in order.tolist():
                    spec = specs[int(offset)]
                    k = int(spec["step"])
                    candidate = int(spec["candidate_position"])
                    if k == 0:
                        prefix_state = cpu["initial"][0]
                        prefix_flow = cpu["previous_flow"][0]
                    else:
                        prefix_state = cpu["states"][0, k - 1]
                        prefix_flow = cpu["flows"][0, k - 1]
                    prev_state = prefix_state[None].repeat(2, 1, 1).to(device)
                    prev_flow = prefix_flow[None].repeat(2, 1).to(device)
                    setting = torch.stack((cpu["settings"][0, k], cpu["settings"][candidate, k])).to(device)
                    oracle_flow = torch.stack((cpu["flows"][0, k], cpu["flows"][candidate, k])).to(device)
                    rainfall = cpu["rainfall"][0, k][None].repeat(2, 1).to(device)
                    pred = _oracle_transition_prediction(
                        model, prev_state=prev_state, previous_flow=prev_flow, setting=setting,
                        oracle_flow=oracle_flow, rainfall=rainfall, static=static
                    )
                    truth = torch.stack((cpu["states"][0, k], cpu["states"][candidate, k])).to(device)
                    losses.append(
                        _direct_state_effect_loss(
                            model,
                            predicted_delta=pred[1] - pred[0],
                            true_delta=truth[1] - truth[0],
                            state_weights=state_weights,
                        )
                    )
                    selected_pairs += 1
                direct = torch.stack(losses).mean()
                if not bool(torch.isfinite(direct)):
                    raise RuntimeError(f"{name}: A1 direct next-state effect loss is non-finite")
                direct.backward()
                direct_reports.append(float(direct.detach()))
            torch.nn.utils.clip_grad_norm_(trainable, design.grad_clip)
            optimizer.step()
            absolute_reports.append(absolute_group)
            del cpu
    finally:
        for parameter, flag in zip(model.actuator.parameters(), actuator_flags, strict=True):
            parameter.requires_grad_(flag)

    row = {
        "stage": "v128_oracle_managed_flow_hydraulic_a1",
        "contract": ORACLE_HYDRAULIC_A1_V3_CONTRACT,
        "absolute_state_loss": float(np.mean(absolute_reports)) if absolute_reports else 0.0,
        "direct_next_state_effect_loss": float(np.mean(direct_reports)) if direct_reports else 0.0,
        "selected_direct_counterfactual_pairs": int(selected_pairs),
        "authoritative_managed_flow_injection": True,
        "actuator_parameters_frozen": True,
        "response_weighted_direct_state_effect": True,
        "gradient_label_used": False,
        "fit_only": True,
        **_cuda_peak(device),
    }
    print("[V128_COUNTERFACTUAL_A1] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    gc.collect()
    return row


def train_joint_direct_a2_v3(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    depth_index: int,
    flood_rate_index: int,
    design: V127ControlTrainingDesign,
) -> list[dict[str, Any]]:
    """A2: joint one-step fit plus same-prefix direct flow/state response."""
    design.validate()
    model.train().to(device)
    static = _static(graph, device)
    state_weights = _state_weights(
        model, depth_index=depth_index, flood_rate_index=flood_rate_index, design=design, device=device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=design.learning_rate, weight_decay=design.weight_decay)
    history: list[dict[str, Any]] = []
    _reset_cuda_peak(device)

    for epoch in range(1, design.hydraulic_epochs + 1):
        phase = (epoch - 1) % design.teacher_stride
        absolute_reports: list[float] = []
        flow_mag_reports: list[float] = []
        flow_dir_reports: list[float] = []
        state_reports: list[float] = []
        selected_pairs = 0
        for source, name in _ordered(source_groups, epoch + 3400, design.seed):
            cpu = _cpu_group(source_caches[source], name, normalization)
            branches, horizon = cpu["settings"].shape[:2]
            optimizer.zero_grad(set_to_none=True)
            absolute_group = 0.0
            steps = len(range(phase, horizon, design.teacher_stride))
            for start in range(0, branches, design.hydraulic_branch_chunk):
                stop = min(start + design.hydraulic_branch_chunk, branches)
                positions = np.arange(start, stop, dtype=np.int64)
                chunk = _select_to_device(cpu, positions, device=device, include_truth=True)
                physics_norm, identity = model.actuator.prepare_static(static["physics"], batch_size=len(positions))
                static_norm, edges, inv = model.transition.prepare_static(
                    static["static"], static["edges"], batch_size=len(positions), dtype=torch.float32
                )
                term = torch.zeros((), device=device)
                for k in range(phase, horizon, design.teacher_stride):
                    pred_state, q = _teacher_step(
                        model, chunk=chunk, k=k, static=static, physics_norm=physics_norm,
                        identity=identity, static_norm=static_norm, edges=edges, inv=inv
                    )
                    state_error = (pred_state - chunk["states"][:, k]) / model.transition.state_std
                    flow_error = (q - chunk["flows"][:, k]) / model.actuator.flow_std
                    term = term + F.smooth_l1_loss(
                        state_error * state_weights, torch.zeros_like(state_error), beta=0.5
                    ) + design.flow_weight * F.smooth_l1_loss(
                        flow_error, torch.zeros_like(flow_error), beta=0.5
                    )
                weighted = (term / float(steps)) * (float(len(positions)) / float(branches))
                if not bool(torch.isfinite(weighted)):
                    raise RuntimeError(f"{name}: A2 absolute joint loss is non-finite")
                weighted.backward()
                absolute_group += float(weighted.detach())
                del chunk, term, weighted

            specs = _cpu_direct_specs(cpu)
            budget = min(4, len(specs))
            if budget:
                order = _zero_based_spec_order(
                    len(specs), group_name=name, epoch=epoch + 3400, seed=design.seed
                )[:budget]
                losses: list[torch.Tensor] = []
                flow_mags: list[torch.Tensor] = []
                flow_dirs: list[torch.Tensor] = []
                state_losses: list[torch.Tensor] = []
                for offset in order.tolist():
                    spec = specs[int(offset)]
                    k = int(spec["step"])
                    candidate = int(spec["candidate_position"])
                    if k == 0:
                        prefix_state = cpu["initial"][0]
                        prefix_flow = cpu["previous_flow"][0]
                    else:
                        prefix_state = cpu["states"][0, k - 1]
                        prefix_flow = cpu["flows"][0, k - 1]
                    prev_state = prefix_state[None].repeat(2, 1, 1).to(device)
                    prev_flow = prefix_flow[None].repeat(2, 1).to(device)
                    setting = torch.stack((cpu["settings"][0, k], cpu["settings"][candidate, k])).to(device)
                    rainfall = cpu["rainfall"][0, k][None].repeat(2, 1).to(device)
                    physics_norm, identity = model.actuator.prepare_static(static["physics"], batch_size=2)
                    static_norm, edges, inv = model.transition.prepare_static(
                        static["static"], static["edges"], batch_size=2, dtype=torch.float32
                    )
                    q, response = model.actuator.forward_prepared(
                        prev_state[:, static["up"]], prev_state[:, static["down"]],
                        setting, prev_flow, physics_norm, identity
                    )
                    injection = torch.zeros(2, prev_state.shape[1], 1, device=device, dtype=prev_state.dtype)
                    injection = injection.index_add(1, static["up"], -q[..., None]).index_add(
                        1, static["down"], q[..., None]
                    )
                    action_context = model._typed_action_context(
                        state=prev_state, setting=setting, previous_flow=prev_flow, predicted_flow=q,
                        responsiveness=response, upstream=static["up"], downstream=static["down"],
                        physics_norm=physics_norm, identity_embedding=identity
                    )
                    pred_state = model.transition.forward_prepared(
                        prev_state, rainfall, static_norm, injection, edges, inv, action_context
                    )
                    target_flow = torch.stack((cpu["flows"][0, k], cpu["flows"][candidate, k])).to(device)
                    target_state = torch.stack((cpu["states"][0, k], cpu["states"][candidate, k])).to(device)
                    actuator = int(spec["actuator_index"])
                    flow_loss, flow_mag, flow_dir = _direct_effect_loss(
                        q[1, actuator] - q[0, actuator],
                        target_flow[1, actuator] - target_flow[0, actuator],
                        model.actuator.direct_action_flow_scale[actuator],
                    )
                    state_loss = _direct_state_effect_loss(
                        model,
                        predicted_delta=pred_state[1] - pred_state[0],
                        true_delta=target_state[1] - target_state[0],
                        state_weights=state_weights,
                    )
                    losses.append(flow_loss + state_loss)
                    flow_mags.append(flow_mag)
                    flow_dirs.append(flow_dir)
                    state_losses.append(state_loss)
                    selected_pairs += 1
                direct = torch.stack(losses).mean()
                if not bool(torch.isfinite(direct)):
                    raise RuntimeError(f"{name}: A2 direct counterfactual loss is non-finite")
                direct.backward()
                flow_mag_reports.append(float(torch.stack(flow_mags).mean().detach()))
                flow_dir_reports.append(float(torch.stack(flow_dirs).mean().detach()))
                state_reports.append(float(torch.stack(state_losses).mean().detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            absolute_reports.append(absolute_group)
            del cpu

        row = {
            "stage": "v128_counterfactual_first_joint_a2",
            "contract": JOINT_DIRECT_A2_V3_CONTRACT,
            "epoch": int(epoch),
            "teacher_phase": int(phase),
            "absolute_loss": float(np.mean(absolute_reports)) if absolute_reports else 0.0,
            "direct_flow_magnitude_loss": float(np.mean(flow_mag_reports)) if flow_mag_reports else 0.0,
            "direct_flow_direction_loss": float(np.mean(flow_dir_reports)) if flow_dir_reports else 0.0,
            "direct_next_state_effect_loss": float(np.mean(state_reports)) if state_reports else 0.0,
            "selected_direct_counterfactual_pairs": int(selected_pairs),
            "response_weighted_direct_state_effect": True,
            "common_authoritative_prefix_used": True,
            "full_horizon_feedback_used_as_direct_label": False,
            "gradient_label_used": False,
            **_cuda_peak(device),
        }
        history.append(row)
        print("[V128_COUNTERFACTUAL_A2] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        gc.collect()
    return history


def train_counterfactual_first_stage_a_v3(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    depth_index: int,
    flood_rate_index: int,
    design: V127ControlTrainingDesign,
) -> list[dict[str, Any]]:
    a0 = train_direct_action_flow_a0_v3(
        model, source_caches=source_caches, source_groups=source_groups,
        normalization=normalization, graph=graph, device=device, design=design
    )
    a1 = train_oracle_hydraulic_a1_v3(
        model, source_caches=source_caches, source_groups=source_groups,
        normalization=normalization, graph=graph, device=device,
        depth_index=depth_index, flood_rate_index=flood_rate_index, design=design
    )
    a2 = train_joint_direct_a2_v3(
        model, source_caches=source_caches, source_groups=source_groups,
        normalization=normalization, graph=graph, device=device,
        depth_index=depth_index, flood_rate_index=flood_rate_index, design=design
    )
    return [a0, a1, *a2]


__all__ = [
    "COUNTERFACTUAL_STAGE_A_V3_CONTRACT",
    "DIRECT_FLOW_A0_V3_CONTRACT",
    "ORACLE_HYDRAULIC_A1_V3_CONTRACT",
    "JOINT_DIRECT_A2_V3_CONTRACT",
    "_zero_based_spec_order",
    "train_direct_action_flow_a0_v3",
    "train_oracle_hydraulic_a1_v3",
    "train_joint_direct_a2_v3",
    "train_counterfactual_first_stage_a_v3",
]
