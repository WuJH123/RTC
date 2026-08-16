"""Final Development curriculum for counterfactual-first Project7 V128.

This module removes two hidden ambiguities from the earlier repair:

* Stage A, B0 and the post-objective trajectory anchor call the V128 lazy mmap helpers directly;
  they do not rely on monkey-patching function aliases imported before lazy streaming is installed.
* A1 direct-effect supervision uses a strict managed-flow-only oracle transition.  Reference and
  candidate share one action-context setting so the hydraulic transition cannot bypass managed
  flow through a branch-varying ``setting -> typed message -> state`` path.

Training remains trajectory/counterfactual-first.  SWMM action-gradient labels are never used.
The exact H360 TFV pairwise objective is retained downstream after direct and autoregressive
Development gates.
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
    _PREFIX_FLOW_ATOL,
    _PREFIX_STATE_ATOL,
    _SETTING_TOL,
    direct_effect_loss,
)
from .step2_lazy_stream_v128 import (
    cpu_group_v128_lazy,
    install_v128_lazy_streaming,
    select_to_device_v128_lazy,
)
from .step2_oracle_isolation_v128 import (
    ORACLE_FLOW_ISOLATION_CONTRACT,
    oracle_flow_transition_prediction,
    shared_reference_setting,
)
from .step2_train_response_v60 import InputNormalizationV60
from .step2_train_v127 import _ordered, _static
from .step2_train_v127_control import V127ControlTrainingDesign, _candidate_permutation, _state_weights
from .step2_train_v127_streaming import _cuda_peak, _reset_cuda_peak
from .step2_train_v128_exact import train_objective_stage_streaming_v128

COUNTERFACTUAL_STAGE_A_V5_CONTRACT = (
    "PROJECT7_V128_COUNTERFACTUAL_FIRST_STAGE_A_V5_EXPLICIT_LAZY_ORACLE_ISOLATED"
)
DIRECT_FLOW_A0_V5_CONTRACT = "PROJECT7_V128_DIRECT_SAME_PREFIX_FLOW_A0_V5_EXPLICIT_LAZY"
ORACLE_HYDRAULIC_A1_V5_CONTRACT = (
    "PROJECT7_V128_ORACLE_FLOW_ONLY_HYDRAULIC_A1_V5_SETTING_BYPASS_BLOCKED"
)
JOINT_DIRECT_A2_V5_CONTRACT = "PROJECT7_V128_JOINT_DIRECT_HYDRAULIC_A2_V5_EXPLICIT_LAZY"
COUNTERFACTUAL_B0_V5_CONTRACT = (
    "PROJECT7_V128_COUNTERFACTUAL_AUTOREGRESSIVE_B0_V5_EXPLICIT_LAZY_FEEDBACK"
)
POST_OBJECTIVE_TRAJECTORY_ANCHOR_V5_CONTRACT = (
    "PROJECT7_V128_POST_EXACT_TRAJECTORY_ANCHOR_V5_EXPLICIT_LAZY"
)


def _zero_based_spec_order(count: int, *, group_name: str, epoch: int, seed: int) -> np.ndarray:
    """Adapt canonical one-based candidate positions to a zero-based direct-spec list."""
    values = _candidate_permutation(count, group_name=group_name, epoch=epoch, seed=seed)
    if values.size == 0:
        return values
    result = values - 1
    if int(result.min()) < 0 or int(result.max()) >= count:
        raise RuntimeError("direct-pair permutation escaped zero-based spec range")
    return result


def _truth_slice_cpu(value: Any, positions: Sequence[int], *, horizon: int) -> torch.Tensor:
    logical = np.asarray(tuple(int(v) for v in positions), dtype=np.int64)
    if logical.size <= 0 or int(horizon) <= 0:
        raise ValueError("direct truth slice requires positions and positive horizon")
    if hasattr(value, "select"):
        array = value.select(logical, horizon=int(horizon))
        return torch.as_tensor(array, dtype=torch.float32, device="cpu")
    tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    index = torch.as_tensor(logical, dtype=torch.long, device="cpu")
    return tensor.index_select(0, index)[:, : int(horizon)]


def _direct_specs_lazy(cpu: dict[str, Any]) -> list[dict[str, Any]]:
    """Find first-divergence, single-actuator, common-prefix direct pairs without full materialization."""
    settings = torch.as_tensor(cpu["settings"], dtype=torch.float32, device="cpu")
    if settings.ndim != 3:
        raise ValueError("direct-pair settings must be [branch,H,actuator]")
    branches, _, _ = settings.shape
    specs: list[dict[str, Any]] = []
    reference = settings[0]
    for candidate in range(1, int(branches)):
        delta = settings[candidate] - reference
        steps = torch.nonzero(
            torch.any(torch.abs(delta) > _SETTING_TOL, dim=1), as_tuple=False
        ).reshape(-1)
        if steps.numel() == 0:
            continue
        step = int(steps[0])
        changed = torch.nonzero(
            torch.abs(delta[step]) > _SETTING_TOL, as_tuple=False
        ).reshape(-1)
        if changed.numel() != 1:
            continue
        actuator = int(changed[0])
        if step == 0:
            state_gap = 0.0
            flow_gap = 0.0
            target_flow = _truth_slice_cpu(cpu["flows"], (0, candidate), horizon=1)[:, 0]
        else:
            state_pair = _truth_slice_cpu(cpu["states"], (0, candidate), horizon=step)
            flow_pair = _truth_slice_cpu(cpu["flows"], (0, candidate), horizon=step + 1)
            state_gap = float(
                torch.max(torch.abs(state_pair[1, step - 1] - state_pair[0, step - 1]))
            )
            flow_gap = float(
                torch.max(torch.abs(flow_pair[1, step - 1] - flow_pair[0, step - 1]))
            )
            target_flow = flow_pair[:, step]
        if state_gap > _PREFIX_STATE_ATOL or flow_gap > _PREFIX_FLOW_ATOL:
            continue
        specs.append(
            {
                "candidate_position": int(candidate),
                "step": int(step),
                "actuator_index": actuator,
                "setting_delta": float(delta[step, actuator]),
                "true_flow_delta": float(target_flow[1, actuator] - target_flow[0, actuator]),
                "prefix_state_max_abs": state_gap,
                "prefix_flow_max_abs": flow_gap,
            }
        )
    return specs


def _direct_pair_cpu(cpu: dict[str, Any], spec: dict[str, Any]) -> dict[str, torch.Tensor]:
    candidate = int(spec["candidate_position"])
    step = int(spec["step"])
    settings = torch.as_tensor(cpu["settings"], dtype=torch.float32, device="cpu")
    if step == 0:
        prefix_state = torch.as_tensor(cpu["initial"][0], dtype=torch.float32, device="cpu")
        prefix_flow = torch.as_tensor(cpu["previous_flow"][0], dtype=torch.float32, device="cpu")
    else:
        state_pair = _truth_slice_cpu(cpu["states"], (0, candidate), horizon=step)
        flow_pair = _truth_slice_cpu(cpu["flows"], (0, candidate), horizon=step)
        prefix_state = state_pair[0, step - 1]
        prefix_flow = flow_pair[0, step - 1]
    target_state = _truth_slice_cpu(cpu["states"], (0, candidate), horizon=step + 1)[:, step]
    target_flow = _truth_slice_cpu(cpu["flows"], (0, candidate), horizon=step + 1)[:, step]
    return {
        "prefix_state": prefix_state,
        "prefix_flow": prefix_flow,
        "setting": torch.stack((settings[0, step], settings[candidate, step])),
        "target_state": target_state,
        "target_flow": target_flow,
        "rainfall": torch.as_tensor(cpu["rainfall"][0, step], dtype=torch.float32, device="cpu"),
    }


def _direct_rainfall_batch(
    pair: dict[str, torch.Tensor], *, branches: int, device: torch.device
) -> torch.Tensor:
    """Expand one common-prefix direct-pair rainfall frame across candidate branches."""
    rainfall = pair["rainfall"]
    if rainfall.ndim != 2:
        raise ValueError("direct-pair rainfall must be [node, rainfall_feature]")
    if int(branches) <= 0:
        raise ValueError("direct-pair rainfall batch requires a positive branch count")
    return rainfall.unsqueeze(0).expand(int(branches), -1, -1).to(device)


def _predict_direct_flow(
    model: Any,
    *,
    pair: dict[str, torch.Tensor],
    static: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    state = pair["prefix_state"][None].repeat(2, 1, 1).to(device)
    flow = pair["prefix_flow"][None].repeat(2, 1).to(device)
    setting = pair["setting"].to(device)
    physics_norm, identity = model.actuator.prepare_static(static["physics"], batch_size=2)
    q, response = model.actuator.forward_prepared(
        state[:, static["up"]], state[:, static["down"]], setting, flow, physics_norm, identity
    )
    return q, response, state, flow, setting


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


def train_direct_action_flow_a0_v5(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    design: V127ControlTrainingDesign,
) -> dict[str, Any]:
    """A0: absolute managed-flow fit plus same-prefix direct magnitude/direction supervision."""
    design.validate()
    torch.manual_seed(design.seed + 5201)
    np.random.seed(design.seed + 5201)
    model.train().to(device)
    static = _static(graph, device)
    optimizer = torch.optim.AdamW(
        model.actuator.parameters(),
        lr=float(design.learning_rate),
        weight_decay=float(design.weight_decay),
    )
    absolute_reports: list[float] = []
    effect_reports: list[float] = []
    magnitude_reports: list[float] = []
    direction_reports: list[float] = []
    selected_pairs = 0
    direct_nonzero = 0
    _reset_cuda_peak(device)

    for source, name in _ordered(source_groups, 5201, design.seed):
        cpu = cpu_group_v128_lazy(source_caches[source], name, normalization)
        branches, horizon = cpu["settings"].shape[:2]
        optimizer.zero_grad(set_to_none=True)
        absolute_group = 0.0
        for start in range(0, branches, design.hydraulic_branch_chunk):
            stop = min(start + design.hydraulic_branch_chunk, branches)
            positions = np.arange(start, stop, dtype=np.int64)
            chunk = select_to_device_v128_lazy(
                cpu, positions, device=device, include_truth=True
            )
            physics_norm, identity = model.actuator.prepare_static(
                static["physics"], batch_size=len(positions)
            )
            term = torch.zeros((), device=device)
            for k in range(horizon):
                prev_state = chunk["initial"] if k == 0 else chunk["states"][:, k - 1]
                prev_flow = chunk["previous_flow"] if k == 0 else chunk["flows"][:, k - 1]
                q, _ = model.actuator.forward_prepared(
                    prev_state[:, static["up"]],
                    prev_state[:, static["down"]],
                    chunk["settings"][:, k],
                    prev_flow,
                    physics_norm,
                    identity,
                )
                error = (q - chunk["flows"][:, k]) / model.actuator.flow_std
                term = term + F.smooth_l1_loss(
                    error, torch.zeros_like(error), beta=0.5
                )
            weighted = (term / float(horizon)) * (float(len(positions)) / float(branches))
            if not bool(torch.isfinite(weighted)):
                raise RuntimeError(f"{name}: A0 absolute-flow loss is non-finite")
            weighted.backward()
            absolute_group += float(weighted.detach())
            del chunk, term, weighted

        specs = _direct_specs_lazy(cpu)
        budget = min(8 if str(source).upper() == "D2" else 4, len(specs))
        if budget:
            order = _zero_based_spec_order(
                len(specs), group_name=name, epoch=5201, seed=design.seed
            )[:budget]
            losses: list[torch.Tensor] = []
            magnitudes: list[torch.Tensor] = []
            directions: list[torch.Tensor] = []
            for offset in order.tolist():
                spec = specs[int(offset)]
                pair = _direct_pair_cpu(cpu, spec)
                q, _, _, _, _ = _predict_direct_flow(
                    model, pair=pair, static=static, device=device
                )
                actuator = int(spec["actuator_index"])
                target = pair["target_flow"].to(device)
                true_delta = target[1, actuator] - target[0, actuator]
                loss, magnitude, direction = direct_effect_loss(
                    q[1, actuator] - q[0, actuator],
                    true_delta,
                    model.actuator.direct_action_flow_scale[actuator],
                )
                losses.append(loss)
                magnitudes.append(magnitude)
                directions.append(direction)
                selected_pairs += 1
                direct_nonzero += int(
                    abs(float(true_delta.detach().cpu())) > _DIRECT_EFFECT_FLOOR
                )
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
        "contract": DIRECT_FLOW_A0_V5_CONTRACT,
        "absolute_flow_loss": float(np.mean(absolute_reports)) if absolute_reports else 0.0,
        "direct_flow_effect_loss": float(np.mean(effect_reports)) if effect_reports else 0.0,
        "direct_flow_magnitude_loss": float(np.mean(magnitude_reports)) if magnitude_reports else 0.0,
        "direct_flow_direction_loss": float(np.mean(direction_reports)) if direction_reports else 0.0,
        "selected_direct_counterfactual_pairs": int(selected_pairs),
        "nonzero_direct_counterfactual_pairs": int(direct_nonzero),
        "common_authoritative_prefix_used": True,
        "explicit_lazy_mmap_helpers": True,
        "full_horizon_feedback_used_as_direct_label": False,
        "gradient_label_used": False,
        "fit_only": True,
        **_cuda_peak(device),
    }
    print("[V128_COUNTERFACTUAL_A0] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    gc.collect()
    return row


def train_oracle_hydraulic_a1_v5(
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
    """A1: train q->next-state dynamics and block branch-varying setting bypass in direct pairs."""
    design.validate()
    model.train().to(device)
    static = _static(graph, device)
    state_weights = _state_weights(
        model,
        depth_index=depth_index,
        flood_rate_index=flood_rate_index,
        design=design,
        device=device,
    )
    actuator_flags = [p.requires_grad for p in model.actuator.parameters()]
    for parameter in model.actuator.parameters():
        parameter.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(design.learning_rate),
        weight_decay=float(design.weight_decay),
    )
    absolute_reports: list[float] = []
    direct_reports: list[float] = []
    selected_pairs = 0
    _reset_cuda_peak(device)
    try:
        for source, name in _ordered(source_groups, 5301, design.seed):
            cpu = cpu_group_v128_lazy(source_caches[source], name, normalization)
            branches, horizon = cpu["settings"].shape[:2]
            optimizer.zero_grad(set_to_none=True)
            absolute_group = 0.0
            for start in range(0, branches, design.hydraulic_branch_chunk):
                stop = min(start + design.hydraulic_branch_chunk, branches)
                positions = np.arange(start, stop, dtype=np.int64)
                chunk = select_to_device_v128_lazy(
                    cpu, positions, device=device, include_truth=True
                )
                term = torch.zeros((), device=device)
                steps = len(range(0, horizon, design.teacher_stride))
                for k in range(0, horizon, design.teacher_stride):
                    prev_state = chunk["initial"] if k == 0 else chunk["states"][:, k - 1]
                    prev_flow = chunk["previous_flow"] if k == 0 else chunk["flows"][:, k - 1]
                    pred = oracle_flow_transition_prediction(
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
                        error * state_weights,
                        torch.zeros_like(error),
                        beta=0.5,
                    )
                weighted = (term / float(steps)) * (float(len(positions)) / float(branches))
                if not bool(torch.isfinite(weighted)):
                    raise RuntimeError(f"{name}: A1 oracle-flow absolute loss is non-finite")
                weighted.backward()
                absolute_group += float(weighted.detach())
                del chunk, term, weighted

            specs = _direct_specs_lazy(cpu)
            budget = min(4, len(specs))
            if budget:
                order = _zero_based_spec_order(
                    len(specs), group_name=name, epoch=5301, seed=design.seed
                )[:budget]
                losses: list[torch.Tensor] = []
                for offset in order.tolist():
                    spec = specs[int(offset)]
                    pair = _direct_pair_cpu(cpu, spec)
                    prev_state = pair["prefix_state"][None].repeat(2, 1, 1).to(device)
                    prev_flow = pair["prefix_flow"][None].repeat(2, 1).to(device)
                    setting = pair["setting"].to(device)
                    oracle_flow = pair["target_flow"].to(device)
                    rainfall = _direct_rainfall_batch(pair, branches=2, device=device)
                    pred = oracle_flow_transition_prediction(
                        model,
                        prev_state=prev_state,
                        previous_flow=prev_flow,
                        setting=setting,
                        oracle_flow=oracle_flow,
                        rainfall=rainfall,
                        static=static,
                        action_context_setting=shared_reference_setting(setting),
                    )
                    truth = pair["target_state"].to(device)
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
        "contract": ORACLE_HYDRAULIC_A1_V5_CONTRACT,
        "oracle_isolation_contract": ORACLE_FLOW_ISOLATION_CONTRACT,
        "absolute_state_loss": float(np.mean(absolute_reports)) if absolute_reports else 0.0,
        "direct_next_state_effect_loss": float(np.mean(direct_reports)) if direct_reports else 0.0,
        "selected_direct_counterfactual_pairs": int(selected_pairs),
        "authoritative_managed_flow_injection": True,
        "actuator_parameters_frozen": True,
        "direct_pair_setting_bypass_blocked": True,
        "response_weighted_direct_state_effect": True,
        "explicit_lazy_mmap_helpers": True,
        "gradient_label_used": False,
        "fit_only": True,
        **_cuda_peak(device),
    }
    print("[V128_COUNTERFACTUAL_A1] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    gc.collect()
    return row


def train_joint_direct_a2_v5(
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
    """A2: joint predicted-flow teacher forcing plus same-prefix direct flow/state effects."""
    design.validate()
    model.train().to(device)
    static = _static(graph, device)
    state_weights = _state_weights(
        model,
        depth_index=depth_index,
        flood_rate_index=flood_rate_index,
        design=design,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=design.learning_rate, weight_decay=design.weight_decay
    )
    history: list[dict[str, Any]] = []
    _reset_cuda_peak(device)

    for epoch in range(1, design.hydraulic_epochs + 1):
        phase = (epoch - 1) % design.teacher_stride
        absolute_reports: list[float] = []
        flow_mag_reports: list[float] = []
        flow_dir_reports: list[float] = []
        state_reports: list[float] = []
        selected_pairs = 0
        for source, name in _ordered(source_groups, epoch + 5400, design.seed):
            cpu = cpu_group_v128_lazy(source_caches[source], name, normalization)
            branches, horizon = cpu["settings"].shape[:2]
            optimizer.zero_grad(set_to_none=True)
            absolute_group = 0.0
            steps = len(range(phase, horizon, design.teacher_stride))
            if steps <= 0:
                raise RuntimeError("A2 selected no teacher-forced transitions")
            for start in range(0, branches, design.hydraulic_branch_chunk):
                stop = min(start + design.hydraulic_branch_chunk, branches)
                positions = np.arange(start, stop, dtype=np.int64)
                chunk = select_to_device_v128_lazy(
                    cpu, positions, device=device, include_truth=True
                )
                physics_norm, identity = model.actuator.prepare_static(
                    static["physics"], batch_size=len(positions)
                )
                static_norm, edges, inv = model.transition.prepare_static(
                    static["static"],
                    static["edges"],
                    batch_size=len(positions),
                    dtype=torch.float32,
                )
                term = torch.zeros((), device=device)
                for k in range(phase, horizon, design.teacher_stride):
                    pred_state, q = _teacher_step(
                        model,
                        chunk=chunk,
                        k=k,
                        static=static,
                        physics_norm=physics_norm,
                        identity=identity,
                        static_norm=static_norm,
                        edges=edges,
                        inv=inv,
                    )
                    state_error = (pred_state - chunk["states"][:, k]) / model.transition.state_std
                    flow_error = (q - chunk["flows"][:, k]) / model.actuator.flow_std
                    term = term + F.smooth_l1_loss(
                        state_error * state_weights,
                        torch.zeros_like(state_error),
                        beta=0.5,
                    )
                    term = term + design.flow_weight * F.smooth_l1_loss(
                        flow_error, torch.zeros_like(flow_error), beta=0.5
                    )
                weighted = (term / float(steps)) * (float(len(positions)) / float(branches))
                if not bool(torch.isfinite(weighted)):
                    raise RuntimeError(f"{name}: A2 absolute joint loss is non-finite")
                weighted.backward()
                absolute_group += float(weighted.detach())
                del chunk, term, weighted

            specs = _direct_specs_lazy(cpu)
            budget = min(4, len(specs))
            if budget:
                order = _zero_based_spec_order(
                    len(specs), group_name=name, epoch=epoch + 5400, seed=design.seed
                )[:budget]
                losses: list[torch.Tensor] = []
                flow_mags: list[torch.Tensor] = []
                flow_dirs: list[torch.Tensor] = []
                state_losses: list[torch.Tensor] = []
                for offset in order.tolist():
                    spec = specs[int(offset)]
                    pair = _direct_pair_cpu(cpu, spec)
                    q, response, prev_state, prev_flow, setting = _predict_direct_flow(
                        model, pair=pair, static=static, device=device
                    )
                    physics_norm, identity = model.actuator.prepare_static(
                        static["physics"], batch_size=2
                    )
                    static_norm, edges, inv = model.transition.prepare_static(
                        static["static"],
                        static["edges"],
                        batch_size=2,
                        dtype=torch.float32,
                    )
                    injection = torch.zeros(
                        2,
                        prev_state.shape[1],
                        1,
                        device=device,
                        dtype=prev_state.dtype,
                    )
                    injection = injection.index_add(1, static["up"], -q[..., None])
                    injection = injection.index_add(1, static["down"], q[..., None])
                    context = model._typed_action_context(
                        state=prev_state,
                        setting=setting,
                        previous_flow=prev_flow,
                        predicted_flow=q,
                        responsiveness=response,
                        upstream=static["up"],
                        downstream=static["down"],
                        physics_norm=physics_norm,
                        identity_embedding=identity,
                    )
                    rainfall = _direct_rainfall_batch(pair, branches=2, device=device)
                    pred_state = model.transition.forward_prepared(
                        prev_state,
                        rainfall,
                        static_norm,
                        injection,
                        edges,
                        inv,
                        context,
                    )
                    target_flow = pair["target_flow"].to(device)
                    target_state = pair["target_state"].to(device)
                    actuator = int(spec["actuator_index"])
                    flow_loss, flow_mag, flow_dir = direct_effect_loss(
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
            "contract": JOINT_DIRECT_A2_V5_CONTRACT,
            "epoch": int(epoch),
            "teacher_phase": int(phase),
            "absolute_loss": float(np.mean(absolute_reports)) if absolute_reports else 0.0,
            "direct_flow_magnitude_loss": float(np.mean(flow_mag_reports)) if flow_mag_reports else 0.0,
            "direct_flow_direction_loss": float(np.mean(flow_dir_reports)) if flow_dir_reports else 0.0,
            "direct_next_state_effect_loss": float(np.mean(state_reports)) if state_reports else 0.0,
            "selected_direct_counterfactual_pairs": int(selected_pairs),
            "response_weighted_direct_state_effect": True,
            "common_authoritative_prefix_used": True,
            "explicit_lazy_mmap_helpers": True,
            "full_horizon_feedback_used_as_direct_label": False,
            "gradient_label_used": False,
            **_cuda_peak(device),
        }
        history.append(row)
        print("[V128_COUNTERFACTUAL_A2] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        gc.collect()
    return history


def train_counterfactual_first_stage_a_v5(
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
    a0 = train_direct_action_flow_a0_v5(
        model,
        source_caches=source_caches,
        source_groups=source_groups,
        normalization=normalization,
        graph=graph,
        device=device,
        design=design,
    )
    a1 = train_oracle_hydraulic_a1_v5(
        model,
        source_caches=source_caches,
        source_groups=source_groups,
        normalization=normalization,
        graph=graph,
        device=device,
        depth_index=depth_index,
        flood_rate_index=flood_rate_index,
        design=design,
    )
    a2 = train_joint_direct_a2_v5(
        model,
        source_caches=source_caches,
        source_groups=source_groups,
        normalization=normalization,
        graph=graph,
        device=device,
        depth_index=depth_index,
        flood_rate_index=flood_rate_index,
        design=design,
    )
    return [a0, a1, *a2]


def train_counterfactual_rollout_b0_v5(
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
    """B0: explicit-lazy autoregressive trajectory training where network feedback belongs."""
    design.validate()
    static = _static(graph, device)
    state_weights = _state_weights(
        model,
        depth_index=depth_index,
        flood_rate_index=flood_rate_index,
        design=design,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=design.rollout_learning_rate,
        weight_decay=design.weight_decay,
    )
    model.train().to(device)
    history: list[dict[str, Any]] = []
    _reset_cuda_peak(device)

    for epoch, horizon in enumerate(design.rollout_horizons, start=1):
        losses: list[float] = []
        delta_flow_reports: list[float] = []
        delta_state_reports: list[float] = []
        for source, name in _ordered(source_groups, epoch + 601, design.seed):
            cpu = cpu_group_v128_lazy(source_caches[source], name, normalization)
            branches = int(cpu["settings"].shape[0])
            candidates = _candidate_permutation(
                branches - 1, group_name=name, epoch=epoch, seed=design.seed
            )[: min(design.rollout_candidates_per_group, branches - 1)]
            positions = np.concatenate((np.asarray([0], dtype=np.int64), candidates))
            chunk = select_to_device_v128_lazy(
                cpu,
                positions,
                device=device,
                horizon=horizon,
                include_truth=True,
            )
            optimizer.zero_grad(set_to_none=True)
            output = model.rollout(
                chunk["initial"],
                chunk["rainfall"],
                chunk["settings"],
                chunk["previous_flow"],
                static["up"],
                static["down"],
                static["physics"],
                static["static"],
                static["edges"],
            )
            state_error = (output.states - chunk["states"]) / model.transition.state_std
            flow_error = (output.actuator_flows - chunk["flows"]) / model.actuator.flow_std
            absolute = F.smooth_l1_loss(
                state_error * state_weights,
                torch.zeros_like(state_error),
                beta=0.5,
            )
            absolute = absolute + design.flow_weight * F.smooth_l1_loss(
                flow_error, torch.zeros_like(flow_error), beta=0.5
            )
            pred_state_delta = output.states[1:] - output.states[0:1]
            true_state_delta = chunk["states"][1:] - chunk["states"][0:1]
            pred_flow_delta = output.actuator_flows[1:] - output.actuator_flows[0:1]
            true_flow_delta = chunk["flows"][1:] - chunk["flows"][0:1]
            delta_state = _response_weighted_effect_loss(
                pred_state_delta * state_weights,
                true_state_delta * state_weights,
                model.transition.delta_state_scale,
            )
            # B0 learns full network-feedback trajectory effects, not local dq/du.  Normalize
            # branch-wise flow effects with the physical training standard deviation rather than
            # the direct same-prefix action scale or the temporal residual scale.
            delta_flow = _response_weighted_effect_loss(
                pred_flow_delta,
                true_flow_delta,
                model.actuator.flow_std,
            )
            loss = absolute + delta_state + delta_flow
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"{name}: counterfactual B0 rollout loss is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach()))
            delta_state_reports.append(float(delta_state.detach()))
            delta_flow_reports.append(float(delta_flow.detach()))
            del cpu, chunk, output, loss, absolute, delta_state, delta_flow

        row = {
            "stage": "v128_counterfactual_autoregressive_rollout",
            "contract": COUNTERFACTUAL_B0_V5_CONTRACT,
            "epoch": int(epoch),
            "horizon_steps": int(horizon),
            "horizon_minutes": int(horizon * 5),
            "candidates_per_group": int(design.rollout_candidates_per_group),
            "loss": float(np.mean(losses)),
            "delta_state_effect_loss": float(np.mean(delta_state_reports)),
            "delta_flow_effect_loss": float(np.mean(delta_flow_reports)),
            "feedback_flow_effect_scale": "actuator_flow_std_not_direct_action_scale",
            "full_network_feedback_permitted": True,
            "explicit_lazy_mmap_helpers": True,
            "fit_only_action_effect_supervision": True,
            **_cuda_peak(device),
        }
        history.append(row)
        print("[V128_COUNTERFACTUAL_B0] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        gc.collect()
    return history


def _post_objective_trajectory_anchor_v5(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    design: V127ControlTrainingDesign,
) -> dict[str, Any]:
    """One FIT-only H360 trajectory-effect rehearsal after exact TFV pairwise training."""
    static = _static(graph, device)
    model.train().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max(float(design.objective_learning_rate) * 0.25, 1.0e-6),
        weight_decay=design.weight_decay,
    )
    losses: list[float] = []
    selected_pairs = 0
    _reset_cuda_peak(device)
    for source, name in _ordered(source_groups, 991, design.seed):
        cpu = cpu_group_v128_lazy(source_caches[source], name, normalization)
        candidate_count = int(cpu["settings"].shape[0]) - 1
        if candidate_count <= 0:
            del cpu
            continue
        candidate = _candidate_permutation(
            candidate_count, group_name=name, epoch=991, seed=design.seed
        )[:1]
        positions = np.concatenate((np.asarray([0], dtype=np.int64), candidate))
        chunk = select_to_device_v128_lazy(
            cpu, positions, device=device, horizon=72, include_truth=True
        )
        optimizer.zero_grad(set_to_none=True)
        output = model.rollout(
            chunk["initial"],
            chunk["rainfall"],
            chunk["settings"],
            chunk["previous_flow"],
            static["up"],
            static["down"],
            static["physics"],
            static["static"],
            static["edges"],
        )
        delta_state = _response_weighted_effect_loss(
            output.states[1:] - output.states[0:1],
            chunk["states"][1:] - chunk["states"][0:1],
            model.transition.delta_state_scale,
        )
        delta_flow = _response_weighted_effect_loss(
            output.actuator_flows[1:] - output.actuator_flows[0:1],
            chunk["flows"][1:] - chunk["flows"][0:1],
            model.actuator.flow_std,
        )
        loss = 0.5 * (delta_state + delta_flow)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"{name}: post-objective trajectory anchor is non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach()))
        selected_pairs += 1
        del cpu, chunk, output, delta_state, delta_flow, loss
    row = {
        "stage": "v128_post_exact_trajectory_effect_anchor",
        "contract": POST_OBJECTIVE_TRAJECTORY_ANCHOR_V5_CONTRACT,
        "loss": float(np.mean(losses)) if losses else 0.0,
        "selected_fit_pairs": int(selected_pairs),
        "horizon_steps": 72,
        "feedback_flow_effect_scale": "actuator_flow_std_not_direct_action_scale",
        "learning_rate_fraction_of_exact_objective": 0.25,
        "fit_only_action_effect_supervision": True,
        "changes_exact_pairwise_census": False,
        "explicit_lazy_mmap_helpers": True,
        **_cuda_peak(device),
    }
    print("[V128_COUNTERFACTUAL_POST_OBJECTIVE] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    gc.collect()
    return row


def train_counterfactual_objective_stage_v5(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    flood_rate_index: int,
    design: V127ControlTrainingDesign,
) -> list[dict[str, Any]]:
    """Run exact H360 pairwise TFV training then one explicit-lazy trajectory rehearsal."""
    # The exact objective has its own imported streaming globals.  Install the audited lazy
    # helpers immediately before invoking it so its function globals are guaranteed to be patched.
    install_v128_lazy_streaming()
    history = list(
        train_objective_stage_streaming_v128(
            model,
            source_caches=source_caches,
            source_groups=source_groups,
            normalization=normalization,
            graph=graph,
            device=device,
            flood_rate_index=flood_rate_index,
            design=design,
        )
    )
    history.append(
        _post_objective_trajectory_anchor_v5(
            model,
            source_caches=source_caches,
            source_groups=source_groups,
            normalization=normalization,
            graph=graph,
            device=device,
            design=design,
        )
    )
    return history


__all__ = [
    "COUNTERFACTUAL_B0_V5_CONTRACT",
    "COUNTERFACTUAL_STAGE_A_V5_CONTRACT",
    "DIRECT_FLOW_A0_V5_CONTRACT",
    "JOINT_DIRECT_A2_V5_CONTRACT",
    "ORACLE_HYDRAULIC_A1_V5_CONTRACT",
    "POST_OBJECTIVE_TRAJECTORY_ANCHOR_V5_CONTRACT",
    "_direct_specs_lazy",
    "_zero_based_spec_order",
    "train_counterfactual_first_stage_a_v5",
    "train_counterfactual_objective_stage_v5",
    "train_counterfactual_rollout_b0_v5",
]
