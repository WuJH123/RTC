"""Balanced direct-supervision Stage A for the current Project7 V128 Development surrogate.

This module keeps the existing model, causal data boundaries and A0/A1/A2 objectives intact.
It changes only which same-prefix direct pairs are used inside Stage A:

* prefer authoritative non-zero managed-flow responses;
* prefer actuators that have been selected fewer times in the current stage/epoch;
* within equal coverage, prefer larger authoritative direct responses;
* use a modestly larger D2 budget because D2 is the dedicated single-actuator response source.

The purpose is to use the already-generated Development data efficiently before changing model
architecture. Validation/Final/Formal data and SWMM gradient labels are never used.
"""
from __future__ import annotations

import gc
import hashlib
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_action_identifiable_v128 import _teacher_step
from .step2_counterfactual_first_v128 import _DIRECT_EFFECT_FLOOR, direct_effect_loss
from .step2_counterfactual_training_v5 import (
    _direct_pair_cpu,
    _direct_rainfall_batch,
    _direct_specs_lazy,
    _direct_state_effect_loss,
    _predict_direct_flow,
)
from .step2_lazy_stream_v128 import cpu_group_v128_lazy, select_to_device_v128_lazy
from .step2_oracle_isolation_v128 import (
    ORACLE_FLOW_ISOLATION_CONTRACT,
    oracle_flow_transition_prediction,
    shared_reference_setting,
)
from .step2_train_response_v60 import InputNormalizationV60
from .step2_train_v127 import _ordered, _static
from .step2_train_v127_control import V127ControlTrainingDesign, _state_weights
from .step2_train_v127_streaming import _cuda_peak, _reset_cuda_peak

BALANCED_STAGE_A_CONTRACT = (
    "PROJECT7_V128_STAGE_A_BALANCED_DIRECT_SUPERVISION_V1"
)
BALANCED_A0_CONTRACT = (
    "PROJECT7_V128_DIRECT_SAME_PREFIX_FLOW_A0_BALANCED_V1"
)
BALANCED_A1_CONTRACT = (
    "PROJECT7_V128_ORACLE_FLOW_ONLY_HYDRAULIC_A1_BALANCED_V1"
)
BALANCED_A2_CONTRACT = (
    "PROJECT7_V128_JOINT_DIRECT_HYDRAULIC_A2_BALANCED_V1"
)
DIRECT_PAIR_SAMPLING_CONTRACT = (
    "PROJECT7_V128_FIT_ONLY_NONZERO_FIRST_ACTUATOR_BALANCED_DIRECT_PAIR_SAMPLING_V1"
)

D2_DIRECT_PAIR_BUDGET = 12
OTHER_DIRECT_PAIR_BUDGET = 6


def _stable_tie(group_name: str, epoch: int, spec: dict[str, Any], seed: int) -> str:
    text = (
        f"{seed}:{epoch}:{group_name}:"
        f"{int(spec['candidate_position'])}:{int(spec['actuator_index'])}"
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def select_balanced_direct_specs(
    specs: Sequence[dict[str, Any]],
    *,
    source: str,
    group_name: str,
    epoch: int,
    seed: int,
    actuator_counts: np.ndarray,
) -> list[dict[str, Any]]:
    """Select informative direct pairs while equalising actuator exposure across groups."""
    if actuator_counts.ndim != 1:
        raise ValueError("actuator_counts must be one-dimensional")
    budget = D2_DIRECT_PAIR_BUDGET if str(source).upper() == "D2" else OTHER_DIRECT_PAIR_BUDGET
    budget = min(int(budget), len(specs))
    if budget <= 0:
        return []

    ranked: list[tuple[int, int, float, str, int]] = []
    for index, spec in enumerate(specs):
        actuator = int(spec["actuator_index"])
        if not 0 <= actuator < actuator_counts.size:
            raise ValueError("direct-pair actuator index exceeds frozen actuator count")
        effect = abs(float(spec["true_flow_delta"]))
        informative_rank = 0 if effect > _DIRECT_EFFECT_FLOOR else 1
        ranked.append(
            (
                informative_rank,
                int(actuator_counts[actuator]),
                -effect,
                _stable_tie(group_name, epoch, spec, seed),
                int(index),
            )
        )

    chosen = [int(item[-1]) for item in sorted(ranked)[:budget]]
    result = [dict(specs[index]) for index in chosen]
    for spec in result:
        actuator_counts[int(spec["actuator_index"])] += 1
    return result


def _coverage_payload(
    *,
    selected_actuators: set[int],
    informative_actuators: set[int],
    counts: np.ndarray,
) -> dict[str, Any]:
    positive = counts[counts > 0]
    return {
        "direct_pair_sampling_contract": DIRECT_PAIR_SAMPLING_CONTRACT,
        "direct_pair_budget_d2": int(D2_DIRECT_PAIR_BUDGET),
        "direct_pair_budget_other": int(OTHER_DIRECT_PAIR_BUDGET),
        "selected_direct_unique_actuators": int(len(selected_actuators)),
        "selected_direct_informative_unique_actuators": int(len(informative_actuators)),
        "selected_direct_min_pairs_per_covered_actuator": int(positive.min()) if positive.size else 0,
        "selected_direct_max_pairs_per_covered_actuator": int(positive.max()) if positive.size else 0,
        "informative_pairs_prioritized": True,
        "actuator_balanced_sampling": True,
    }


def train_direct_action_flow_a0_balanced(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    design: V127ControlTrainingDesign,
) -> dict[str, Any]:
    """A0 with broader, actuator-balanced same-prefix direct-flow supervision."""
    design.validate()
    torch.manual_seed(design.seed + 6201)
    np.random.seed(design.seed + 6201)
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
    selected_actuators: set[int] = set()
    informative_actuators: set[int] = set()
    counts = np.zeros(int(model.actuator.actuator_count), dtype=np.int64)
    _reset_cuda_peak(device)

    for source, name in _ordered(source_groups, 6201, design.seed):
        cpu = cpu_group_v128_lazy(source_caches[source], name, normalization)
        branches, horizon = cpu["settings"].shape[:2]
        optimizer.zero_grad(set_to_none=True)
        absolute_group = 0.0
        for start in range(0, branches, design.hydraulic_branch_chunk):
            stop = min(start + design.hydraulic_branch_chunk, branches)
            positions = np.arange(start, stop, dtype=np.int64)
            chunk = select_to_device_v128_lazy(cpu, positions, device=device, include_truth=True)
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
                term = term + F.smooth_l1_loss(error, torch.zeros_like(error), beta=0.5)
            weighted = (term / float(horizon)) * (float(len(positions)) / float(branches))
            if not bool(torch.isfinite(weighted)):
                raise RuntimeError(f"{name}: balanced A0 absolute-flow loss is non-finite")
            weighted.backward()
            absolute_group += float(weighted.detach())
            del chunk, term, weighted

        specs = select_balanced_direct_specs(
            _direct_specs_lazy(cpu),
            source=source,
            group_name=name,
            epoch=6201,
            seed=design.seed,
            actuator_counts=counts,
        )
        if specs:
            losses: list[torch.Tensor] = []
            magnitudes: list[torch.Tensor] = []
            directions: list[torch.Tensor] = []
            for spec in specs:
                pair = _direct_pair_cpu(cpu, spec)
                q, _, _, _, _ = _predict_direct_flow(model, pair=pair, static=static, device=device)
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
                selected_actuators.add(actuator)
                informative = abs(float(true_delta.detach().cpu())) > _DIRECT_EFFECT_FLOOR
                direct_nonzero += int(informative)
                if informative:
                    informative_actuators.add(actuator)
            effect = torch.stack(losses).mean()
            if not bool(torch.isfinite(effect)):
                raise RuntimeError(f"{name}: balanced A0 direct-flow loss is non-finite")
            effect.backward()
            effect_reports.append(float(effect.detach()))
            magnitude_reports.append(float(torch.stack(magnitudes).mean().detach()))
            direction_reports.append(float(torch.stack(directions).mean().detach()))
        torch.nn.utils.clip_grad_norm_(model.actuator.parameters(), design.grad_clip)
        optimizer.step()
        absolute_reports.append(absolute_group)
        del cpu

    row = {
        "stage": "v128_direct_same_prefix_action_flow_a0_balanced",
        "contract": BALANCED_A0_CONTRACT,
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
        **_coverage_payload(
            selected_actuators=selected_actuators,
            informative_actuators=informative_actuators,
            counts=counts,
        ),
        **_cuda_peak(device),
    }
    print("[V128_BALANCED_A0] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    gc.collect()
    return row


def train_oracle_hydraulic_a1_balanced(
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
    """A1 with authoritative q and broader informative direct hydraulic supervision."""
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
    selected_actuators: set[int] = set()
    informative_actuators: set[int] = set()
    counts = np.zeros(int(model.actuator.actuator_count), dtype=np.int64)
    _reset_cuda_peak(device)
    try:
        for source, name in _ordered(source_groups, 6301, design.seed):
            cpu = cpu_group_v128_lazy(source_caches[source], name, normalization)
            branches, horizon = cpu["settings"].shape[:2]
            optimizer.zero_grad(set_to_none=True)
            absolute_group = 0.0
            for start in range(0, branches, design.hydraulic_branch_chunk):
                stop = min(start + design.hydraulic_branch_chunk, branches)
                positions = np.arange(start, stop, dtype=np.int64)
                chunk = select_to_device_v128_lazy(cpu, positions, device=device, include_truth=True)
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
                    raise RuntimeError(f"{name}: balanced A1 absolute loss is non-finite")
                weighted.backward()
                absolute_group += float(weighted.detach())
                del chunk, term, weighted

            specs = select_balanced_direct_specs(
                _direct_specs_lazy(cpu),
                source=source,
                group_name=name,
                epoch=6301,
                seed=design.seed,
                actuator_counts=counts,
            )
            if specs:
                losses: list[torch.Tensor] = []
                for spec in specs:
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
                    actuator = int(spec["actuator_index"])
                    selected_pairs += 1
                    selected_actuators.add(actuator)
                    if abs(float(spec["true_flow_delta"])) > _DIRECT_EFFECT_FLOOR:
                        informative_actuators.add(actuator)
                direct = torch.stack(losses).mean()
                if not bool(torch.isfinite(direct)):
                    raise RuntimeError(f"{name}: balanced A1 direct-state loss is non-finite")
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
        "stage": "v128_oracle_managed_flow_hydraulic_a1_balanced",
        "contract": BALANCED_A1_CONTRACT,
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
        **_coverage_payload(
            selected_actuators=selected_actuators,
            informative_actuators=informative_actuators,
            counts=counts,
        ),
        **_cuda_peak(device),
    }
    print("[V128_BALANCED_A1] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    gc.collect()
    return row


def train_joint_direct_a2_balanced(
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
    """A2 joint teacher forcing with the same balanced direct-pair policy."""
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
        selected_actuators: set[int] = set()
        informative_actuators: set[int] = set()
        counts = np.zeros(int(model.actuator.actuator_count), dtype=np.int64)
        for source, name in _ordered(source_groups, epoch + 6400, design.seed):
            cpu = cpu_group_v128_lazy(source_caches[source], name, normalization)
            branches, horizon = cpu["settings"].shape[:2]
            optimizer.zero_grad(set_to_none=True)
            absolute_group = 0.0
            steps = len(range(phase, horizon, design.teacher_stride))
            if steps <= 0:
                raise RuntimeError("balanced A2 selected no teacher-forced transitions")
            for start in range(0, branches, design.hydraulic_branch_chunk):
                stop = min(start + design.hydraulic_branch_chunk, branches)
                positions = np.arange(start, stop, dtype=np.int64)
                chunk = select_to_device_v128_lazy(cpu, positions, device=device, include_truth=True)
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
                    raise RuntimeError(f"{name}: balanced A2 absolute loss is non-finite")
                weighted.backward()
                absolute_group += float(weighted.detach())
                del chunk, term, weighted

            specs = select_balanced_direct_specs(
                _direct_specs_lazy(cpu),
                source=source,
                group_name=name,
                epoch=epoch + 6400,
                seed=design.seed,
                actuator_counts=counts,
            )
            if specs:
                losses: list[torch.Tensor] = []
                flow_mags: list[torch.Tensor] = []
                flow_dirs: list[torch.Tensor] = []
                state_losses: list[torch.Tensor] = []
                for spec in specs:
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
                    selected_actuators.add(actuator)
                    if abs(float(spec["true_flow_delta"])) > _DIRECT_EFFECT_FLOOR:
                        informative_actuators.add(actuator)
                direct = torch.stack(losses).mean()
                if not bool(torch.isfinite(direct)):
                    raise RuntimeError(f"{name}: balanced A2 direct loss is non-finite")
                direct.backward()
                flow_mag_reports.append(float(torch.stack(flow_mags).mean().detach()))
                flow_dir_reports.append(float(torch.stack(flow_dirs).mean().detach()))
                state_reports.append(float(torch.stack(state_losses).mean().detach()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            absolute_reports.append(absolute_group)
            del cpu

        row = {
            "stage": "v128_counterfactual_first_joint_a2_balanced",
            "contract": BALANCED_A2_CONTRACT,
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
            **_coverage_payload(
                selected_actuators=selected_actuators,
                informative_actuators=informative_actuators,
                counts=counts,
            ),
            **_cuda_peak(device),
        }
        history.append(row)
        print("[V128_BALANCED_A2] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        gc.collect()
    return history


def train_counterfactual_first_stage_a_balanced_v128(
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
    """Run the existing A0/A1/A2 logic with balanced informative direct-pair exposure."""
    a0 = train_direct_action_flow_a0_balanced(
        model,
        source_caches=source_caches,
        source_groups=source_groups,
        normalization=normalization,
        graph=graph,
        device=device,
        design=design,
    )
    a1 = train_oracle_hydraulic_a1_balanced(
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
    a2 = train_joint_direct_a2_balanced(
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


__all__ = [
    "BALANCED_A0_CONTRACT",
    "BALANCED_A1_CONTRACT",
    "BALANCED_A2_CONTRACT",
    "BALANCED_STAGE_A_CONTRACT",
    "D2_DIRECT_PAIR_BUDGET",
    "DIRECT_PAIR_SAMPLING_CONTRACT",
    "OTHER_DIRECT_PAIR_BUDGET",
    "select_balanced_direct_specs",
    "train_counterfactual_first_stage_a_balanced_v128",
]
