"""Actuator-only FIT warm-up for the current action-identifiable V128 repair.

The diagnosed failure begins before the full hydraulic transition: held-out managed-flow effect
signs are often correct while their magnitude is nearly zero.  The warm-up therefore gives the
actuator submodel a short, cheap optimization phase before it competes with the 932-node graph
transition for gradient budget.

Only TrainFit groups supplied by the current runner are used.  The transition and action-message
network are not evaluated here.  Absolute managed-flow prediction and candidate-minus-reference
managed-flow effects are both supervised using authoritative SWMM cache values.
"""
from __future__ import annotations

import gc
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_action_identifiable_v128 import (
    ACTION_IDENTIFIABLE_TRAINING_CONTRACT,
    _response_weighted_effect_loss,
    train_action_identifiable_hydraulic_stage_v128,
)
from .step2_train_response_v60 import InputNormalizationV60
from .step2_train_v127 import _ordered, _static
from .step2_train_v127_control import V127ControlTrainingDesign, _candidate_permutation
from .step2_train_v127_streaming import (
    _cpu_group,
    _cuda_peak,
    _reset_cuda_peak,
    _select_to_device,
)

ACTION_FLOW_WARMUP_CONTRACT = "PROJECT7_V128_ACTUATOR_ONLY_COUNTERFACTUAL_FLOW_WARMUP_DEV_V1"


def _predict_teacher_forced_flow(
    model: Any,
    *,
    chunk: dict[str, torch.Tensor],
    k: int,
    static: dict[str, torch.Tensor],
    physics_norm: torch.Tensor,
    identity: torch.Tensor | None,
) -> torch.Tensor:
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
    return q


def pretrain_action_flow_warmup_v128(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    design: V127ControlTrainingDesign,
) -> dict[str, float | int | str | bool]:
    """Run one deterministic FIT-only actuator optimization pass."""
    design.validate()
    torch.manual_seed(design.seed + 1701)
    np.random.seed(design.seed + 1701)
    model.train().to(device)
    static = _static(graph, device)
    optimizer = torch.optim.AdamW(
        model.actuator.parameters(),
        lr=float(design.learning_rate),
        weight_decay=float(design.weight_decay),
    )
    absolute_reports: list[float] = []
    effect_reports: list[float] = []
    group_reports: list[float] = []
    transition_count = 0
    selected_pairs = 0
    _reset_cuda_peak(device)

    for source, name in _ordered(source_groups, 1701, design.seed):
        cpu = _cpu_group(source_caches[source], name, normalization)
        branches, horizon = cpu["settings"].shape[:2]
        optimizer.zero_grad(set_to_none=True)
        absolute_group = 0.0

        # Full branch coverage for the cheap actuator-only absolute-flow term.
        for start in range(0, branches, design.hydraulic_branch_chunk):
            stop = min(start + design.hydraulic_branch_chunk, branches)
            positions = np.arange(start, stop, dtype=np.int64)
            chunk = _select_to_device(cpu, positions, device=device, include_truth=True)
            physics_norm, identity = model.actuator.prepare_static(
                static["physics"], batch_size=len(positions)
            )
            loss = torch.zeros((), device=device)
            for k in range(horizon):
                q = _predict_teacher_forced_flow(
                    model,
                    chunk=chunk,
                    k=k,
                    static=static,
                    physics_norm=physics_norm,
                    identity=identity,
                )
                flow_error = (q - chunk["flows"][:, k]) / model.actuator.flow_std
                loss = loss + F.smooth_l1_loss(
                    flow_error, torch.zeros_like(flow_error), beta=0.5
                )
            weighted = (loss / float(horizon)) * (float(len(positions)) / float(branches))
            if not bool(torch.isfinite(weighted)):
                raise RuntimeError(f"{name}: actuator flow warm-up absolute loss non-finite")
            weighted.backward()
            absolute_group += float(weighted.detach())
            transition_count += int(len(positions) * horizon)
            del chunk, loss, weighted

        # A rotating bounded candidate subset supplies the direct counterfactual flow effect.
        candidate_count = branches - 1
        pair_budget = min(8 if str(source).upper() == "D2" else 4, candidate_count)
        effect_loss = torch.zeros((), device=device)
        if pair_budget > 0:
            candidates = _candidate_permutation(
                candidate_count,
                group_name=name,
                epoch=1701,
                seed=design.seed,
            )[:pair_budget]
            positions = np.concatenate((np.asarray([0], dtype=np.int64), candidates))
            pair = _select_to_device(cpu, positions, device=device, include_truth=True)
            physics_norm, identity = model.actuator.prepare_static(
                static["physics"], batch_size=len(positions)
            )
            terms: list[torch.Tensor] = []
            for k in range(horizon):
                q = _predict_teacher_forced_flow(
                    model,
                    chunk=pair,
                    k=k,
                    static=static,
                    physics_norm=physics_norm,
                    identity=identity,
                )
                terms.append(
                    _response_weighted_effect_loss(
                        q[1:] - q[0:1],
                        pair["flows"][1:, k] - pair["flows"][0:1, k],
                        model.actuator.delta_flow_scale,
                    )
                )
            effect_loss = torch.stack(terms).mean()
            if not bool(torch.isfinite(effect_loss)):
                raise RuntimeError(f"{name}: actuator flow warm-up effect loss non-finite")
            effect_loss.backward()
            selected_pairs += int(pair_budget)
            del pair, terms

        torch.nn.utils.clip_grad_norm_(model.actuator.parameters(), design.grad_clip)
        optimizer.step()
        absolute_reports.append(absolute_group)
        effect_reports.append(float(effect_loss.detach()))
        group_reports.append(absolute_group + float(effect_loss.detach()))
        del cpu, effect_loss

    row: dict[str, float | int | str | bool] = {
        "stage": "v128_actuator_only_action_flow_warmup",
        "contract": ACTION_FLOW_WARMUP_CONTRACT,
        "parent_training_contract": ACTION_IDENTIFIABLE_TRAINING_CONTRACT,
        "loss": float(np.mean(group_reports)) if group_reports else 0.0,
        "absolute_flow_loss": float(np.mean(absolute_reports)) if absolute_reports else 0.0,
        "delta_flow_effect_loss": float(np.mean(effect_reports)) if effect_reports else 0.0,
        "teacher_forced_flow_transitions": int(transition_count),
        "selected_counterfactual_pairs": int(selected_pairs),
        "fit_only": True,
        "transition_used": False,
        "action_message_network_used": False,
        **_cuda_peak(device),
    }
    print(
        "[V128_ACTION_FLOW_WARMUP] "
        + " ".join(f"{key}={value}" for key, value in row.items()),
        flush=True,
    )
    gc.collect()
    return row


def train_hydraulic_stage_with_flow_warmup_v128(
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
    warmup = pretrain_action_flow_warmup_v128(
        model,
        source_caches=source_caches,
        source_groups=source_groups,
        normalization=normalization,
        graph=graph,
        device=device,
        design=design,
    )
    joint = train_action_identifiable_hydraulic_stage_v128(
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
    return [warmup, *joint]


__all__ = [
    "ACTION_FLOW_WARMUP_CONTRACT",
    "pretrain_action_flow_warmup_v128",
    "train_hydraulic_stage_with_flow_warmup_v128",
]
