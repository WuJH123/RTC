"""V128 teacher-forced hydraulic training with the same typed action path used online."""
from __future__ import annotations

import gc
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_train_response_v60 import InputNormalizationV60
from .step2_train_v127 import _ordered, _static
from .step2_train_v127_control import V127ControlTrainingDesign, _state_weights
from .step2_train_v127_streaming import (
    _cpu_group,
    _cuda_peak,
    _reset_cuda_peak,
    _select_to_device,
)

V128_HYDRAULIC_TRAINING_CONTRACT = (
    "PROJECT7_V128_TYPED_ACTION_TEACHER_FORCED_HYDRAULIC_STREAMING_V1"
)


def train_hydraulic_stage_streaming_v128(
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
) -> list[dict[str, float | int | str]]:
    """Teacher force V128 through its typed actuator-message transition.

    This must not call the inherited V127 ``_setting_context`` helper. Training and online
    rollout use the identical V128 endpoint-state/setting/flow/physics/identity message.
    """
    if not hasattr(model, "_typed_action_context"):
        raise TypeError("V128 hydraulic trainer requires typed actuator-message model")
    design.validate()
    torch.manual_seed(design.seed)
    np.random.seed(design.seed)
    model.train().to(device)
    static = _static(graph, device)
    weights = _state_weights(
        model,
        depth_index=depth_index,
        flood_rate_index=flood_rate_index,
        design=design,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=design.learning_rate, weight_decay=design.weight_decay
    )
    history: list[dict[str, float | int | str]] = []
    phase_seen: dict[str, set[int]] = {
        name: set() for values in source_groups.values() for name in values
    }
    _reset_cuda_peak(device)

    for epoch in range(1, design.hydraulic_epochs + 1):
        phase = (epoch - 1) % design.teacher_stride
        losses: list[float] = []
        by_source: dict[str, list[float]] = {key: [] for key in source_caches}
        transition_count = 0
        for source, name in _ordered(source_groups, epoch, design.seed):
            cpu = _cpu_group(source_caches[source], name, normalization)
            branches, horizon = cpu["settings"].shape[:2]
            steps = len(range(phase, horizon, design.teacher_stride))
            if steps <= 0:
                raise RuntimeError("V128 hydraulic stage selected no transitions")
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            phase_seen[name].add(phase)
            transition_count += int(steps * branches)

            for start in range(0, branches, design.hydraulic_branch_chunk):
                stop = min(start + design.hydraulic_branch_chunk, branches)
                positions = np.arange(start, stop, dtype=np.int64)
                chunk = _select_to_device(
                    cpu, positions, device=device, include_truth=True
                )
                chunk_n = len(positions)
                physics_norm, identity = model.actuator.prepare_static(
                    static["physics"], batch_size=chunk_n
                )
                static_norm, edges, inv = model.transition.prepare_static(
                    static["static"],
                    static["edges"],
                    batch_size=chunk_n,
                    dtype=torch.float32,
                )
                chunk_loss = torch.zeros((), device=device)
                for k in range(phase, horizon, design.teacher_stride):
                    prev_state = (
                        chunk["initial"] if k == 0 else chunk["states"][:, k - 1]
                    )
                    prev_flow = (
                        chunk["previous_flow"] if k == 0 else chunk["flows"][:, k - 1]
                    )
                    setting = chunk["settings"][:, k]
                    q, response = model.actuator.forward_prepared(
                        prev_state[:, static["up"]],
                        prev_state[:, static["down"]],
                        setting,
                        prev_flow,
                        physics_norm,
                        identity,
                    )
                    injection = torch.zeros(
                        chunk_n,
                        prev_state.shape[1],
                        1,
                        device=device,
                        dtype=prev_state.dtype,
                    )
                    injection = injection.index_add(
                        1, static["up"], -q[..., None]
                    ).index_add(1, static["down"], q[..., None])
                    action_context = model._typed_action_context(
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
                    pred_state = model.transition.forward_prepared(
                        prev_state,
                        chunk["rainfall"][:, k],
                        static_norm,
                        injection,
                        edges,
                        inv,
                        action_context,
                    )
                    state_error = (
                        pred_state - chunk["states"][:, k]
                    ) / model.transition.state_std
                    flow_error = (q - chunk["flows"][:, k]) / model.actuator.flow_std
                    chunk_loss = chunk_loss + F.smooth_l1_loss(
                        state_error * weights,
                        torch.zeros_like(state_error),
                        beta=0.5,
                    )
                    chunk_loss = chunk_loss + design.flow_weight * F.smooth_l1_loss(
                        flow_error, torch.zeros_like(flow_error), beta=0.5
                    )
                weighted = (chunk_loss / steps) * (float(chunk_n) / float(branches))
                if not bool(torch.isfinite(weighted)):
                    raise RuntimeError(f"{name}: V128 hydraulic loss non-finite")
                weighted.backward()
                group_loss += float(weighted.detach())
                del chunk, chunk_loss, weighted
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            losses.append(group_loss)
            by_source[source].append(group_loss)
            del cpu

        row: dict[str, float | int | str] = {
            "stage": "v128_typed_hydraulic_teacher_forced_cpu_stream_gpu_chunk",
            "contract": V128_HYDRAULIC_TRAINING_CONTRACT,
            "epoch": epoch,
            "teacher_phase": phase,
            "loss": float(np.mean(losses)),
            "hydraulic_branch_chunk": int(design.hydraulic_branch_chunk),
            "teacher_transitions": int(transition_count),
            "min_group_teacher_phase_coverage": min(
                (len(values) / design.teacher_stride for values in phase_seen.values()),
                default=0.0,
            ),
            "typed_action_context_used": True,
            **_cuda_peak(device),
        }
        for source, values in by_source.items():
            if values:
                row[f"loss_{source.lower()}"] = float(np.mean(values))
        history.append(row)
        print(
            "[V128_STREAM_HYDRAULIC] "
            + " ".join(f"{key}={value}" for key, value in row.items()),
            flush=True,
        )
        gc.collect()
    if any(len(values) < design.teacher_stride for values in phase_seen.values()):
        raise RuntimeError("V128 hydraulic curriculum missed a teacher phase")
    return history


__all__ = [
    "V128_HYDRAULIC_TRAINING_CONTRACT",
    "train_hydraulic_stage_streaming_v128",
]
