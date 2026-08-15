"""V128 control-identification training extensions.

The V127 streaming objective is memory-safe on an 8-GB GPU, but candidate-candidate
pairwise supervision is formed only inside each small GPU microbatch.  With 24 candidates
and a chunk of two, most within-state action-order pairs never contribute a direct ranking
term in an epoch.

V128 keeps the same small H360 microbatch but stores each completed candidate's smooth TFV
prediction as a detached scalar.  Later microbatches compare their live predictions with
all earlier detached predictions.  Because the optimizer is stepped only after the whole
counterfactual group, those detached values were produced by the same parameter snapshot.
This yields full within-group candidate-pair ranking coverage without retaining earlier
H360 autograd graphs in VRAM.
"""
from __future__ import annotations

import gc
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_train_response_v60 import InputNormalizationV60
from .step2_train_v127 import _branch_indices, _ordered, _static, _truth_node_volume
from .step2_train_v127_control import (
    V127ControlTrainingDesign,
    _candidate_permutation,
    informative_pair_threshold_v127,
)
from .step2_train_v127_streaming import (
    _cpu_group,
    _cuda_peak,
    _reset_cuda_peak,
    _retained,
    _select_to_device,
)

V128_OBJECTIVE_TRAINING_CONTRACT = (
    "PROJECT7_V128_H360_FULL_WITHIN_GROUP_RANKING_CPU_STREAM_GPU_MICROBATCH_V1"
)


def _candidate_loss_v128(
    model: Any,
    *,
    chunk: dict[str, torch.Tensor],
    truth_volume: torch.Tensor,
    reference_volume: torch.Tensor,
    ref_hard_tfv: torch.Tensor,
    ref_smooth_tfv: torch.Tensor,
    ref_smooth_node: torch.Tensor,
    static: dict[str, torch.Tensor],
    flood_rate_index: int,
    design: V127ControlTrainingDesign,
    tfv_scale: torch.Tensor,
    delta_scale: torch.Tensor,
    node_delta_scale: torch.Tensor,
    informative_threshold: float,
    previous_truth_delta: torch.Tensor | None,
    previous_smooth_delta: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    output = model.objective_rollout(
        initial_state=chunk["initial"],
        rainfall=chunk["rainfall"],
        settings=chunk["settings"],
        previous_actuator_flow=chunk["previous_flow"],
        actuator_upstream=static["up"],
        actuator_downstream=static["down"],
        actuator_physics=static["physics"],
        static_node_features=static["static"],
        edge_index=static["edges"],
        flood_rate_index=flood_rate_index,
        priority_indices=None,
        dt_seconds=300.0,
    )
    true_tfv = truth_volume.sum(-1)
    hard_abs = F.smooth_l1_loss(
        (output.tfv_m3 - true_tfv) / tfv_scale,
        torch.zeros_like(true_tfv),
        beta=0.5,
    )
    true_delta_node = truth_volume - reference_volume
    smooth_delta_node = output.optimization_node_flood_volume_m3 - ref_smooth_node.detach()
    node_effect = F.smooth_l1_loss(
        smooth_delta_node / node_delta_scale,
        true_delta_node / node_delta_scale,
        beta=0.5,
    )
    true_delta = true_tfv - reference_volume.sum()
    hard_delta = output.tfv_m3 - ref_hard_tfv.detach()
    smooth_delta = output.optimization_tfv_m3 - ref_smooth_tfv.detach()
    hard_delta_loss = F.smooth_l1_loss(
        hard_delta / delta_scale, true_delta / delta_scale, beta=0.5
    )
    smooth_delta_loss = F.smooth_l1_loss(
        smooth_delta / delta_scale, true_delta / delta_scale, beta=0.5
    )

    pair_terms: list[torch.Tensor] = []
    reference_pairs = 0
    candidate_pairs = 0
    informative = torch.abs(true_delta) > float(informative_threshold)
    reference_pairs += int(informative.sum().item())
    if bool(informative.any()):
        pair_terms.append(
            F.softplus(
                -torch.sign(true_delta[informative])
                * smooth_delta[informative]
                / delta_scale
            )
        )

    # Pairs within the current GPU microbatch keep gradients on both candidates.
    if len(true_delta) > 1:
        ii, jj = torch.triu_indices(
            len(true_delta), len(true_delta), offset=1, device=true_delta.device
        )
        truth_pair = true_delta[ii] - true_delta[jj]
        pred_pair = smooth_delta[ii] - smooth_delta[jj]
        keep = torch.abs(truth_pair) > float(informative_threshold)
        candidate_pairs += int(keep.sum().item())
        if bool(keep.any()):
            pair_terms.append(
                F.softplus(-torch.sign(truth_pair[keep]) * pred_pair[keep] / delta_scale)
            )

    # Cross-microbatch pairs: previous predictions are detached so VRAM does not scale with
    # group candidate count. Parameters have not stepped yet, therefore these are not stale
    # model predictions; only the earlier autograd graphs are intentionally discarded.
    if previous_truth_delta is not None and previous_smooth_delta is not None:
        truth_pair = true_delta[:, None] - previous_truth_delta[None, :]
        pred_pair = smooth_delta[:, None] - previous_smooth_delta.detach()[None, :]
        keep = torch.abs(truth_pair) > float(informative_threshold)
        candidate_pairs += int(keep.sum().item())
        if bool(keep.any()):
            pair_terms.append(
                F.softplus(-torch.sign(truth_pair[keep]) * pred_pair[keep] / delta_scale)
            )

    pair_loss = (
        torch.cat([term.reshape(-1) for term in pair_terms]).mean()
        if pair_terms
        else smooth_delta.new_zeros(())
    )
    retained = _retained(true_delta.device)
    pred_states = output.rollout.states.index_select(1, retained)
    state_loss = F.smooth_l1_loss(
        (pred_states - chunk["states"].index_select(1, retained))
        / model.transition.state_std,
        torch.zeros_like(pred_states),
        beta=0.5,
    )
    loss = (
        design.node_flood_weight * node_effect
        + design.tfv_weight * (hard_abs + hard_delta_loss + smooth_delta_loss)
        + design.pairwise_weight * pair_loss
        + design.rollout_state_weight * state_loss
    )
    return (
        loss,
        true_delta.detach(),
        smooth_delta.detach(),
        reference_pairs,
        candidate_pairs,
    )


def train_objective_stage_streaming_v128(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    flood_rate_index: int,
    design: V127ControlTrainingDesign,
) -> list[dict[str, float | int | str]]:
    """H360 objective with full within-state ranking and bounded GPU memory."""
    design.validate()
    static = _static(graph, device)
    model.train().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=design.objective_learning_rate,
        weight_decay=design.weight_decay,
    )
    history: list[dict[str, float | int | str]] = []
    _reset_cuda_peak(device)

    for epoch in range(1, design.objective_epochs + 1):
        records: list[float] = []
        total_candidates = 0
        reference_pair_terms = 0
        candidate_pair_terms = 0
        possible_candidate_pairs = 0
        for source, name in _ordered(source_groups, epoch + 17, design.seed):
            cpu = _cpu_group(source_caches[source], name, normalization)
            truth_np = _truth_node_volume(source_caches[source], name).astype(np.float32)
            branches = int(cpu["settings"].shape[0])
            if truth_np.shape[0] != branches:
                raise RuntimeError(f"{name}: SWMM label count does not match branch order")
            candidate_count = branches - 1
            if candidate_count <= 0:
                raise RuntimeError(f"{name}: objective group has no candidate")
            possible_candidate_pairs += candidate_count * (candidate_count - 1) // 2

            true_tfv_np = truth_np.sum(axis=1, dtype=np.float64)
            true_delta_np = true_tfv_np[1:] - true_tfv_np[0]
            tfv_scale = torch.tensor(
                max(float(np.quantile(true_tfv_np, 0.75)), 100.0),
                dtype=torch.float32,
                device=device,
            )
            delta_scale = torch.tensor(
                max(float(np.quantile(np.abs(true_delta_np), 0.75)), 100.0),
                dtype=torch.float32,
                device=device,
            )
            node_delta_np = truth_np[1:] - truth_np[0:1]
            node_delta_scale = torch.tensor(
                max(float(np.quantile(np.abs(node_delta_np).reshape(-1), 0.75)), 25.0),
                dtype=torch.float32,
                device=device,
            )
            threshold = informative_pair_threshold_v127(float(true_tfv_np[0]), design)
            optimizer.zero_grad(set_to_none=True)

            ref = _select_to_device(cpu, [0], device=device, include_truth=True)
            ref_volume = torch.as_tensor(truth_np[0:1], dtype=torch.float32, device=device)
            ref_output = model.objective_rollout(
                initial_state=ref["initial"],
                rainfall=ref["rainfall"],
                settings=ref["settings"],
                previous_actuator_flow=ref["previous_flow"],
                actuator_upstream=static["up"],
                actuator_downstream=static["down"],
                actuator_physics=static["physics"],
                static_node_features=static["static"],
                edge_index=static["edges"],
                flood_rate_index=flood_rate_index,
                priority_indices=None,
                dt_seconds=300.0,
            )
            ref_truth_tfv = ref_volume.sum(-1)
            ref_hard_loss = F.smooth_l1_loss(
                (ref_output.tfv_m3 - ref_truth_tfv) / tfv_scale,
                torch.zeros_like(ref_truth_tfv),
                beta=0.5,
            )
            retained = _retained(device)
            ref_pred_states = ref_output.rollout.states.index_select(1, retained)
            ref_state_loss = F.smooth_l1_loss(
                (ref_pred_states - ref["states"].index_select(1, retained))
                / model.transition.state_std,
                torch.zeros_like(ref_pred_states),
                beta=0.5,
            )
            ref_loss = ref_hard_loss + design.rollout_state_weight * ref_state_loss
            if not bool(torch.isfinite(ref_loss)):
                raise RuntimeError(f"{name}: V128 reference objective loss non-finite")
            ref_loss.backward()
            ref_hard = ref_output.tfv_m3.detach()[0]
            ref_smooth = ref_output.optimization_tfv_m3.detach()[0]
            ref_smooth_node = ref_output.optimization_node_flood_volume_m3.detach()[0:1]
            group_loss = float(ref_loss.detach())
            del ref_output, ref_loss, ref_hard_loss, ref_state_loss, ref_pred_states

            previous_truth: list[torch.Tensor] = []
            previous_pred: list[torch.Tensor] = []
            order = _candidate_permutation(
                candidate_count, group_name=name, epoch=epoch, seed=design.seed
            )
            for start in range(0, candidate_count, design.objective_candidate_chunk):
                positions = order[start : start + design.objective_candidate_chunk]
                chunk = _select_to_device(cpu, positions, device=device, include_truth=True)
                truth_chunk = torch.as_tensor(
                    truth_np[positions], dtype=torch.float32, device=device
                )
                memory_truth = (
                    torch.cat(previous_truth)
                    if previous_truth
                    else None
                )
                memory_pred = (
                    torch.cat(previous_pred)
                    if previous_pred
                    else None
                )
                loss, true_delta, smooth_delta, ref_pairs, cand_pairs = _candidate_loss_v128(
                    model,
                    chunk=chunk,
                    truth_volume=truth_chunk,
                    reference_volume=ref_volume,
                    ref_hard_tfv=ref_hard,
                    ref_smooth_tfv=ref_smooth,
                    ref_smooth_node=ref_smooth_node,
                    static=static,
                    flood_rate_index=flood_rate_index,
                    design=design,
                    tfv_scale=tfv_scale,
                    delta_scale=delta_scale,
                    node_delta_scale=node_delta_scale,
                    informative_threshold=threshold,
                    previous_truth_delta=memory_truth,
                    previous_smooth_delta=memory_pred,
                )
                weight = float(len(positions)) / float(candidate_count)
                weighted = loss * weight
                if not bool(torch.isfinite(weighted)):
                    raise RuntimeError(f"{name}: V128 candidate objective loss non-finite")
                weighted.backward()
                group_loss += float(weighted.detach())
                reference_pair_terms += int(ref_pairs)
                candidate_pair_terms += int(cand_pairs)
                previous_truth.append(true_delta)
                previous_pred.append(smooth_delta)
                del chunk, truth_chunk, loss, weighted, true_delta, smooth_delta
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            records.append(group_loss)
            total_candidates += candidate_count
            del cpu, ref, ref_volume, previous_truth, previous_pred

        row: dict[str, float | int | str] = {
            "stage": "h360_v128_full_within_group_ranking_cpu_stream_gpu_microbatch",
            "contract": V128_OBJECTIVE_TRAINING_CONTRACT,
            "epoch": epoch,
            "loss": float(np.mean(records)) if records else float("nan"),
            "candidate_total": int(total_candidates),
            "candidate_seen_unique": int(total_candidates),
            "candidate_coverage_fraction": 1.0 if total_candidates else 0.0,
            "objective_candidate_chunk": int(design.objective_candidate_chunk),
            "informative_reference_pair_terms": int(reference_pair_terms),
            "informative_candidate_pair_terms": int(candidate_pair_terms),
            "candidate_pair_partition_coverage": (
                float(candidate_pair_terms) / float(possible_candidate_pairs)
                if possible_candidate_pairs
                else 0.0
            ),
            "cross_microbatch_pairwise_memory": "detached_same_parameter_snapshot",
            "ranking_threshold": "absolute SWMM effect floor from V128 training design",
            **_cuda_peak(device),
        }
        if not math.isfinite(float(row["loss"])):
            raise RuntimeError("V128 H360 objective produced no finite loss")
        history.append(row)
        print(
            "[V128_STREAM_OBJECTIVE] "
            + " ".join(f"{key}={value}" for key, value in row.items()),
            flush=True,
        )
        gc.collect()
    return history


__all__ = [
    "V128_OBJECTIVE_TRAINING_CONTRACT",
    "train_objective_stage_streaming_v128",
]
