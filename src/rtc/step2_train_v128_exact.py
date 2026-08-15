"""Exact full within-group V128 pairwise gradients with bounded GPU memory.

A single-pass detached-memory scheme can expose every candidate pair but gives cross-chunk
pairs gradient only through the candidate whose graph is currently live. V128 uses a two-pass
same-parameter-snapshot algorithm:

1. inference/no-grad H360 pass over every candidate to cache smooth TFV deltas;
2. gradient H360 pass over the same candidates in small GPU microbatches;
3. each live candidate is compared with every other cached candidate prediction.

Every unordered candidate pair is visited twice, once with each endpoint live. Directed pair
contributions are divided by the original unordered-pair denominator, reproducing the full
first-order gradient while keeping one H360 autograd microbatch resident at a time.

Truth partitioning is deliberately canonicalized in float32. SWMM node-volume labels are
stored as float32 and live training tensors are float32; re-summing those labels in NumPy
float64 can move near-threshold pairs across the frozen 1 m3 informative floor. Census,
reported pair loss and live candidate pair gradients therefore share the exact same precomputed
float32 candidate-delta tensor.
"""
from __future__ import annotations

import gc
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_train_response_v60 import InputNormalizationV60
from .step2_train_v127 import _ordered, _static, _truth_node_volume
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
    "PROJECT7_V128_H360_EXACT_FULL_PAIRWISE_GRADIENT_TWO_PASS_MICROBATCH_V4_CANONICAL_FLOAT32_TRUTH"
)
V128_TRUTH_PARTITION_CONTRACT = "PROJECT7_V128_CANONICAL_FLOAT32_TRUTH_DELTA_V1"


def canonical_truth_tfv_delta_v128(truth_node_volume_m3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return reference/candidate TFV and candidate deltas using one float32 reduction.

    This array is the single source of truth for all threshold predicates and pairwise targets.
    It intentionally mirrors the precision of cached labels/model losses rather than silently
    promoting one part of the training contract to float64.
    """
    truth = np.asarray(truth_node_volume_m3, dtype=np.float32)
    if truth.ndim != 2 or truth.shape[0] < 2:
        raise ValueError("V128 truth node-volume matrix must be [reference+candidate,node]")
    tfv = np.sum(truth, axis=1, dtype=np.float32).astype(np.float32, copy=False)
    delta = (tfv[1:] - tfv[0]).astype(np.float32, copy=False)
    return tfv, delta


def _informative_pair_totals(
    true_delta: np.ndarray, *, threshold: float
) -> tuple[int, int, int]:
    values = np.asarray(true_delta, dtype=np.float32).reshape(-1)
    threshold32 = np.float32(threshold)
    reference = int(np.sum(np.abs(values) > threshold32))
    if len(values) < 2:
        return reference, 0, reference
    ii, jj = np.triu_indices(len(values), k=1)
    candidate = int(np.sum(np.abs(values[ii] - values[jj]) > threshold32))
    return reference, candidate, reference + candidate


def _candidate_base_loss(
    model: Any,
    *,
    chunk: dict[str, torch.Tensor],
    truth_volume: torch.Tensor,
    reference_volume: torch.Tensor,
    ref_hard_tfv: torch.Tensor,
    ref_smooth_tfv: torch.Tensor,
    ref_smooth_node: torch.Tensor,
    canonical_true_delta: torch.Tensor,
    static: dict[str, torch.Tensor],
    flood_rate_index: int,
    design: V127ControlTrainingDesign,
    tfv_scale: torch.Tensor,
    delta_scale: torch.Tensor,
    node_delta_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    if canonical_true_delta.shape != true_tfv.shape:
        raise RuntimeError("V128 canonical candidate truth delta does not align with live chunk")
    true_delta = canonical_true_delta.to(device=true_tfv.device, dtype=true_tfv.dtype)
    hard_delta = output.tfv_m3 - ref_hard_tfv.detach()
    smooth_delta = output.optimization_tfv_m3 - ref_smooth_tfv.detach()
    hard_delta_loss = F.smooth_l1_loss(
        hard_delta / delta_scale, true_delta / delta_scale, beta=0.5
    )
    smooth_delta_loss = F.smooth_l1_loss(
        smooth_delta / delta_scale, true_delta / delta_scale, beta=0.5
    )
    retained = _retained(true_delta.device)
    pred_states = output.rollout.states.index_select(1, retained)
    state_loss = F.smooth_l1_loss(
        (pred_states - chunk["states"].index_select(1, retained))
        / model.transition.state_std,
        torch.zeros_like(pred_states),
        beta=0.5,
    )
    base_loss = (
        design.node_flood_weight * node_effect
        + design.tfv_weight * (hard_abs + hard_delta_loss + smooth_delta_loss)
        + design.rollout_state_weight * state_loss
    )
    return base_loss, true_delta, smooth_delta


def _first_pass_candidate_predictions(
    model: Any,
    *,
    cpu: dict[str, torch.Tensor],
    candidate_count: int,
    ref_smooth_tfv: torch.Tensor,
    static: dict[str, torch.Tensor],
    flood_rate_index: int,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    """Return smooth candidate deltas in physical branch positions 1..candidate_count."""
    values = torch.empty(candidate_count, dtype=torch.float32, device=device)
    with torch.inference_mode():
        for start in range(1, candidate_count + 1, int(chunk_size)):
            stop = min(start + int(chunk_size), candidate_count + 1)
            positions = np.arange(start, stop, dtype=np.int64)
            chunk = _select_to_device(cpu, positions, device=device, include_truth=False)
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
            values[start - 1 : stop - 1] = (
                output.optimization_tfv_m3 - ref_smooth_tfv.detach()
            ).to(dtype=torch.float32)
            del chunk, output
    return values.detach()


def _exact_reported_pair_loss(
    *,
    truth_delta: torch.Tensor,
    predicted_delta: torch.Tensor,
    threshold: float,
    delta_scale: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    terms: list[torch.Tensor] = []
    informative_ref = torch.abs(truth_delta) > float(threshold)
    ref_count = int(informative_ref.sum().item())
    if bool(informative_ref.any()):
        terms.append(
            F.softplus(
                -torch.sign(truth_delta[informative_ref])
                * predicted_delta[informative_ref]
                / delta_scale
            )
        )
    cand_count = 0
    if len(truth_delta) > 1:
        ii, jj = torch.triu_indices(
            len(truth_delta), len(truth_delta), offset=1, device=truth_delta.device
        )
        truth_pair = truth_delta[ii] - truth_delta[jj]
        pred_pair = predicted_delta[ii] - predicted_delta[jj]
        keep = torch.abs(truth_pair) > float(threshold)
        cand_count = int(keep.sum().item())
        if bool(keep.any()):
            terms.append(
                F.softplus(-torch.sign(truth_pair[keep]) * pred_pair[keep] / delta_scale)
            )
    if not terms:
        return predicted_delta.new_zeros(()), ref_count, cand_count
    return torch.cat([term.reshape(-1) for term in terms]).mean(), ref_count, cand_count


def _directed_pair_gradient_sum(
    *,
    live_truth: torch.Tensor,
    live_pred: torch.Tensor,
    live_positions: np.ndarray,
    all_truth: torch.Tensor,
    all_pred_detached: torch.Tensor,
    threshold: float,
    delta_scale: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    terms: list[torch.Tensor] = []
    ref_keep = torch.abs(live_truth) > float(threshold)
    ref_count = int(ref_keep.sum().item())
    if bool(ref_keep.any()):
        terms.append(
            F.softplus(
                -torch.sign(live_truth[ref_keep]) * live_pred[ref_keep] / delta_scale
            )
        )

    truth_pair = live_truth[:, None] - all_truth[None, :]
    pred_pair = live_pred[:, None] - all_pred_detached[None, :]
    keep = torch.abs(truth_pair) > float(threshold)
    row = torch.arange(len(live_positions), device=live_truth.device)
    col = torch.as_tensor(live_positions - 1, dtype=torch.long, device=live_truth.device)
    keep[row, col] = False
    candidate_directed_count = int(keep.sum().item())
    if bool(keep.any()):
        terms.append(
            F.softplus(-torch.sign(truth_pair[keep]) * pred_pair[keep] / delta_scale)
        )
    if not terms:
        return live_pred.new_zeros(()), ref_count, candidate_directed_count
    return (
        torch.cat([term.reshape(-1) for term in terms]).sum(),
        ref_count,
        candidate_directed_count,
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
) -> list[dict[str, float | int | str | bool]]:
    design.validate()
    static = _static(graph, device)
    model.train().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=design.objective_learning_rate,
        weight_decay=design.weight_decay,
    )
    history: list[dict[str, float | int | str | bool]] = []
    _reset_cuda_peak(device)

    for epoch in range(1, design.objective_epochs + 1):
        records: list[float] = []
        total_candidates = 0
        expected_reference_pairs = 0
        expected_candidate_pairs = 0
        observed_reference_directed = 0
        observed_candidate_directed = 0
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

            true_tfv_np, true_delta_np = canonical_truth_tfv_delta_v128(truth_np)
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
            group_ref_total, group_cand_total, group_pair_total = _informative_pair_totals(
                true_delta_np, threshold=threshold
            )
            expected_reference_pairs += group_ref_total
            expected_candidate_pairs += group_cand_total
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
            ref_truth_tfv = torch.as_tensor(
                true_tfv_np[0:1], dtype=torch.float32, device=device
            )
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
            group_base_report = float(ref_loss.detach())
            del ref_output, ref_loss, ref_hard_loss, ref_state_loss, ref_pred_states

            first_pred = _first_pass_candidate_predictions(
                model,
                cpu=cpu,
                candidate_count=candidate_count,
                ref_smooth_tfv=ref_smooth,
                static=static,
                flood_rate_index=flood_rate_index,
                device=device,
                chunk_size=design.objective_candidate_chunk,
            )
            all_truth = torch.as_tensor(true_delta_np, dtype=torch.float32, device=device)
            pair_report, report_ref, report_cand = _exact_reported_pair_loss(
                truth_delta=all_truth,
                predicted_delta=first_pred,
                threshold=threshold,
                delta_scale=delta_scale,
            )
            if report_ref != group_ref_total or report_cand != group_cand_total:
                raise RuntimeError(f"{name}: V128 first-pass pair census mismatch")

            order = _candidate_permutation(
                candidate_count, group_name=name, epoch=epoch, seed=design.seed
            )
            group_ref_seen = 0
            group_cand_directed_seen = 0
            group_base_candidate_report = 0.0
            for start in range(0, candidate_count, design.objective_candidate_chunk):
                positions = order[start : start + design.objective_candidate_chunk]
                chunk = _select_to_device(cpu, positions, device=device, include_truth=True)
                truth_chunk = torch.as_tensor(
                    truth_np[positions], dtype=torch.float32, device=device
                )
                canonical_chunk_delta = all_truth.index_select(
                    0,
                    torch.as_tensor(
                        np.asarray(positions, dtype=np.int64) - 1,
                        dtype=torch.long,
                        device=device,
                    ),
                )
                base_loss, live_truth, live_pred = _candidate_base_loss(
                    model,
                    chunk=chunk,
                    truth_volume=truth_chunk,
                    reference_volume=ref_volume,
                    ref_hard_tfv=ref_hard,
                    ref_smooth_tfv=ref_smooth,
                    ref_smooth_node=ref_smooth_node,
                    canonical_true_delta=canonical_chunk_delta,
                    static=static,
                    flood_rate_index=flood_rate_index,
                    design=design,
                    tfv_scale=tfv_scale,
                    delta_scale=delta_scale,
                    node_delta_scale=node_delta_scale,
                )
                directed_sum, ref_count, cand_directed_count = _directed_pair_gradient_sum(
                    live_truth=live_truth,
                    live_pred=live_pred,
                    live_positions=np.asarray(positions, dtype=np.int64),
                    all_truth=all_truth,
                    all_pred_detached=first_pred,
                    threshold=threshold,
                    delta_scale=delta_scale,
                )
                candidate_weight = float(len(positions)) / float(candidate_count)
                backward_loss = base_loss * candidate_weight
                if group_pair_total > 0:
                    backward_loss = backward_loss + design.pairwise_weight * (
                        directed_sum / float(group_pair_total)
                    )
                if not bool(torch.isfinite(backward_loss)):
                    raise RuntimeError(f"{name}: V128 candidate objective loss non-finite")
                backward_loss.backward()
                group_base_candidate_report += float(base_loss.detach()) * candidate_weight
                group_ref_seen += int(ref_count)
                group_cand_directed_seen += int(cand_directed_count)
                del (
                    chunk,
                    truth_chunk,
                    canonical_chunk_delta,
                    base_loss,
                    live_truth,
                    live_pred,
                    directed_sum,
                    backward_loss,
                )

            if group_ref_seen != group_ref_total:
                raise RuntimeError(
                    f"{name}: V128 reference-pair gradient coverage "
                    f"{group_ref_seen}/{group_ref_total}"
                )
            if group_cand_directed_seen != 2 * group_cand_total:
                raise RuntimeError(
                    f"{name}: V128 candidate-pair directed gradient coverage "
                    f"{group_cand_directed_seen}/{2 * group_cand_total}"
                )
            observed_reference_directed += group_ref_seen
            observed_candidate_directed += group_cand_directed_seen
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()

            group_report = (
                group_base_report
                + group_base_candidate_report
                + design.pairwise_weight * float(pair_report.detach())
            )
            records.append(group_report)
            total_candidates += candidate_count
            del cpu, ref, ref_volume, first_pred, all_truth, pair_report

        if observed_reference_directed != expected_reference_pairs:
            raise RuntimeError("V128 epoch reference-pair gradient coverage is incomplete")
        if observed_candidate_directed != 2 * expected_candidate_pairs:
            raise RuntimeError("V128 epoch candidate-pair gradient coverage is incomplete")

        row: dict[str, float | int | str | bool] = {
            "stage": "h360_v128_exact_full_pairwise_gradient_two_pass_microbatch",
            "contract": V128_OBJECTIVE_TRAINING_CONTRACT,
            "truth_partition_contract": V128_TRUTH_PARTITION_CONTRACT,
            "epoch": epoch,
            "loss": float(np.mean(records)) if records else float("nan"),
            "candidate_total": int(total_candidates),
            "candidate_seen_unique": int(total_candidates),
            "candidate_coverage_fraction": 1.0 if total_candidates else 0.0,
            "objective_candidate_chunk": int(design.objective_candidate_chunk),
            "informative_reference_pairs": int(expected_reference_pairs),
            "informative_candidate_pairs_unordered": int(expected_candidate_pairs),
            "candidate_pair_possible_unordered": int(possible_candidate_pairs),
            "candidate_pair_directed_gradient_terms": int(observed_candidate_directed),
            "candidate_pair_gradient_coverage": 1.0,
            "pairwise_first_derivative_exact_under_same_parameter_snapshot": True,
            "pairwise_reported_scalar_not_doubled": True,
            "pairwise_training_passes_per_candidate": 2,
            "canonical_float32_truth_delta_shared": True,
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
    "V128_TRUTH_PARTITION_CONTRACT",
    "_informative_pair_totals",
    "canonical_truth_tfv_delta_v128",
    "train_objective_stage_streaming_v128",
]
