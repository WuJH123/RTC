from __future__ import annotations

from dataclasses import dataclass
import gc
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_train_response_v60 import InputNormalizationV60
from .step2_train_v127 import (
    _branch_indices,
    _ordered,
    _spearman,
    _static,
    _truth_node_volume,
)
from .step2_train_v127_control import (
    V127ControlTrainingDesign,
    _candidate_permutation,
    _pairwise_accuracy,
    _state_weights,
    informative_pair_threshold_v127,
)

V127_STREAMING_TRAINING_CONTRACT = (
    "PROJECT7_V127_CONTROL_STREAMING_V1_CPU_GROUP_GPU_MICROBATCH"
)


@dataclass(frozen=True)
class V127StreamingMemoryDesign:
    """Execution-only memory budget for a 16-GB RAM / 8-GB VRAM workstation."""

    hydraulic_branch_chunk: int = 4
    rollout_candidates_per_group: int = 2
    objective_candidate_chunk: int = 2
    evaluation_branch_chunk: int = 4
    residual_sample_rows: int = 131_072

    def validate(self) -> None:
        for value in (
            self.hydraulic_branch_chunk,
            self.rollout_candidates_per_group,
            self.objective_candidate_chunk,
            self.evaluation_branch_chunk,
            self.residual_sample_rows,
        ):
            if int(value) <= 0:
                raise ValueError("V127 streaming memory budgets must be positive")


def _cpu_group(
    cache: Any,
    name: str,
    normalization: InputNormalizationV60,
) -> dict[str, torch.Tensor]:
    """Load one counterfactual group on CPU only, keeping shared inputs unexpanded."""

    batch = cache.batch(name, normalization, torch.device("cpu"))
    dtype = batch.initial_state.dtype

    def tensor(value: Any) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device="cpu")

    initial = (
        batch.initial_state * tensor(normalization.state_std).clamp_min(1.0e-6)
        + tensor(normalization.state_mean)
    )
    rainfall = (
        batch.rainfall * tensor(normalization.rainfall_std).clamp_min(1.0e-6)
        + tensor(normalization.rainfall_mean)
    )
    previous_flow = (
        batch.previous_actuator_flow * tensor(normalization.flow_std).clamp_min(1.0e-6)
        + tensor(normalization.flow_mean)
    )
    ref_states, cand_states = batch.true_reference_states, batch.true_candidate_states
    ref_flows, cand_flows = batch.true_reference_flows, batch.true_candidate_flows
    if ref_states.ndim == cand_states.ndim - 1:
        ref_states = ref_states[:, None]
    if ref_flows.ndim == cand_flows.ndim - 1:
        ref_flows = ref_flows[:, None]
    states = torch.cat((ref_states, cand_states), dim=1)[0]
    flows = torch.cat((ref_flows, cand_flows), dim=1)[0]
    settings = torch.cat((batch.reference_settings[:, None], batch.candidate_settings), dim=1)[0]
    branches = int(settings.shape[0])
    if states.shape[0] != branches or flows.shape[0] != branches:
        raise RuntimeError("V127 streaming group branch tensors are misaligned")
    return {
        "initial": initial,
        "rainfall": rainfall,
        "previous_flow": previous_flow,
        "states": states,
        "flows": flows,
        "settings": settings,
    }


def _select_to_device(
    data: dict[str, torch.Tensor],
    positions: Sequence[int] | np.ndarray,
    *,
    device: torch.device,
    horizon: int | None = None,
    include_truth: bool = True,
) -> dict[str, torch.Tensor]:
    pos = torch.as_tensor(np.asarray(positions, dtype=np.int64), dtype=torch.long, device="cpu")
    if pos.numel() <= 0:
        raise ValueError("V127 streaming selection is empty")
    h = int(data["settings"].shape[1] if horizon is None else horizon)
    if h <= 0 or h > int(data["settings"].shape[1]):
        raise ValueError("V127 streaming horizon is invalid")
    n = int(pos.numel())
    initial = data["initial"].to(device).expand(n, -1, -1)
    rainfall = data["rainfall"][:, :h].to(device).expand(n, -1, -1, -1)
    previous = data["previous_flow"].to(device).expand(n, -1)
    result = {
        "initial": initial,
        "rainfall": rainfall,
        "previous_flow": previous,
        "settings": data["settings"].index_select(0, pos)[:, :h].to(device),
    }
    if include_truth:
        result["states"] = data["states"].index_select(0, pos)[:, :h].to(device)
        result["flows"] = data["flows"].index_select(0, pos)[:, :h].to(device)
    return result


def _reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _cuda_peak(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"cuda_peak_allocated_gb": 0.0, "cuda_peak_reserved_gb": 0.0}
    gib = float(1024**3)
    return {
        "cuda_peak_allocated_gb": float(torch.cuda.max_memory_allocated(device) / gib),
        "cuda_peak_reserved_gb": float(torch.cuda.max_memory_reserved(device) / gib),
    }


def derive_residual_scales_streaming_v127(
    caches_and_names: Sequence[tuple[Any, Sequence[str]]],
    *,
    sample_rows: int = 131_072,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Approximate the frozen 0.995 residual scales with a bounded deterministic sample.

    Historical code sampled up to 16k state rows *per branch* and concatenated all samples,
    so RAM scaled with the full D2-D4 population. Here the total sample budget is global.
    Each branch contributes an equal deterministic set of time/node rows directly from mmap.
    """

    if int(sample_rows) <= 0:
        raise ValueError("V127 streaming residual sample budget must be positive")
    entries: list[tuple[Any, str]] = []
    total_branches = 0
    for cache, names in caches_and_names:
        for name in names:
            entries.append((cache, str(name)))
            total_branches += len(_branch_indices(cache.entry(str(name))))
    if total_branches <= 0:
        raise ValueError("V127 streaming residual scales require non-empty data")
    per_branch = max(1, int(math.ceil(float(sample_rows) / float(total_branches))))
    state_samples: list[np.ndarray] = []
    flow_samples: list[np.ndarray] = []

    for cache, name in entries:
        entry = cache.entry(name)
        arrays = entry.arrays
        for raw_index in _branch_indices(entry):
            states = np.asarray(arrays["target_states"][raw_index], dtype=np.float32)
            initial = np.asarray(arrays["initial_state"][raw_index], dtype=np.float32)
            if states.ndim != 3 or initial.shape != states.shape[1:]:
                raise ValueError(f"{name}: invalid state trajectory for streaming scale")
            timesteps, nodes, _ = states.shape
            q = min(per_branch, timesteps * nodes)
            flat = np.linspace(0, timesteps * nodes - 1, q, dtype=np.int64)
            tt, nn = flat // nodes, flat % nodes
            current = states[tt, nn]
            previous = np.empty_like(current)
            first = tt == 0
            if np.any(first):
                previous[first] = initial[nn[first]]
            if np.any(~first):
                previous[~first] = states[tt[~first] - 1, nn[~first]]
            state_samples.append(np.abs(current - previous).astype(np.float32, copy=False))

            flows = np.asarray(arrays["target_actuator_flows"][raw_index], dtype=np.float32)
            f0 = np.asarray(arrays["previous_actuator_flow"][raw_index], dtype=np.float32)
            if flows.ndim != 2 or f0.shape != flows.shape[1:]:
                raise ValueError(f"{name}: invalid actuator-flow trajectory for streaming scale")
            qf = min(per_branch, flows.shape[0])
            ft = np.linspace(0, flows.shape[0] - 1, qf, dtype=np.int64)
            fprev = np.empty_like(flows[ft])
            ffirst = ft == 0
            if np.any(ffirst):
                fprev[ffirst] = f0
            if np.any(~ffirst):
                fprev[~ffirst] = flows[ft[~ffirst] - 1]
            flow_samples.append(np.abs(flows[ft] - fprev).astype(np.float32, copy=False))

    state_matrix = np.concatenate(state_samples, axis=0)
    flow_matrix = np.concatenate(flow_samples, axis=0)
    state_scale = np.quantile(state_matrix, 0.995, axis=0).clip(min=1.0e-5).astype(np.float32)
    flow_scale = np.quantile(flow_matrix, 0.995, axis=0).clip(min=1.0e-5).astype(np.float32)
    telemetry = {
        "requested_sample_rows": int(sample_rows),
        "total_branches": int(total_branches),
        "sample_rows_per_branch": int(per_branch),
        "state_sample_rows": int(state_matrix.shape[0]),
        "flow_sample_rows": int(flow_matrix.shape[0]),
    }
    return state_scale, flow_scale, telemetry


def train_hydraulic_stage_streaming_v127(
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
    design.validate()
    torch.manual_seed(design.seed)
    np.random.seed(design.seed)
    model.train().to(device)
    static = _static(graph, device)
    weights = _state_weights(
        model, depth_index=depth_index, flood_rate_index=flood_rate_index, design=design, device=device
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
                raise RuntimeError("V127 streaming hydraulic stage selected no transitions")
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            phase_seen[name].add(phase)
            transition_count += int(steps * branches)
            for start in range(0, branches, design.hydraulic_branch_chunk):
                stop = min(start + design.hydraulic_branch_chunk, branches)
                positions = np.arange(start, stop, dtype=np.int64)
                chunk = _select_to_device(cpu, positions, device=device, include_truth=True)
                chunk_n = len(positions)
                physics_norm, identity = model.actuator.prepare_static(
                    static["physics"], batch_size=chunk_n
                )
                static_norm, edges, inv = model.transition.prepare_static(
                    static["static"], static["edges"], batch_size=chunk_n, dtype=torch.float32
                )
                chunk_loss = torch.zeros((), device=device)
                for k in range(phase, horizon, design.teacher_stride):
                    prev_state = chunk["initial"] if k == 0 else chunk["states"][:, k - 1]
                    prev_flow = chunk["previous_flow"] if k == 0 else chunk["flows"][:, k - 1]
                    setting = chunk["settings"][:, k]
                    q, _ = model.actuator.forward_prepared(
                        prev_state[:, static["up"]],
                        prev_state[:, static["down"]],
                        setting,
                        prev_flow,
                        physics_norm,
                        identity,
                    )
                    injection = torch.zeros(
                        chunk_n, prev_state.shape[1], 1, device=device, dtype=prev_state.dtype
                    )
                    injection = injection.index_add(1, static["up"], -q[..., None])
                    injection = injection.index_add(1, static["down"], q[..., None])
                    action_context = model._setting_context(
                        setting,
                        static["up"],
                        static["down"],
                        node_count=prev_state.shape[1],
                        dtype=prev_state.dtype,
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
                    state_error = (pred_state - chunk["states"][:, k]) / model.transition.state_std
                    flow_error = (q - chunk["flows"][:, k]) / model.actuator.flow_std
                    chunk_loss = chunk_loss + F.smooth_l1_loss(
                        state_error * weights, torch.zeros_like(state_error), beta=0.5
                    )
                    chunk_loss = chunk_loss + design.flow_weight * F.smooth_l1_loss(
                        flow_error, torch.zeros_like(flow_error), beta=0.5
                    )
                weighted = (chunk_loss / steps) * (float(chunk_n) / float(branches))
                if not bool(torch.isfinite(weighted)):
                    raise RuntimeError(f"{name}: V127 streaming hydraulic loss non-finite")
                weighted.backward()
                group_loss += float(weighted.detach())
                del chunk, chunk_loss, weighted
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            losses.append(group_loss)
            by_source[source].append(group_loss)
            del cpu
        row: dict[str, float | int | str] = {
            "stage": "hydraulic_teacher_forced_cpu_stream_gpu_chunk",
            "epoch": epoch,
            "teacher_phase": phase,
            "loss": float(np.mean(losses)),
            "hydraulic_branch_chunk": int(design.hydraulic_branch_chunk),
            "teacher_transitions": int(transition_count),
            "min_group_teacher_phase_coverage": min(
                (len(v) / design.teacher_stride for v in phase_seen.values()), default=0.0
            ),
            **_cuda_peak(device),
        }
        for source, values in by_source.items():
            if values:
                row[f"loss_{source.lower()}"] = float(np.mean(values))
        history.append(row)
        print("[V127_STREAM_HYDRAULIC] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        gc.collect()
    if any(len(v) < design.teacher_stride for v in phase_seen.values()):
        raise RuntimeError("V127 streaming hydraulic curriculum missed a teacher phase")
    return history


def train_truncated_rollout_stage_streaming_v127(
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
    design.validate()
    static = _static(graph, device)
    weights = _state_weights(
        model, depth_index=depth_index, flood_rate_index=flood_rate_index, design=design, device=device
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=design.rollout_learning_rate, weight_decay=design.weight_decay
    )
    model.train().to(device)
    history: list[dict[str, float | int | str]] = []
    _reset_cuda_peak(device)
    for epoch, horizon in enumerate(design.rollout_horizons, start=1):
        losses: list[float] = []
        by_source: dict[str, list[float]] = {key: [] for key in source_caches}
        for source, name in _ordered(source_groups, epoch + 101, design.seed):
            cpu = _cpu_group(source_caches[source], name, normalization)
            branches = int(cpu["settings"].shape[0])
            candidates = _candidate_permutation(
                branches - 1, group_name=name, epoch=epoch, seed=design.seed
            )[: min(design.rollout_candidates_per_group, branches - 1)]
            positions = np.concatenate((np.asarray([0], dtype=np.int64), candidates))
            chunk = _select_to_device(
                cpu, positions, device=device, horizon=horizon, include_truth=True
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
            loss = F.smooth_l1_loss(
                state_error * weights, torch.zeros_like(state_error), beta=0.5
            ) + design.flow_weight * F.smooth_l1_loss(
                flow_error, torch.zeros_like(flow_error), beta=0.5
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"{name}: V127 streaming rollout loss non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            value = float(loss.detach())
            losses.append(value)
            by_source[source].append(value)
            del cpu, chunk, output, loss
        row: dict[str, float | int | str] = {
            "stage": "autoregressive_truncated_rollout_cpu_stream",
            "epoch": epoch,
            "horizon_steps": int(horizon),
            "horizon_minutes": int(horizon * 5),
            "candidates_per_group": int(design.rollout_candidates_per_group),
            "loss": float(np.mean(losses)),
            **_cuda_peak(device),
        }
        for source, values in by_source.items():
            if values:
                row[f"loss_{source.lower()}"] = float(np.mean(values))
        history.append(row)
        print("[V127_STREAM_ROLLOUT] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        gc.collect()
    return history


def _retained(device: torch.device) -> torch.Tensor:
    return torch.as_tensor([5, 11, 23, 35, 47, 71], dtype=torch.long, device=device)


def _candidate_loss_streaming(
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
) -> tuple[torch.Tensor, int]:
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
    hard_delta_loss = F.smooth_l1_loss(hard_delta / delta_scale, true_delta / delta_scale, beta=0.5)
    smooth_delta_loss = F.smooth_l1_loss(
        smooth_delta / delta_scale, true_delta / delta_scale, beta=0.5
    )
    pair_terms: list[torch.Tensor] = []
    informative = torch.abs(true_delta) > float(informative_threshold)
    informative_pairs = int(informative.sum().item())
    if bool(informative.any()):
        pair_terms.append(
            F.softplus(-torch.sign(true_delta[informative]) * smooth_delta[informative] / delta_scale)
        )
    if len(true_delta) > 1:
        ii, jj = torch.triu_indices(len(true_delta), len(true_delta), offset=1, device=true_delta.device)
        truth_pair = true_delta[ii] - true_delta[jj]
        pred_pair = smooth_delta[ii] - smooth_delta[jj]
        keep = torch.abs(truth_pair) > float(informative_threshold)
        informative_pairs += int(keep.sum().item())
        if bool(keep.any()):
            pair_terms.append(F.softplus(-torch.sign(truth_pair[keep]) * pred_pair[keep] / delta_scale))
    pair_loss = torch.cat(pair_terms).mean() if pair_terms else smooth_delta.new_zeros(())
    retained = _retained(true_delta.device)
    state_loss = F.smooth_l1_loss(
        (output.rollout.states.index_select(1, retained) - chunk["states"].index_select(1, retained))
        / model.transition.state_std,
        torch.zeros_like(output.rollout.states.index_select(1, retained)),
        beta=0.5,
    )
    loss = (
        design.node_flood_weight * node_effect
        + design.tfv_weight * (hard_abs + hard_delta_loss + smooth_delta_loss)
        + design.pairwise_weight * pair_loss
        + design.rollout_state_weight * state_loss
    )
    return loss, informative_pairs


def train_objective_stage_streaming_v127(
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
    design.validate()
    static = _static(graph, device)
    model.train().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=design.objective_learning_rate, weight_decay=design.weight_decay
    )
    history: list[dict[str, float | int | str]] = []
    pair_seen: dict[str, set[tuple[int, int]]] = {
        name: set() for values in source_groups.values() for name in values
    }
    _reset_cuda_peak(device)
    for epoch in range(1, design.objective_epochs + 1):
        records: list[float] = []
        total_candidates = 0
        informative_pairs = 0
        for source, name in _ordered(source_groups, epoch + 17, design.seed):
            cpu = _cpu_group(source_caches[source], name, normalization)
            truth_np = _truth_node_volume(source_caches[source], name).astype(np.float32)
            branches = int(cpu["settings"].shape[0])
            if truth_np.shape[0] != branches:
                raise RuntimeError(f"{name}: SWMM label count does not match branch order")
            candidate_count = branches - 1
            if candidate_count <= 0:
                raise RuntimeError(f"{name}: objective group has no candidate")
            true_tfv_np = truth_np.sum(axis=1, dtype=np.float64)
            true_delta_np = true_tfv_np[1:] - true_tfv_np[0]
            tfv_scale = torch.tensor(
                max(float(np.quantile(true_tfv_np, 0.75)), 100.0), dtype=torch.float32, device=device
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
            ref_state_loss = F.smooth_l1_loss(
                (ref_output.rollout.states.index_select(1, retained) - ref["states"].index_select(1, retained))
                / model.transition.state_std,
                torch.zeros_like(ref_output.rollout.states.index_select(1, retained)),
                beta=0.5,
            )
            ref_loss = ref_hard_loss + design.rollout_state_weight * ref_state_loss
            if not bool(torch.isfinite(ref_loss)):
                raise RuntimeError(f"{name}: streaming reference objective loss non-finite")
            ref_loss.backward()
            ref_hard = ref_output.tfv_m3.detach()[0]
            ref_smooth = ref_output.optimization_tfv_m3.detach()[0]
            ref_smooth_node = ref_output.optimization_node_flood_volume_m3.detach()[0:1]
            group_loss = float(ref_loss.detach())
            del ref_output, ref_loss, ref_hard_loss, ref_state_loss

            order = _candidate_permutation(
                candidate_count, group_name=name, epoch=epoch, seed=design.seed
            )
            for start in range(0, candidate_count, design.objective_candidate_chunk):
                positions = order[start : start + design.objective_candidate_chunk]
                chunk = _select_to_device(cpu, positions, device=device, include_truth=True)
                truth_chunk = torch.as_tensor(truth_np[positions], dtype=torch.float32, device=device)
                loss, pair_count = _candidate_loss_streaming(
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
                )
                weight = float(len(positions)) / float(candidate_count)
                weighted = loss * weight
                if not bool(torch.isfinite(weighted)):
                    raise RuntimeError(f"{name}: streaming candidate objective loss non-finite")
                weighted.backward()
                group_loss += float(weighted.detach())
                informative_pairs += int(pair_count)
                ids = [int(x) for x in positions]
                for i in ids:
                    pair_seen[name].add((0, i))
                for i_pos, i in enumerate(ids):
                    for j in ids[i_pos + 1 :]:
                        pair_seen[name].add(tuple(sorted((i, j))))
                del chunk, truth_chunk, loss, weighted
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            records.append(group_loss)
            total_candidates += candidate_count
            del cpu, ref, ref_volume

        possible_pairs = 0
        seen_pairs = 0
        for source, names in source_groups.items():
            for name in names:
                n = len(_branch_indices(source_caches[source].entry(name))) - 1
                possible_pairs += n + n * (n - 1) // 2
                seen_pairs += len(pair_seen[name])
        row: dict[str, float | int | str] = {
            "stage": "h360_control_oriented_cpu_stream_gpu_microbatch",
            "epoch": epoch,
            "loss": float(np.mean(records)) if records else float("nan"),
            "candidate_total": int(total_candidates),
            "candidate_seen_unique": int(total_candidates),
            "candidate_coverage_fraction": 1.0 if total_candidates else 0.0,
            "objective_candidate_chunk": int(design.objective_candidate_chunk),
            "informative_pair_terms": int(informative_pairs),
            "cumulative_pair_partition_coverage": (
                float(seen_pairs) / float(possible_pairs) if possible_pairs else 0.0
            ),
            "effect_scale": "TrainFit within-group abs(delta TFV) P75",
            "near_ties_used_for_sign_ranking": False,
            **_cuda_peak(device),
        }
        if not math.isfinite(float(row["loss"])):
            raise RuntimeError("V127 streaming H360 objective produced no finite loss")
        history.append(row)
        print("[V127_STREAM_OBJECTIVE] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        gc.collect()
    return history


def evaluate_objective_groups_streaming_v127(
    model: Any,
    *,
    cache: Any,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    flood_rate_index: int,
    design: V127ControlTrainingDesign,
) -> dict[str, float | str]:
    design.validate()
    model.eval().to(device)
    static = _static(graph, device)
    hard_rank: list[float] = []
    smooth_rank: list[float] = []
    hard_pair: list[float] = []
    informative_pair: list[float] = []
    smooth_pair: list[float] = []
    hard_top1: list[float] = []
    smooth_top1: list[float] = []
    hard_mae: list[float] = []
    hard_delta_mae: list[float] = []
    smooth_delta_mae: list[float] = []
    regret: list[float] = []
    normalized_regret: list[float] = []
    _reset_cuda_peak(device)
    with torch.no_grad():
        for name in names:
            cpu = _cpu_group(cache, name, normalization)
            truth = _truth_node_volume(cache, name).sum(axis=1).astype(np.float64)
            branch_count = int(cpu["settings"].shape[0])
            if truth.shape[0] != branch_count:
                raise RuntimeError(f"{name}: streaming evaluation label/action order mismatch")
            hard_values: list[float] = []
            smooth_values: list[float] = []
            for start in range(0, branch_count, design.evaluation_branch_chunk):
                stop = min(start + design.evaluation_branch_chunk, branch_count)
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
                hard_values.extend(output.tfv_m3.detach().cpu().numpy().astype(float).tolist())
                smooth_values.extend(
                    output.optimization_tfv_m3.detach().cpu().numpy().astype(float).tolist()
                )
                del chunk, output
            hard = np.asarray(hard_values, dtype=float)
            smooth = np.asarray(smooth_values, dtype=float)
            hard_rank.append(_spearman(hard, truth))
            smooth_rank.append(_spearman(smooth, truth))
            hard_pair.append(_pairwise_accuracy(truth, hard))
            smooth_pair.append(_pairwise_accuracy(truth, smooth))
            threshold = informative_pair_threshold_v127(float(truth[0]), design)
            informative_pair.append(_pairwise_accuracy(truth, hard, threshold=threshold))
            optimum = float(np.min(truth))
            optimum_mask = np.isclose(truth, optimum, rtol=0.0, atol=1.0e-9)
            hard_selected = int(np.argmin(hard))
            smooth_selected = int(np.argmin(smooth))
            hard_top1.append(float(bool(optimum_mask[hard_selected])))
            smooth_top1.append(float(bool(optimum_mask[smooth_selected])))
            hard_mae.append(float(np.mean(np.abs(hard - truth))))
            hard_delta_mae.append(
                float(np.mean(np.abs((hard[1:] - hard[0]) - (truth[1:] - truth[0]))))
            )
            smooth_delta_mae.append(
                float(np.mean(np.abs((smooth[1:] - smooth[0]) - (truth[1:] - truth[0]))))
            )
            selected_regret = float(truth[hard_selected] - optimum)
            regret.append(selected_regret)
            normalized_regret.append(selected_regret / max(float(np.max(truth) - optimum), 1.0))
            del cpu

    def mean(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else float("nan")

    return {
        "groups": float(len(names)),
        "rank": mean(hard_rank),
        "pairwise": mean(hard_pair),
        "top1": mean(hard_top1),
        "tfv_mae_m3": mean(hard_mae),
        "selected_regret_m3": mean(regret),
        "hard_tfv_rank": mean(hard_rank),
        "hard_tfv_pairwise": mean(hard_pair),
        "hard_tfv_informative_pairwise": mean(informative_pair),
        "hard_tfv_top1": mean(hard_top1),
        "smooth_tfv_rank": mean(smooth_rank),
        "smooth_tfv_pairwise": mean(smooth_pair),
        "smooth_tfv_top1": mean(smooth_top1),
        "hard_tfv_delta_mae_m3": mean(hard_delta_mae),
        "smooth_tfv_delta_mae_m3": mean(smooth_delta_mae),
        "normalized_selected_regret": mean(normalized_regret),
        "authoritative_label_order": "reference_then_candidates",
        "prediction_metric": "hard surrogate TFV vs authoritative SWMM cumulative TFV",
        "smooth_metric_role": "optimization/action-effect diagnostic only",
        "informative_pair_threshold": "max(1 m3, 0.001 * reference TFV)",
        "streaming_execution": V127_STREAMING_TRAINING_CONTRACT,
        **_cuda_peak(device),
    }


__all__ = [
    "V127StreamingMemoryDesign",
    "V127_STREAMING_TRAINING_CONTRACT",
    "derive_residual_scales_streaming_v127",
    "evaluate_objective_groups_streaming_v127",
    "train_hydraulic_stage_streaming_v127",
    "train_objective_stage_streaming_v127",
    "train_truncated_rollout_stage_streaming_v127",
]
