from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_train_response_v60 import InputNormalizationV60
from .step2_train_v127 import (
    _branch_indices,
    _denormalize_group,
    _ordered,
    _spearman,
    _static,
    _truth_node_volume,
)

V127_CONTROL_TRAINING_CONTRACT = (
    "PROJECT7_V127_CONTROL_ORIENTED_CURRICULUM_V1_MEMORY_SAFE_FULL_COVERAGE"
)


@dataclass(frozen=True)
class V127ControlTrainingDesign:
    hydraulic_epochs: int = 4
    hydraulic_branch_chunk: int = 8
    teacher_stride: int = 4
    rollout_horizons: tuple[int, ...] = (12, 24)
    rollout_candidates_per_group: int = 3
    objective_epochs: int = 3
    objective_candidate_chunk: int = 3
    evaluation_branch_chunk: int = 8
    learning_rate: float = 2.0e-4
    rollout_learning_rate: float = 1.0e-4
    objective_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    grad_clip: float = 5.0
    depth_weight: float = 2.0
    flood_rate_weight: float = 3.0
    flow_weight: float = 1.0
    node_flood_weight: float = 1.0
    tfv_weight: float = 1.0
    rollout_state_weight: float = 0.5
    pairwise_weight: float = 0.35
    informative_pair_absolute_m3: float = 1.0
    informative_pair_reference_fraction: float = 1.0e-3
    seed: int = 42

    def validate(self) -> None:
        if self.hydraulic_epochs < self.teacher_stride:
            raise ValueError("V127 hydraulic epochs must cover every teacher phase")
        if self.hydraulic_branch_chunk <= 0 or self.objective_candidate_chunk <= 0:
            raise ValueError("V127 branch chunks must be positive")
        if self.evaluation_branch_chunk <= 0 or self.rollout_candidates_per_group <= 0:
            raise ValueError("V127 evaluation/rollout branch budgets must be positive")
        if not self.rollout_horizons or any(h <= 0 or h > 72 for h in self.rollout_horizons):
            raise ValueError("V127 rollout curriculum horizons are invalid")
        if tuple(sorted(self.rollout_horizons)) != self.rollout_horizons:
            raise ValueError("V127 rollout curriculum must increase monotonically")
        if self.objective_epochs <= 0 or self.teacher_stride <= 0:
            raise ValueError("V127 training epochs/teacher stride are invalid")
        for value in (
            self.learning_rate,
            self.rollout_learning_rate,
            self.objective_learning_rate,
            self.weight_decay,
            self.grad_clip,
            self.depth_weight,
            self.flood_rate_weight,
            self.flow_weight,
            self.node_flood_weight,
            self.tfv_weight,
            self.rollout_state_weight,
            self.pairwise_weight,
            self.informative_pair_absolute_m3,
            self.informative_pair_reference_fraction,
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("V127 control-training design contains invalid values")
        if self.seed < 0:
            raise ValueError("V127 training seed must be non-negative")


def _stable_seed(text: str, base: int) -> int:
    return (
        int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) + int(base)
    ) % (2**32)


def _state_weights(model: Any, *, depth_index: int, flood_rate_index: int, design: V127ControlTrainingDesign, device: torch.device) -> torch.Tensor:
    state_dim = int(model.transition.state_mean.numel())
    weights = torch.ones(state_dim, device=device)
    if 0 <= depth_index < state_dim:
        weights[depth_index] = float(design.depth_weight)
    if 0 <= flood_rate_index < state_dim:
        weights[flood_rate_index] = float(design.flood_rate_weight)
    return weights


def _candidate_permutation(count: int, *, group_name: str, epoch: int, seed: int) -> np.ndarray:
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    values = np.arange(1, count + 1, dtype=np.int64)
    rng = np.random.default_rng(_stable_seed(f"{group_name}:objective:{epoch}", seed))
    return rng.permutation(values)


def informative_pair_threshold_v127(reference_tfv_m3: float, design: V127ControlTrainingDesign) -> float:
    return max(
        float(design.informative_pair_absolute_m3),
        float(design.informative_pair_reference_fraction) * abs(float(reference_tfv_m3)),
    )


def train_hydraulic_stage_chunked_v127(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    depth_index: int,
    flood_rate_index: int,
    design: V127ControlTrainingDesign = V127ControlTrainingDesign(),
) -> list[dict[str, float | int | str]]:
    """Teacher-forced hydraulic fit with full branch coverage and bounded GPU memory.

    Each group is split only across the branch axis. Chunk losses are weighted by their
    branch fraction before backward(), so the gradient matches a full-branch mean while
    avoiding the 25-branch GPU OOM seen on the 8-GB development device.
    """
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
    phase_seen_by_group: dict[str, set[int]] = {
        name: set() for values in source_groups.values() for name in values
    }
    for epoch in range(1, design.hydraulic_epochs + 1):
        phase = (epoch - 1) % design.teacher_stride
        losses: list[float] = []
        by_source: dict[str, list[float]] = {key: [] for key in source_caches}
        transition_count = 0
        for source, name in _ordered(source_groups, epoch, design.seed):
            data = _denormalize_group(
                source_caches[source].batch(name, normalization, device), normalization
            )
            branches, horizon = data["settings"].shape[:2]
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            group_steps = len(range(phase, horizon, design.teacher_stride))
            if group_steps <= 0:
                raise RuntimeError("V127 hydraulic stage selected no transitions")
            phase_seen_by_group[name].add(phase)
            transition_count += group_steps * branches
            for start in range(0, branches, design.hydraulic_branch_chunk):
                stop = min(start + design.hydraulic_branch_chunk, branches)
                idx = slice(start, stop)
                chunk_n = stop - start
                physics_norm, identity = model.actuator.prepare_static(
                    static["physics"], batch_size=chunk_n
                )
                static_norm, edges, inv = model.transition.prepare_static(
                    static["static"], static["edges"], batch_size=chunk_n, dtype=torch.float32
                )
                chunk_loss = torch.zeros((), device=device)
                for k in range(phase, horizon, design.teacher_stride):
                    prev_state = data["initial"][idx] if k == 0 else data["states"][idx, k - 1]
                    prev_flow = data["previous_flow"][idx] if k == 0 else data["flows"][idx, k - 1]
                    setting = data["settings"][idx, k]
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
                        data["rainfall"][idx, k],
                        static_norm,
                        injection,
                        edges,
                        inv,
                        action_context,
                    )
                    state_error = (
                        pred_state - data["states"][idx, k]
                    ) / model.transition.state_std
                    flow_error = (q - data["flows"][idx, k]) / model.actuator.flow_std
                    chunk_loss = chunk_loss + F.smooth_l1_loss(
                        state_error * weights, torch.zeros_like(state_error), beta=0.5
                    )
                    chunk_loss = chunk_loss + design.flow_weight * F.smooth_l1_loss(
                        flow_error, torch.zeros_like(flow_error), beta=0.5
                    )
                chunk_loss = chunk_loss / group_steps
                weighted = chunk_loss * (float(chunk_n) / float(branches))
                if not bool(torch.isfinite(weighted)):
                    raise RuntimeError(f"{name}: V127 hydraulic loss non-finite")
                weighted.backward()
                group_loss += float(weighted.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            losses.append(group_loss)
            by_source[source].append(group_loss)
        min_group_phase_coverage = min(
            (len(v) / design.teacher_stride for v in phase_seen_by_group.values()),
            default=0.0,
        )
        row: dict[str, float | int | str] = {
            "stage": "hydraulic_teacher_forced_chunked",
            "epoch": epoch,
            "teacher_phase": phase,
            "loss": float(np.mean(losses)),
            "hydraulic_branch_chunk": int(design.hydraulic_branch_chunk),
            "teacher_transitions": int(transition_count),
            "min_group_teacher_phase_coverage": float(min_group_phase_coverage),
        }
        for source, values in by_source.items():
            if values:
                row[f"loss_{source.lower()}"] = float(np.mean(values))
        history.append(row)
        print("[V127_HYDRAULIC_CHUNKED] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    if any(len(v) < design.teacher_stride for v in phase_seen_by_group.values()):
        raise RuntimeError("V127 hydraulic curriculum did not cover all teacher phases per group")
    return history


def train_truncated_rollout_stage_v127(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    depth_index: int,
    flood_rate_index: int,
    design: V127ControlTrainingDesign = V127ControlTrainingDesign(),
) -> list[dict[str, float | int | str]]:
    """Short-to-H120 autoregressive curriculum to reduce rollout drift.

    This stage uses only TrainFit D2/D3/D4-FIT labels. It does not change the scientific
    target or use future truth online; future hydraulic states are offline supervision.
    """
    design.validate()
    static = _static(graph, device)
    weights = _state_weights(
        model,
        depth_index=depth_index,
        flood_rate_index=flood_rate_index,
        design=design,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=design.rollout_learning_rate, weight_decay=design.weight_decay
    )
    model.train().to(device)
    history: list[dict[str, float | int | str]] = []
    for epoch, horizon in enumerate(design.rollout_horizons, start=1):
        losses: list[float] = []
        by_source: dict[str, list[float]] = {key: [] for key in source_caches}
        for source, name in _ordered(source_groups, epoch + 101, design.seed):
            data = _denormalize_group(
                source_caches[source].batch(name, normalization, device), normalization
            )
            branches = int(data["settings"].shape[0])
            candidates = _candidate_permutation(
                branches - 1, group_name=name, epoch=epoch, seed=design.seed
            )[: min(design.rollout_candidates_per_group, branches - 1)]
            chosen = np.concatenate((np.asarray([0], dtype=np.int64), candidates))
            index = torch.as_tensor(chosen, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            output = model.rollout(
                data["initial"].index_select(0, index),
                data["rainfall"].index_select(0, index)[:, :horizon],
                data["settings"].index_select(0, index)[:, :horizon],
                data["previous_flow"].index_select(0, index),
                static["up"],
                static["down"],
                static["physics"],
                static["static"],
                static["edges"],
            )
            true_states = data["states"].index_select(0, index)[:, :horizon]
            true_flows = data["flows"].index_select(0, index)[:, :horizon]
            state_error = (output.states - true_states) / model.transition.state_std
            flow_error = (output.actuator_flows - true_flows) / model.actuator.flow_std
            loss = F.smooth_l1_loss(
                state_error * weights, torch.zeros_like(state_error), beta=0.5
            ) + design.flow_weight * F.smooth_l1_loss(
                flow_error, torch.zeros_like(flow_error), beta=0.5
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"{name}: V127 truncated-rollout loss non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            value = float(loss.detach())
            losses.append(value)
            by_source[source].append(value)
        row: dict[str, float | int | str] = {
            "stage": "autoregressive_truncated_rollout",
            "epoch": epoch,
            "horizon_steps": int(horizon),
            "horizon_minutes": int(horizon * 5),
            "candidates_per_group": int(design.rollout_candidates_per_group),
            "loss": float(np.mean(losses)),
        }
        for source, values in by_source.items():
            if values:
                row[f"loss_{source.lower()}"] = float(np.mean(values))
        history.append(row)
        print("[V127_TRUNCATED_ROLLOUT] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    return history


def _state_retained_indices(device: torch.device) -> torch.Tensor:
    return torch.as_tensor([5, 11, 23, 35, 47, 71], dtype=torch.long, device=device)


def _candidate_objective_loss(
    model: Any,
    *,
    data: dict[str, torch.Tensor],
    target_volume: torch.Tensor,
    candidates: torch.Tensor,
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
        initial_state=data["initial"].index_select(0, candidates),
        rainfall=data["rainfall"].index_select(0, candidates),
        settings=data["settings"].index_select(0, candidates),
        previous_actuator_flow=data["previous_flow"].index_select(0, candidates),
        actuator_upstream=static["up"],
        actuator_downstream=static["down"],
        actuator_physics=static["physics"],
        static_node_features=static["static"],
        edge_index=static["edges"],
        flood_rate_index=flood_rate_index,
        priority_indices=None,
        dt_seconds=300.0,
    )
    truth = target_volume.index_select(0, candidates)
    true_tfv = truth.sum(-1)
    hard_abs = F.smooth_l1_loss(
        (output.tfv_m3 - true_tfv) / tfv_scale,
        torch.zeros_like(true_tfv),
        beta=0.5,
    )
    true_delta_node = truth - target_volume[0:1]
    smooth_delta_node = output.optimization_node_flood_volume_m3 - ref_smooth_node.detach()
    node_effect = F.smooth_l1_loss(
        smooth_delta_node / node_delta_scale,
        true_delta_node / node_delta_scale,
        beta=0.5,
    )
    true_delta = true_tfv - target_volume[0].sum()
    hard_delta = output.tfv_m3 - ref_hard_tfv.detach()
    smooth_delta = output.optimization_tfv_m3 - ref_smooth_tfv.detach()
    hard_delta_loss = F.smooth_l1_loss(
        hard_delta / delta_scale, true_delta / delta_scale, beta=0.5
    )
    smooth_delta_loss = F.smooth_l1_loss(
        smooth_delta / delta_scale, true_delta / delta_scale, beta=0.5
    )
    pair_terms: list[torch.Tensor] = []
    informative = torch.abs(true_delta) > float(informative_threshold)
    informative_pairs = int(informative.sum().item())
    if bool(informative.any()):
        pair_terms.append(
            F.softplus(
                -torch.sign(true_delta[informative])
                * smooth_delta[informative]
                / delta_scale
            )
        )
    if len(true_delta) > 1:
        ii, jj = torch.triu_indices(
            len(true_delta), len(true_delta), offset=1, device=true_delta.device
        )
        truth_pair = true_delta[ii] - true_delta[jj]
        pred_pair = smooth_delta[ii] - smooth_delta[jj]
        keep = torch.abs(truth_pair) > float(informative_threshold)
        informative_pairs += int(keep.sum().item())
        if bool(keep.any()):
            pair_terms.append(
                F.softplus(-torch.sign(truth_pair[keep]) * pred_pair[keep] / delta_scale)
            )
    pair_loss = (
        torch.cat(pair_terms).mean() if pair_terms else smooth_delta.new_zeros(())
    )
    retained = _state_retained_indices(true_delta.device)
    true_states = data["states"].index_select(0, candidates).index_select(1, retained)
    pred_states = output.rollout.states.index_select(1, retained)
    state_loss = F.smooth_l1_loss(
        (pred_states - true_states) / model.transition.state_std,
        torch.zeros_like(pred_states),
        beta=0.5,
    )
    loss = (
        design.node_flood_weight * node_effect
        + design.tfv_weight * (hard_abs + hard_delta_loss + smooth_delta_loss)
        + design.pairwise_weight * pair_loss
        + design.rollout_state_weight * state_loss
    )
    return loss, informative_pairs


def train_objective_stage_control_v127(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    flood_rate_index: int,
    design: V127ControlTrainingDesign = V127ControlTrainingDesign(),
) -> list[dict[str, float | int | str]]:
    """Full-candidate H360 training with action-effect-scaled ranking supervision."""
    design.validate()
    static = _static(graph, device)
    model.train().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=design.objective_learning_rate,
        weight_decay=design.weight_decay,
    )
    history: list[dict[str, float | int | str]] = []
    pair_seen: dict[str, set[tuple[int, int]]] = {
        name: set() for values in source_groups.values() for name in values
    }
    for epoch in range(1, design.objective_epochs + 1):
        records: list[float] = []
        total_candidates = 0
        informative_pairs = 0
        for source, name in _ordered(source_groups, epoch + 17, design.seed):
            data = _denormalize_group(
                source_caches[source].batch(name, normalization, device), normalization
            )
            target_volume = torch.as_tensor(
                _truth_node_volume(source_caches[source], name),
                dtype=torch.float32,
                device=device,
            )
            branches = int(data["settings"].shape[0])
            if target_volume.shape[0] != branches:
                raise RuntimeError(f"{name}: SWMM label count does not match model branch order")
            candidate_count = branches - 1
            if candidate_count <= 0:
                raise RuntimeError(f"{name}: objective group has no candidate")
            true_tfv_all = target_volume.sum(-1)
            true_delta_all = true_tfv_all[1:] - true_tfv_all[0]
            tfv_scale = torch.quantile(true_tfv_all.detach(), 0.75).clamp_min(100.0)
            delta_scale = torch.quantile(torch.abs(true_delta_all.detach()), 0.75).clamp_min(100.0)
            node_delta_all = target_volume[1:] - target_volume[0:1]
            node_delta_scale = torch.quantile(
                torch.abs(node_delta_all.detach()).reshape(-1), 0.75
            ).clamp_min(25.0)
            informative_threshold = informative_pair_threshold_v127(
                float(true_tfv_all[0].detach()), design
            )
            optimizer.zero_grad(set_to_none=True)

            ref_index = torch.as_tensor([0], dtype=torch.long, device=device)
            ref_output = model.objective_rollout(
                initial_state=data["initial"].index_select(0, ref_index),
                rainfall=data["rainfall"].index_select(0, ref_index),
                settings=data["settings"].index_select(0, ref_index),
                previous_actuator_flow=data["previous_flow"].index_select(0, ref_index),
                actuator_upstream=static["up"],
                actuator_downstream=static["down"],
                actuator_physics=static["physics"],
                static_node_features=static["static"],
                edge_index=static["edges"],
                flood_rate_index=flood_rate_index,
                priority_indices=None,
                dt_seconds=300.0,
            )
            ref_truth = true_tfv_all[0:1]
            ref_hard_loss = F.smooth_l1_loss(
                (ref_output.tfv_m3 - ref_truth) / tfv_scale,
                torch.zeros_like(ref_truth),
                beta=0.5,
            )
            retained = _state_retained_indices(device)
            ref_state_loss = F.smooth_l1_loss(
                (
                    ref_output.rollout.states.index_select(1, retained)
                    - data["states"][0:1].index_select(1, retained)
                )
                / model.transition.state_std,
                torch.zeros_like(ref_output.rollout.states.index_select(1, retained)),
                beta=0.5,
            )
            ref_loss = ref_hard_loss + design.rollout_state_weight * ref_state_loss
            if not bool(torch.isfinite(ref_loss)):
                raise RuntimeError(f"{name}: reference objective loss non-finite")
            ref_loss.backward()
            ref_hard = ref_output.tfv_m3.detach()[0]
            ref_smooth = ref_output.optimization_tfv_m3.detach()[0]
            ref_smooth_node = ref_output.optimization_node_flood_volume_m3.detach()[0:1]
            group_loss = float(ref_loss.detach())

            order = _candidate_permutation(
                candidate_count, group_name=name, epoch=epoch, seed=design.seed
            )
            for start in range(0, candidate_count, design.objective_candidate_chunk):
                chunk_np = order[start : start + design.objective_candidate_chunk]
                chunk = torch.as_tensor(chunk_np, dtype=torch.long, device=device)
                loss, pair_count = _candidate_objective_loss(
                    model,
                    data=data,
                    target_volume=target_volume,
                    candidates=chunk,
                    ref_hard_tfv=ref_hard,
                    ref_smooth_tfv=ref_smooth,
                    ref_smooth_node=ref_smooth_node,
                    static=static,
                    flood_rate_index=flood_rate_index,
                    design=design,
                    tfv_scale=tfv_scale,
                    delta_scale=delta_scale,
                    node_delta_scale=node_delta_scale,
                    informative_threshold=informative_threshold,
                )
                weight = float(len(chunk_np)) / float(candidate_count)
                weighted = loss * weight
                if not bool(torch.isfinite(weighted)):
                    raise RuntimeError(f"{name}: V127 candidate objective loss non-finite")
                weighted.backward()
                group_loss += float(weighted.detach())
                informative_pairs += pair_count
                ids = [int(x) for x in chunk_np]
                for i in ids:
                    pair_seen[name].add((0, i))
                for i_pos, i in enumerate(ids):
                    for j in ids[i_pos + 1 :]:
                        pair_seen[name].add(tuple(sorted((i, j))))
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            records.append(group_loss)
            total_candidates += candidate_count
        possible_pairs = 0
        seen_pairs = 0
        for values in source_groups.values():
            for name in values:
                entry = source_caches[next(k for k, v in source_groups.items() if name in v)].entry(name)
                n = len(_branch_indices(entry)) - 1
                possible_pairs += n + n * (n - 1) // 2
                seen_pairs += len(pair_seen[name])
        row: dict[str, float | int | str] = {
            "stage": "h360_control_oriented_full_candidate_objective",
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
        }
        if not math.isfinite(float(row["loss"])):
            raise RuntimeError("V127 H360 objective stage produced no finite loss")
        history.append(row)
        print("[V127_CONTROL_OBJECTIVE] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    return history


def _pairwise_accuracy(truth: np.ndarray, pred: np.ndarray, threshold: float = 0.0) -> float:
    hits = pairs = 0
    for i in range(len(truth)):
        for j in range(i + 1, len(truth)):
            delta = float(truth[i] - truth[j])
            if abs(delta) <= float(threshold):
                continue
            pairs += 1
            hits += int(np.sign(delta) == np.sign(float(pred[i] - pred[j])))
    return float(hits / pairs) if pairs else float("nan")


def evaluate_objective_groups_control_v127(
    model: Any,
    *,
    cache: Any,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    flood_rate_index: int,
    design: V127ControlTrainingDesign = V127ControlTrainingDesign(),
) -> dict[str, float | str]:
    """Report hard physical and smooth optimization ranking without replacing legacy keys."""
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
    with torch.no_grad():
        for name in names:
            data = _denormalize_group(cache.batch(name, normalization, device), normalization)
            truth = _truth_node_volume(cache, name).sum(axis=1).astype(np.float64)
            branch_count = int(data["settings"].shape[0])
            if truth.shape[0] != branch_count:
                raise RuntimeError(f"{name}: evaluation label/action order mismatch")
            hard_values: list[float] = []
            smooth_values: list[float] = []
            for start in range(0, branch_count, design.evaluation_branch_chunk):
                stop = min(start + design.evaluation_branch_chunk, branch_count)
                index = torch.arange(start, stop, dtype=torch.long, device=device)
                output = model.objective_rollout(
                    initial_state=data["initial"].index_select(0, index),
                    rainfall=data["rainfall"].index_select(0, index),
                    settings=data["settings"].index_select(0, index),
                    previous_actuator_flow=data["previous_flow"].index_select(0, index),
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
            span = max(float(np.max(truth) - optimum), 1.0)
            normalized_regret.append(selected_regret / span)

    def mean(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else float("nan")

    result: dict[str, float | str] = {
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
    }
    return result


__all__ = [
    "V127ControlTrainingDesign",
    "V127_CONTROL_TRAINING_CONTRACT",
    "evaluate_objective_groups_control_v127",
    "informative_pair_threshold_v127",
    "train_hydraulic_stage_chunked_v127",
    "train_objective_stage_control_v127",
    "train_truncated_rollout_stage_v127",
]
