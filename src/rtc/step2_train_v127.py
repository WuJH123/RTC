"""Training/evaluation for the Project7 V127 differentiable hydraulic surrogate.

The curriculum gives each existing dataset a mechanism-relevant role:

* D2: one-actuator hydraulic sensitivity / local Jacobian support.
* targeted D3: coordinated multi-actuator nonlinear response.
* D4-FIT: local first-move response support, re-used as ordinary physical trajectories;
  Sparse-RBC is not a target/reference concept inside V127.
* D4-AUDIT and all development holdouts remain read-only.
* D5, when supplied later, adds explicit central-difference directional-gradient loss.

The model is first trained with teacher-forced hydraulic transitions and actuator flows,
then with short batches of end-to-end H360 rollouts whose objective is exact node flood
volume / TFV.  This avoids the historical mistake of mixing unrelated trajectory and
Value losses in one undifferentiated update while still making the MPC objective a
function of the learned hydraulic dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .flood_volume import trapezoid_node_flood_volume
from .step2_train_response_v60 import InputNormalizationV60, V60TrainCache

V127_TRAINING_CONTRACT = "PROJECT7_V127_HYDRAULIC_THEN_FLOOD_OBJECTIVE_CURRICULUM_V1"


@dataclass(frozen=True)
class V127TrainingDesign:
    seed: int = 42
    hydraulic_epochs: int = 3
    objective_epochs: int = 4
    learning_rate: float = 3.0e-4
    objective_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    grad_clip: float = 5.0
    teacher_stride: int = 3
    rollout_pair_size: int = 2
    depth_weight: float = 2.0
    flood_rate_weight: float = 4.0
    flow_weight: float = 1.0
    node_flood_weight: float = 1.0
    tfv_weight: float = 1.0
    rollout_state_weight: float = 0.20
    pairwise_weight: float = 0.50

    def validate(self) -> None:
        if self.seed != 42:
            raise ValueError("V127 first formal-development recipe is frozen at seed 42")
        if self.hydraulic_epochs <= 0 or self.objective_epochs <= 0:
            raise ValueError("V127 training epochs must be positive")
        if self.teacher_stride <= 0 or self.rollout_pair_size != 2:
            raise ValueError("V127 teacher stride / rollout pair contract is invalid")
        for value in (
            self.learning_rate,
            self.objective_learning_rate,
            self.grad_clip,
            self.depth_weight,
            self.flood_rate_weight,
            self.flow_weight,
            self.node_flood_weight,
            self.tfv_weight,
            self.rollout_state_weight,
            self.pairwise_weight,
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("V127 training design contains invalid values")
        if self.learning_rate <= 0.0 or self.objective_learning_rate <= 0.0:
            raise ValueError("V127 learning rates must be positive")


def _denormalize_group(batch, normalization: InputNormalizationV60) -> dict[str, torch.Tensor]:
    device = batch.initial_state.device
    dtype = batch.initial_state.dtype
    sm = torch.as_tensor(normalization.state_mean, dtype=dtype, device=device)
    ss = torch.as_tensor(normalization.state_std, dtype=dtype, device=device).clamp_min(1e-6)
    rm = torch.as_tensor(normalization.rainfall_mean, dtype=dtype, device=device)
    rs = torch.as_tensor(normalization.rainfall_std, dtype=dtype, device=device).clamp_min(1e-6)
    fm = torch.as_tensor(normalization.flow_mean, dtype=dtype, device=device)
    fs = torch.as_tensor(normalization.flow_std, dtype=dtype, device=device).clamp_min(1e-6)
    initial = batch.initial_state * ss + sm
    rainfall = batch.rainfall * rs + rm
    previous_flow = batch.previous_actuator_flow * fs + fm
    states = torch.cat((batch.true_reference_states, batch.true_candidate_states), dim=1)[0]
    flows = torch.cat((batch.true_reference_flows, batch.true_candidate_flows), dim=1)[0]
    settings = torch.cat((batch.reference_settings[:, None], batch.candidate_settings), dim=1)[0]
    branches = int(settings.shape[0])
    return {
        "initial": initial.expand(branches, -1, -1).contiguous(),
        "rainfall": rainfall.expand(branches, -1, -1, -1).contiguous(),
        "previous_flow": previous_flow.expand(branches, -1).contiguous(),
        "states": states,
        "flows": flows,
        "settings": settings,
    }


def _graph_tensors(graph: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "up": torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device),
        "down": torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device),
        "physics": torch.as_tensor(graph.actuator_physics, dtype=torch.float32, device=device),
        "static": torch.as_tensor(graph.static_node_features, dtype=torch.float32, device=device),
        "edges": torch.as_tensor(graph.edge_index, dtype=torch.long, device=device),
    }


def configure_model_normalization_v127(
    model,
    *,
    normalization: InputNormalizationV60,
    graph: Any,
    state_delta_scale: np.ndarray,
    flow_delta_scale: np.ndarray,
) -> None:
    static = np.asarray(graph.static_node_features, dtype=np.float32)
    physics = np.asarray(graph.actuator_physics, dtype=np.float32)
    model.set_normalization(
        state_mean=torch.as_tensor(normalization.state_mean),
        state_std=torch.as_tensor(normalization.state_std),
        rain_mean=torch.as_tensor(normalization.rainfall_mean),
        rain_std=torch.as_tensor(normalization.rainfall_std),
        static_mean=torch.as_tensor(static.mean(axis=0)),
        static_std=torch.as_tensor(static.std(axis=0).clip(min=1e-6)),
        physics_mean=torch.as_tensor(physics.mean(axis=0)),
        physics_std=torch.as_tensor(physics.std(axis=0).clip(min=1e-6)),
        flow_std=torch.as_tensor([float(np.mean(np.asarray(normalization.flow_std)))], dtype=torch.float32),
    )
    model.transition.set_delta_state_scale(torch.as_tensor(state_delta_scale, dtype=torch.float32))
    model.actuator.set_delta_flow_scale(torch.as_tensor(flow_delta_scale, dtype=torch.float32))


def derive_residual_scales_v127(
    caches_and_names: Sequence[tuple[Any, Sequence[str]]],
) -> tuple[np.ndarray, np.ndarray]:
    state_samples: list[np.ndarray] = []
    flow_samples: list[np.ndarray] = []
    for cache, names in caches_and_names:
        for name in names:
            entry = cache.entry(name)
            arrays = entry.arrays
            for index in entry.indices:
                states = np.asarray(arrays["target_states"][index], dtype=np.float32)
                initial = np.asarray(arrays["initial_state"][index], dtype=np.float32)
                previous = np.concatenate((initial[None], states[:-1]), axis=0)
                state_samples.append(np.abs(states - previous).reshape(-1, states.shape[-1]))
                flows = np.asarray(arrays["target_actuator_flows"][index], dtype=np.float32)
                f0 = np.asarray(arrays["previous_actuator_flow"][index], dtype=np.float32)
                fprev = np.concatenate((f0[None], flows[:-1]), axis=0)
                flow_samples.append(np.abs(flows - fprev))
    if not state_samples or not flow_samples:
        raise ValueError("V127 cannot derive bounded residual scales from empty TrainFit data")
    state = np.concatenate(state_samples, axis=0)
    flow = np.concatenate(flow_samples, axis=0)
    if not np.isfinite(state).all() or not np.isfinite(flow).all():
        raise ValueError("V127 residual-scale source contains non-finite values")
    state_scale = np.quantile(state, 0.995, axis=0).astype(np.float32)
    flow_scale = np.quantile(flow, 0.995, axis=0).astype(np.float32)
    return np.maximum(state_scale, 1.0e-4), np.maximum(flow_scale, 1.0e-4)


def _ordered_sources(source_groups: dict[str, Sequence[str]], epoch: int, seed: int) -> list[tuple[str, str]]:
    rng = np.random.default_rng(seed + 1009 * epoch)
    pools = {key: list(values) for key, values in source_groups.items() if values}
    for values in pools.values():
        rng.shuffle(values)
    result: list[tuple[str, str]] = []
    while any(pools.values()):
        for key in sorted(pools):
            if pools[key]:
                result.append((key, pools[key].pop()))
    return result


def train_hydraulic_stage_v127(
    model,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    depth_index: int,
    flood_rate_index: int,
    design: V127TrainingDesign = V127TrainingDesign(),
) -> list[dict[str, float | int | str]]:
    design.validate()
    torch.manual_seed(design.seed)
    model.train().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=design.learning_rate, weight_decay=design.weight_decay
    )
    static = _graph_tensors(graph, device)
    history: list[dict[str, float | int | str]] = []
    state_dim = int(model.transition.state_mean.numel())
    weights = torch.ones(state_dim, dtype=torch.float32, device=device)
    if 0 <= int(depth_index) < state_dim:
        weights[int(depth_index)] = float(design.depth_weight)
    if 0 <= int(flood_rate_index) < state_dim:
        weights[int(flood_rate_index)] = float(design.flood_rate_weight)

    for epoch in range(1, design.hydraulic_epochs + 1):
        losses: list[float] = []
        by_source: dict[str, list[float]] = {key: [] for key in source_caches}
        for source, name in _ordered_sources(source_groups, epoch, design.seed):
            batch = source_caches[source].batch(name, normalization, device)
            data = _denormalize_group(batch, normalization)
            branches, horizon = data["settings"].shape[:2]
            physics_norm, identity = model.actuator.prepare_static(static["physics"], batch_size=branches)
            static_norm, edges, inv = model.transition.prepare_static(
                static["static"], static["edges"], batch_size=branches, dtype=torch.float32
            )
            # Three epochs with stride 3 cover every 5-min transition exactly once before
            # any repeated exposure, while keeping memory bounded by one true state step.
            steps = range((epoch - 1) % design.teacher_stride, horizon, design.teacher_stride)
            optimizer.zero_grad(set_to_none=True)
            group_loss = torch.zeros((), dtype=torch.float32, device=device)
            count = 0
            for k in steps:
                state_prev = data["initial"] if k == 0 else data["states"][:, k - 1]
                flow_prev = data["previous_flow"] if k == 0 else data["flows"][:, k - 1]
                q, _ = model.actuator.forward_prepared(
                    state_prev[:, static["up"]],
                    state_prev[:, static["down"]],
                    data["settings"][:, k],
                    flow_prev,
                    physics_norm,
                    identity,
                )
                injection = torch.zeros(
                    branches,
                    state_prev.shape[1],
                    1,
                    dtype=state_prev.dtype,
                    device=device,
                )
                injection = injection.index_add(1, static["up"], -q[..., None])
                injection = injection.index_add(1, static["down"], q[..., None])
                action_context = model._setting_context(
                    data["settings"][:, k],
                    static["up"],
                    static["down"],
                    node_count=state_prev.shape[1],
                    dtype=state_prev.dtype,
                )
                pred_state = model.transition.forward_prepared(
                    state_prev,
                    data["rainfall"][:, k],
                    static_norm,
                    injection,
                    edges,
                    inv,
                    action_context,
                )
                state_error = (pred_state - data["states"][:, k]) / model.transition.state_std
                state_loss = F.smooth_l1_loss(
                    state_error * weights,
                    torch.zeros_like(state_error),
                    beta=0.5,
                )
                flow_error = (q - data["flows"][:, k]) / model.actuator.flow_std
                flow_loss = F.smooth_l1_loss(
                    flow_error, torch.zeros_like(flow_error), beta=0.5
                )
                group_loss = group_loss + state_loss + float(design.flow_weight) * flow_loss
                count += 1
            if count <= 0:
                raise RuntimeError("V127 teacher-forced stage selected no transitions")
            group_loss = group_loss / count
            if not bool(torch.isfinite(group_loss)):
                raise RuntimeError(f"{name}: V127 hydraulic loss is non-finite")
            group_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            value = float(group_loss.detach())
            losses.append(value)
            by_source[source].append(value)
        row: dict[str, float | int | str] = {
            "stage": "hydraulic_teacher_forced",
            "epoch": epoch,
            "loss": float(np.mean(losses)),
        }
        for source, values in by_source.items():
            if values:
                row[f"loss_{source.lower()}"] = float(np.mean(values))
        history.append(row)
        print("[V127_HYDRAULIC] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    return history


def _true_group_node_volumes(cache, name: str) -> np.ndarray:
    entry = cache.entry(name)
    return np.asarray(
        entry.arrays["exact_node_flood_volume_m3"][list(entry.indices)], dtype=np.float32
    )


def train_objective_stage_v127(
    model,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    flood_rate_index: int,
    design: V127TrainingDesign = V127TrainingDesign(),
) -> list[dict[str, float | int | str]]:
    """Fine-tune the hydraulic model through the exact flood-volume control objective."""
    design.validate()
    static = _graph_tensors(graph, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=design.objective_learning_rate, weight_decay=design.weight_decay
    )
    history: list[dict[str, float | int | str]] = []
    model.train().to(device)
    for epoch in range(1, design.objective_epochs + 1):
        records: list[float] = []
        for source, name in _ordered_sources(source_groups, epoch + 17, design.seed):
            batch = source_caches[source].batch(name, normalization, device)
            data = _denormalize_group(batch, normalization)
            truth_volume = torch.as_tensor(
                _true_group_node_volumes(source_caches[source], name),
                dtype=torch.float32,
                device=device,
            )
            branch_count = int(data["settings"].shape[0])
            rng = np.random.default_rng(design.seed + epoch * 7919 + abs(hash(name)) % 100000)
            # One antithetic pair of branches per group/epoch keeps H360 backprop memory
            # bounded while preserving within-state action comparison.
            pair = rng.choice(branch_count, size=min(design.rollout_pair_size, branch_count), replace=False)
            if len(pair) < 2:
                continue
            index = torch.as_tensor(pair, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            rollout = model.rollout(
                data["initial"].index_select(0, index),
                data["rainfall"].index_select(0, index),
                data["settings"].index_select(0, index),
                data["previous_flow"].index_select(0, index),
                static["up"],
                static["down"],
                static["physics"],
                static["static"],
                static["edges"],
            )
            pred_volume = trapezoid_node_flood_volume(
                data["initial"].index_select(0, index),
                rollout.states,
                flood_rate_index=int(flood_rate_index),
                dt_seconds=300.0,
            )
            target_volume = truth_volume.index_select(0, index)
            volume_scale = torch.quantile(target_volume.detach().reshape(-1), 0.75).clamp_min(100.0)
            node_loss = F.smooth_l1_loss(
                torch.log1p(pred_volume.clamp_min(0.0) / volume_scale),
                torch.log1p(target_volume.clamp_min(0.0) / volume_scale),
                beta=0.25,
            )
            pred_tfv = pred_volume.sum(dim=-1)
            true_tfv = target_volume.sum(dim=-1)
            tfv_scale = torch.quantile(true_tfv.detach(), 0.75).clamp_min(100.0)
            tfv_loss = F.smooth_l1_loss(
                (pred_tfv - true_tfv) / tfv_scale,
                torch.zeros_like(pred_tfv),
                beta=0.5,
            )
            truth_diff = (true_tfv[0] - true_tfv[1]).detach() / tfv_scale
            pred_diff = (pred_tfv[0] - pred_tfv[1]) / tfv_scale
            pair_loss = F.softplus(-torch.sign(truth_diff) * pred_diff)
            # A sparse state anchor keeps the H360 rollout physically tied to SWMM without
            # resurrecting the historical all-target joint-loss conflict.
            retained = torch.as_tensor([5, 11, 23, 35, 47, 71], dtype=torch.long, device=device)
            true_states = data["states"].index_select(0, index).index_select(1, retained)
            pred_states = rollout.states.index_select(1, retained)
            state_loss = F.smooth_l1_loss(
                (pred_states - true_states) / model.transition.state_std,
                torch.zeros_like(pred_states),
                beta=0.5,
            )
            loss = (
                float(design.node_flood_weight) * node_loss
                + float(design.tfv_weight) * tfv_loss
                + float(design.pairwise_weight) * pair_loss
                + float(design.rollout_state_weight) * state_loss
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"{name}: V127 objective loss is non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            records.append(float(loss.detach()))
        row = {
            "stage": "h360_flood_objective",
            "epoch": epoch,
            "loss": float(np.mean(records)) if records else float("nan"),
        }
        history.append(row)
        print("[V127_OBJECTIVE] " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    return history


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(_rankdata(a), _rankdata(b))[0, 1])


def evaluate_objective_groups_v127(
    model,
    *,
    cache,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    flood_rate_index: int,
) -> dict[str, float]:
    model.eval().to(device)
    static = _graph_tensors(graph, device)
    rank: list[float] = []
    top1: list[float] = []
    mae: list[float] = []
    regret: list[float] = []
    pairwise: list[float] = []
    with torch.no_grad():
        for name in names:
            batch = cache.batch(name, normalization, device)
            data = _denormalize_group(batch, normalization)
            truth = _true_group_node_volumes(cache, name).sum(axis=1).astype(np.float64)
            pred: list[float] = []
            for index in range(int(data["settings"].shape[0])):
                out = model.rollout(
                    data["initial"][index:index+1],
                    data["rainfall"][index:index+1],
                    data["settings"][index:index+1],
                    data["previous_flow"][index:index+1],
                    static["up"], static["down"], static["physics"], static["static"], static["edges"],
                )
                volume = trapezoid_node_flood_volume(
                    data["initial"][index:index+1], out.states,
                    flood_rate_index=int(flood_rate_index), dt_seconds=300.0,
                )
                pred.append(float(volume.sum().detach()))
            predicted = np.asarray(pred, dtype=np.float64)
            rank.append(_spearman(predicted, truth))
            selected = int(np.argmin(predicted))
            optimum = int(np.argmin(truth))
            top1.append(float(selected == optimum))
            mae.append(float(np.mean(np.abs(predicted - truth))))
            regret.append(float(truth[selected] - truth[optimum]))
            pairs = 0
            correct = 0
            for i in range(len(truth)):
                for j in range(i + 1, len(truth)):
                    td = truth[i] - truth[j]
                    pd = predicted[i] - predicted[j]
                    if abs(td) <= 1e-9:
                        continue
                    pairs += 1
                    correct += int(np.sign(td) == np.sign(pd))
            if pairs:
                pairwise.append(correct / pairs)
    def mean(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else float("nan")
    return {
        "groups": float(len(names)),
        "rank": mean(rank),
        "pairwise": mean(pairwise),
        "top1": mean(top1),
        "tfv_mae_m3": mean(mae),
        "selected_regret_m3": mean(regret),
    }


__all__ = [
    "V127TrainingDesign",
    "V127_TRAINING_CONTRACT",
    "configure_model_normalization_v127",
    "derive_residual_scales_v127",
    "evaluate_objective_groups_v127",
    "train_hydraulic_stage_v127",
    "train_objective_stage_v127",
]
