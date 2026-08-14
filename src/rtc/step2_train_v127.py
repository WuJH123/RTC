from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .step2_train_response_v60 import InputNormalizationV60

V127_TRAINING_CONTRACT = "PROJECT7_V127_HYDRAULIC_THEN_DIFFERENTIABLE_COUNTERFACTUAL_FLOOD_OBJECTIVE_V4_LABEL_ALIGNED"


@dataclass(frozen=True)
class V127TrainingDesign:
    hydraulic_epochs: int = 3
    objective_epochs: int = 3
    learning_rate: float = 2.0e-4
    objective_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    grad_clip: float = 5.0
    teacher_stride: int = 4
    depth_weight: float = 2.0
    flood_rate_weight: float = 3.0
    flow_weight: float = 1.0
    node_flood_weight: float = 1.0
    tfv_weight: float = 1.0
    rollout_state_weight: float = 0.5
    pairwise_weight: float = 0.25
    rollout_pair_size: int = 2
    seed: int = 42

    def validate(self) -> None:
        if self.hydraulic_epochs <= 0 or self.objective_epochs <= 0:
            raise ValueError("V127 training epochs must be positive")
        if self.teacher_stride <= 0 or self.rollout_pair_size != 2:
            raise ValueError("V127 teacher stride/rollout pair contract invalid")
        for value in (
            self.learning_rate,
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
        ):
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError("V127 training design contains invalid values")
        if self.seed < 0:
            raise ValueError("V127 training seed must be non-negative")


def _stable_seed(text: str, base: int) -> int:
    return (
        int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) + int(base)
    ) % (2**32)


def _branch_indices(entry: Any) -> list[int]:
    """Return the only branch order used by V127 tensors and SWMM labels."""
    ref = int(entry.reference_index)
    candidates = [int(i) for i in entry.indices if int(i) != ref]
    if not candidates:
        raise ValueError("V127 counterfactual group has no candidates")
    return [ref, *candidates]


def _denormalize_group(
    batch: Any, normalization: InputNormalizationV60
) -> dict[str, torch.Tensor]:
    device, dtype = batch.initial_state.device, batch.initial_state.dtype

    def tensor(value: Any) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=device)

    initial = (
        batch.initial_state * tensor(normalization.state_std).clamp_min(1e-6)
        + tensor(normalization.state_mean)
    )
    rainfall = (
        batch.rainfall * tensor(normalization.rainfall_std).clamp_min(1e-6)
        + tensor(normalization.rainfall_mean)
    )
    previous_flow = (
        batch.previous_actuator_flow * tensor(normalization.flow_std).clamp_min(1e-6)
        + tensor(normalization.flow_mean)
    )
    states = torch.cat(
        (batch.true_reference_states, batch.true_candidate_states), dim=1
    )[0]
    flows = torch.cat(
        (batch.true_reference_flows, batch.true_candidate_flows), dim=1
    )[0]
    settings = torch.cat(
        (batch.reference_settings[:, None], batch.candidate_settings), dim=1
    )[0]
    branches = int(settings.shape[0])
    if states.shape[0] != branches or flows.shape[0] != branches:
        raise RuntimeError("V127 branch tensors disagree on reference/candidate order")
    return {
        "initial": initial.expand(branches, -1, -1).contiguous(),
        "rainfall": rainfall.expand(branches, -1, -1, -1).contiguous(),
        "previous_flow": previous_flow.expand(branches, -1).contiguous(),
        "states": states,
        "flows": flows,
        "settings": settings,
    }


def _static(graph: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "up": torch.as_tensor(graph.actuator_upstream, dtype=torch.long, device=device),
        "down": torch.as_tensor(graph.actuator_downstream, dtype=torch.long, device=device),
        "physics": torch.as_tensor(graph.actuator_physics, dtype=torch.float32, device=device),
        "static": torch.as_tensor(graph.static_node_features, dtype=torch.float32, device=device),
        "edges": torch.as_tensor(graph.edge_index, dtype=torch.long, device=device),
    }


def configure_model_normalization_v127(
    model: Any,
    *,
    normalization: InputNormalizationV60,
    graph: Any,
    state_delta_scale: np.ndarray,
    flow_delta_scale: np.ndarray,
) -> None:
    static = np.asarray(graph.static_node_features, dtype=np.float32)
    physics = np.asarray(graph.actuator_physics, dtype=np.float32)
    model.set_normalization(
        state_mean=torch.as_tensor(normalization.state_mean, dtype=torch.float32),
        state_std=torch.as_tensor(normalization.state_std, dtype=torch.float32),
        rain_mean=torch.as_tensor(normalization.rainfall_mean, dtype=torch.float32),
        rain_std=torch.as_tensor(normalization.rainfall_std, dtype=torch.float32),
        static_mean=torch.as_tensor(static.mean(axis=0), dtype=torch.float32),
        static_std=torch.as_tensor(static.std(axis=0).clip(min=1e-6), dtype=torch.float32),
        physics_mean=torch.as_tensor(physics.mean(axis=0), dtype=torch.float32),
        physics_std=torch.as_tensor(physics.std(axis=0).clip(min=1e-6), dtype=torch.float32),
        flow_std=torch.as_tensor(
            [float(np.mean(normalization.flow_std))], dtype=torch.float32
        ),
    )
    model.transition.set_delta_state_scale(
        torch.as_tensor(state_delta_scale, dtype=torch.float32)
    )
    model.actuator.set_delta_flow_scale(
        torch.as_tensor(flow_delta_scale, dtype=torch.float32)
    )


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
                state_samples.append(
                    np.abs(states - previous).reshape(-1, states.shape[-1])
                )
                flows = np.asarray(
                    arrays["target_actuator_flows"][index], dtype=np.float32
                )
                f0 = np.asarray(
                    arrays["previous_actuator_flow"][index], dtype=np.float32
                )
                fprev = np.concatenate((f0[None], flows[:-1]), axis=0)
                flow_samples.append(np.abs(flows - fprev))
    if not state_samples or not flow_samples:
        raise ValueError("V127 cannot derive residual scales from empty data")
    return (
        np.quantile(np.concatenate(state_samples, axis=0), 0.995, axis=0)
        .clip(min=1.0e-5)
        .astype(np.float32),
        np.quantile(np.concatenate(flow_samples, axis=0), 0.995, axis=0)
        .clip(min=1.0e-5)
        .astype(np.float32),
    )


def _ordered(
    source_groups: dict[str, Sequence[str]], epoch: int, seed: int
) -> list[tuple[str, str]]:
    pools = {key: list(values) for key, values in source_groups.items() if values}
    rng = np.random.default_rng(seed + 1009 * epoch)
    for values in pools.values():
        rng.shuffle(values)
    result: list[tuple[str, str]] = []
    while any(pools.values()):
        for key in sorted(pools):
            if pools[key]:
                result.append((key, pools[key].pop()))
    return result


def train_hydraulic_stage_v127(
    model: Any,
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
    np.random.seed(design.seed)
    model.train().to(device)
    static = _static(graph, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=design.learning_rate, weight_decay=design.weight_decay
    )
    state_dim = int(model.transition.state_mean.numel())
    weights = torch.ones(state_dim, device=device)
    if 0 <= depth_index < state_dim:
        weights[depth_index] = design.depth_weight
    if 0 <= flood_rate_index < state_dim:
        weights[flood_rate_index] = design.flood_rate_weight
    history: list[dict[str, float | int | str]] = []
    for epoch in range(1, design.hydraulic_epochs + 1):
        losses: list[float] = []
        by_source: dict[str, list[float]] = {key: [] for key in source_caches}
        for source, name in _ordered(source_groups, epoch, design.seed):
            data = _denormalize_group(
                source_caches[source].batch(name, normalization, device), normalization
            )
            branches, horizon = data["settings"].shape[:2]
            physics_norm, identity = model.actuator.prepare_static(
                static["physics"], batch_size=branches
            )
            static_norm, edges, inv = model.transition.prepare_static(
                static["static"],
                static["edges"],
                batch_size=branches,
                dtype=torch.float32,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = torch.zeros((), device=device)
            count = 0
            for k in range(
                (epoch - 1) % design.teacher_stride,
                horizon,
                design.teacher_stride,
            ):
                prev_state = data["initial"] if k == 0 else data["states"][:, k - 1]
                prev_flow = (
                    data["previous_flow"] if k == 0 else data["flows"][:, k - 1]
                )
                q, _ = model.actuator.forward_prepared(
                    prev_state[:, static["up"]],
                    prev_state[:, static["down"]],
                    data["settings"][:, k],
                    prev_flow,
                    physics_norm,
                    identity,
                )
                injection = torch.zeros(
                    branches, prev_state.shape[1], 1, device=device
                )
                injection = injection.index_add(
                    1, static["up"], -q[..., None]
                ).index_add(1, static["down"], q[..., None])
                action_context = model._setting_context(
                    data["settings"][:, k],
                    static["up"],
                    static["down"],
                    node_count=prev_state.shape[1],
                    dtype=prev_state.dtype,
                )
                pred_state = model.transition.forward_prepared(
                    prev_state,
                    data["rainfall"][:, k],
                    static_norm,
                    injection,
                    edges,
                    inv,
                    action_context,
                )
                state_error = (
                    pred_state - data["states"][:, k]
                ) / model.transition.state_std
                flow_error = (
                    q - data["flows"][:, k]
                ) / model.actuator.flow_std
                loss = loss + F.smooth_l1_loss(
                    state_error * weights,
                    torch.zeros_like(state_error),
                    beta=0.5,
                )
                loss = loss + design.flow_weight * F.smooth_l1_loss(
                    flow_error, torch.zeros_like(flow_error), beta=0.5
                )
                count += 1
            if count == 0:
                raise RuntimeError("V127 hydraulic stage selected no transitions")
            loss = loss / count
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"{name}: V127 hydraulic loss non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            value = float(loss.detach())
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
        print(
            "[V127_HYDRAULIC] "
            + " ".join(f"{k}={v}" for k, v in row.items()),
            flush=True,
        )
    return history


def _truth_node_volume(cache: Any, name: str) -> np.ndarray:
    entry = cache.entry(name)
    order = _branch_indices(entry)
    values = np.asarray(
        entry.arrays["exact_node_flood_volume_m3"][order], dtype=np.float32
    )
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < -1e-6):
        raise ValueError(f"{name}: authoritative node flood-volume labels are invalid")
    return np.clip(values, 0.0, None)


def train_objective_stage_v127(
    model: Any,
    *,
    source_caches: dict[str, Any],
    source_groups: dict[str, Sequence[str]],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    flood_rate_index: int,
    design: V127TrainingDesign = V127TrainingDesign(),
) -> list[dict[str, float | int | str]]:
    """Train H360 physical magnitude plus differentiable counterfactual action effects.

    The smooth Softplus flood proxy has a common positive zero-rate offset.  That offset is
    harmless for action ranking/gradients because every branch in a counterfactual group
    has the same node/time grid, but it must not be fitted as an absolute SWMM volume.  We
    therefore fit hard physical TFV to absolute SWMM truth and fit the smooth proxy only to
    branch-to-branch node-volume/TFV differences.  This preserves useful action gradients
    without creating an artificial absolute flood-volume target.
    """
    design.validate()
    static = _static(graph, device)
    model.train().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=design.objective_learning_rate,
        weight_decay=design.weight_decay,
    )
    history: list[dict[str, float | int | str]] = []
    for epoch in range(1, design.objective_epochs + 1):
        records: list[float] = []
        for source, name in _ordered(source_groups, epoch + 17, design.seed):
            data = _denormalize_group(
                source_caches[source].batch(name, normalization, device), normalization
            )
            truth_volume = torch.as_tensor(
                _truth_node_volume(source_caches[source], name),
                dtype=torch.float32,
                device=device,
            )
            branch_count = int(data["settings"].shape[0])
            if truth_volume.shape[0] != branch_count:
                raise RuntimeError(
                    f"{name}: SWMM label count does not match model branch order"
                )
            rng = np.random.default_rng(
                _stable_seed(name, design.seed + epoch * 7919)
            )
            pair = rng.choice(
                branch_count,
                size=min(design.rollout_pair_size, branch_count),
                replace=False,
            )
            if len(pair) < 2:
                continue
            index = torch.as_tensor(pair, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
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
            target_volume = truth_volume.index_select(0, index)
            true_tfv = target_volume.sum(-1)
            tfv_scale = torch.quantile(true_tfv.detach(), 0.75).clamp_min(100.0)
            # Absolute magnitude is evaluated in the physical hard-clamp operator.
            hard_tfv_loss = F.smooth_l1_loss(
                (output.tfv_m3 - true_tfv) / tfv_scale,
                torch.zeros_like(true_tfv),
                beta=0.5,
            )
            # The differentiable proxy is trained on counterfactual effects only; its
            # Softplus zero-rate offset cancels exactly on the common node/time grid.
            true_node_delta = target_volume[0] - target_volume[1]
            smooth_node_delta = (
                output.optimization_node_flood_volume_m3[0]
                - output.optimization_node_flood_volume_m3[1]
            )
            node_delta_scale = torch.quantile(
                torch.abs(true_node_delta.detach()), 0.75
            ).clamp_min(25.0)
            node_effect_loss = F.smooth_l1_loss(
                smooth_node_delta / node_delta_scale,
                true_node_delta / node_delta_scale,
                beta=0.5,
            )
            true_tfv_delta = (true_tfv[0] - true_tfv[1]).detach()
            smooth_tfv_delta = (
                output.optimization_tfv_m3[0] - output.optimization_tfv_m3[1]
            )
            tfv_effect_loss = F.smooth_l1_loss(
                smooth_tfv_delta / tfv_scale,
                true_tfv_delta / tfv_scale,
                beta=0.5,
            )
            if abs(float(true_tfv_delta)) > 1e-9:
                pair_loss = F.softplus(
                    -torch.sign(true_tfv_delta) * smooth_tfv_delta / tfv_scale
                )
            else:
                pair_loss = smooth_tfv_delta.new_zeros(())
            retained = torch.as_tensor(
                [5, 11, 23, 35, 47, 71], dtype=torch.long, device=device
            )
            true_states = data["states"].index_select(0, index).index_select(
                1, retained
            )
            pred_states = output.rollout.states.index_select(1, retained)
            state_loss = F.smooth_l1_loss(
                (pred_states - true_states) / model.transition.state_std,
                torch.zeros_like(pred_states),
                beta=0.5,
            )
            loss = (
                design.node_flood_weight * node_effect_loss
                + design.tfv_weight * (hard_tfv_loss + tfv_effect_loss)
                + design.pairwise_weight * pair_loss
                + design.rollout_state_weight * state_loss
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"{name}: V127 H360 objective loss non-finite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), design.grad_clip)
            optimizer.step()
            records.append(float(loss.detach()))
        row: dict[str, float | int | str] = {
            "stage": "h360_physical_magnitude_plus_differentiable_action_effect",
            "epoch": epoch,
            "loss": float(np.mean(records)) if records else float("nan"),
        }
        if not math.isfinite(float(row["loss"])):
            raise RuntimeError("V127 H360 objective stage produced no finite loss")
        history.append(row)
        print(
            "[V127_OBJECTIVE] "
            + " ".join(f"{k}={v}" for k, v in row.items()),
            flush=True,
        )
    return history


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, matching the statistical definition of Spearman rho."""
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    start = 0
    while start < x.size:
        end = start + 1
        while end < x.size and x[order[end]] == x[order[start]]:
            end += 1
        average = 0.5 * (start + end - 1)
        ranks[order[start:end]] = average
        start = end
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rankdata(a), _rankdata(b)
    if len(ra) < 2 or np.std(ra) <= 1e-12 or np.std(rb) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate_objective_groups_v127(
    model: Any,
    *,
    cache: Any,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    flood_rate_index: int,
) -> dict[str, float]:
    model.eval().to(device)
    static = _static(graph, device)
    rank: list[float] = []
    top1: list[float] = []
    mae: list[float] = []
    regret: list[float] = []
    pairwise: list[float] = []
    with torch.no_grad():
        for name in names:
            data = _denormalize_group(
                cache.batch(name, normalization, device), normalization
            )
            truth = _truth_node_volume(cache, name).sum(axis=1).astype(float)
            if truth.shape[0] != int(data["settings"].shape[0]):
                raise RuntimeError(f"{name}: evaluation label/action order mismatch")
            pred: list[float] = []
            for i in range(int(data["settings"].shape[0])):
                output = model.objective_rollout(
                    initial_state=data["initial"][i : i + 1],
                    rainfall=data["rainfall"][i : i + 1],
                    settings=data["settings"][i : i + 1],
                    previous_actuator_flow=data["previous_flow"][i : i + 1],
                    actuator_upstream=static["up"],
                    actuator_downstream=static["down"],
                    actuator_physics=static["physics"],
                    static_node_features=static["static"],
                    edge_index=static["edges"],
                    flood_rate_index=flood_rate_index,
                    priority_indices=None,
                    dt_seconds=300.0,
                )
                pred.append(float(output.tfv_m3[0]))
            predicted = np.asarray(pred, dtype=float)
            rank.append(_spearman(predicted, truth))
            selected = int(np.argmin(predicted))
            optimum_value = float(np.min(truth))
            optimum_mask = np.isclose(
                truth, optimum_value, rtol=0.0, atol=1e-9
            )
            top1.append(float(bool(optimum_mask[selected])))
            mae.append(float(np.mean(np.abs(predicted - truth))))
            regret.append(float(truth[selected] - optimum_value))
            pairs = hits = 0
            for i in range(len(truth)):
                for j in range(i + 1, len(truth)):
                    td, pd = truth[i] - truth[j], predicted[i] - predicted[j]
                    if abs(td) <= 1e-9:
                        continue
                    pairs += 1
                    hits += int(np.sign(td) == np.sign(pd))
            if pairs:
                pairwise.append(hits / pairs)

    def mean(values: Sequence[float]) -> float:
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        return float(arr.mean()) if arr.size else float("nan")

    return {
        "groups": float(len(names)),
        "rank": mean(rank),
        "pairwise": mean(pairwise),
        "top1": mean(top1),
        "tfv_mae_m3": mean(mae),
        "selected_regret_m3": mean(regret),
        "authoritative_label_order": "reference_then_candidates",
        "prediction_metric": "hard surrogate TFV vs authoritative SWMM cumulative TFV",
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
