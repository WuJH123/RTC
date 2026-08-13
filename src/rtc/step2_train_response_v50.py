"""Independent V5.0 value/hydraulic training contracts."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .step2_control_response_v50 import StateConditionedCandidateManifoldModelV50
from .step2_counterfactual import counterfactual_groups, reference_index
from .step2_training_cache import load_step2_training_cache
from .step2_train_response_v41 import (
    CounterfactualDeltaScalesV41,
    ResponseGroupBatchV41,
    ResponseNormalizationV4,
    evaluate_response_groups_v41,
    group_metrics_v41,
    prepare_graph_v41,
    stack_response_group_v41,
)
from .step2_train_response_v4 import ResponsePairV4


VALUE_LOSS_CONTRACT_V50 = "STEP2_V50_VALUE_LOSS_CONTRACT"
TRAJECTORY_LOSS_CONTRACT_V50 = "STEP2_V50_TRAJECTORY_LOSS_CONTRACT"
TRAINING_CONTRACT_V50 = "PROJECT7_STEP2_V50_CANDIDATE_MANIFOLD_DIRECT_RESPONSE"


def _normalise_row_v50(row: dict[str, np.ndarray], norm: ResponseNormalizationV4) -> dict[str, np.ndarray]:
    def normal(value: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return ((value - mean) / np.maximum(std, 1e-6)).astype(np.float32, copy=False)

    return {
        "initial_state": normal(row["initial_state"], norm.state_mean, norm.state_std),
        "rainfall": normal(row["rainfall"], norm.rain_mean, norm.rain_std),
        "settings": np.ascontiguousarray(row["settings"], dtype=np.float32),
        "previous_actuator_flow": normal(row["previous_actuator_flow"], np.zeros_like(norm.flow_std), norm.flow_std),
        "target_states": normal(row["target_states"], norm.state_mean, norm.state_std),
        "target_actuator_flows": normal(row["target_actuator_flows"], np.zeros_like(norm.flow_std), norm.flow_std),
        "elapsed_seconds": np.asarray(row["elapsed_seconds"], dtype=np.float32),
        "target_states_physical": np.asarray(row["target_states"], dtype=np.float32),
        "initial_state_physical": np.asarray(row["initial_state"], dtype=np.float32),
        "target_actuator_flows_physical": np.asarray(row["target_actuator_flows"], dtype=np.float32),
        "exact_node_flood_volume_m3": np.asarray(row["exact_node_flood_volume_m3"], dtype=np.float32),
    }


class LazyTrainGroupsV50:
    """Memory-bounded view over the rebuildable Train cache.

    Group arrays remain mmap-backed and are materialised only while one optimizer
    step is running.  This is necessary because the full 72x932 target trajectories
    are intentionally not duplicated into a Python-wide list.
    """

    def __init__(self, cache_manifest: str, normalization: ResponseNormalizationV4, names: list[str] | None = None) -> None:
        self.normalization = normalization
        cache = load_step2_training_cache(cache_manifest)
        index: dict[str, tuple[dict[str, np.ndarray], list[int], int]] = {}
        for item in cache["shards"]:
            arrays = item["arrays"]
            for group, indices in counterfactual_groups(arrays).items():
                if names is not None and group not in names:
                    continue
                index[group] = (arrays, list(indices), reference_index(arrays, indices))
        if names is not None:
            missing = sorted(set(names) - set(index))
            if missing:
                raise ValueError(f"selected Train-only groups missing from cache: {missing[:5]}")
        self._index = index

    def keys(self):
        return self._index.keys()

    def __iter__(self):
        return iter(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, key: str) -> bool:
        return key in self._index

    def __getitem__(self, group: str) -> list[ResponsePairV4]:
        arrays, indices, ref_index = self._index[group]
        source = group.split("::", 1)[0]
        names = {
            "initial_state", "rainfall", "settings", "previous_actuator_flow",
            "target_states", "target_actuator_flows", "elapsed_seconds",
            "exact_node_flood_volume_m3",
        }

        def row(index: int) -> dict[str, np.ndarray]:
            return {name: np.asarray(arrays[name][index]) for name in names}

        reference = _normalise_row_v50(row(ref_index), self.normalization)
        pairs: list[ResponsePairV4] = []
        for candidate in sorted(indices):
            if candidate == ref_index:
                continue
            pairs.append(ResponsePairV4(source, group, reference, _normalise_row_v50(row(candidate), self.normalization)))
        return pairs

    def subset(self, names: list[str]) -> "LazyTrainGroupsV50":
        # Reuse the same cache path/index contract without copying arrays.  The
        # parent index is intentionally filtered after construction.
        subset = object.__new__(LazyTrainGroupsV50)
        subset.normalization = self.normalization
        subset._index = {name: self._index[name] for name in names}
        return subset


@dataclass(frozen=True)
class ValueLossWeightsV50:
    exact_delta_tfv: float = 1.0
    group_centered: float = 0.5
    listwise_rank: float = 1.0
    topk_regret_surrogate: float = 0.25


def listwise_pl_loss_v50(predicted: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    """Plackett-Luce/ListMLE loss for a complete same-prefix candidate group."""

    if predicted.shape != truth.shape or predicted.ndim != 2:
        raise ValueError("predicted and truth must both be [B,C]")
    target_order = torch.argsort(truth.detach(), dim=1, descending=False)
    remaining = predicted
    losses: list[torch.Tensor] = []
    for position in range(predicted.shape[1]):
        index = target_order[:, position]
        log_prob = F.log_softmax(-remaining, dim=1).gather(1, index[:, None]).squeeze(1)
        losses.append(-log_prob)
        if position + 1 < predicted.shape[1]:
            keep = torch.ones_like(remaining, dtype=torch.bool)
            keep.scatter_(1, index[:, None], False)
            remaining = remaining.masked_fill(~keep, float("inf"))
    return torch.stack(losses, dim=1).mean()


def value_loss_components_v50(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    *,
    source_scale: torch.Tensor | float,
    weights: ValueLossWeightsV50 = ValueLossWeightsV50(),
) -> tuple[torch.Tensor, dict[str, float]]:
    """Exact-value + centred + listwise objective with no trajectory terms."""

    scale = torch.as_tensor(source_scale, device=predicted.device, dtype=predicted.dtype).reshape(()).clamp_min(1.0)
    exact = F.smooth_l1_loss((predicted - truth.detach()) / scale, torch.zeros_like(predicted))
    group_scale = torch.maximum(
        truth.detach().amax(dim=1) - truth.detach().amin(dim=1),
        0.05 * scale,
    ).clamp_min(1.0)
    centred = F.smooth_l1_loss(
        (predicted - predicted.mean(dim=1, keepdim=True) - truth.detach() + truth.detach().mean(dim=1, keepdim=True))
        / group_scale[:, None],
        torch.zeros_like(predicted),
    )
    listwise = listwise_pl_loss_v50(predicted / scale, truth)
    best_truth = truth.detach().amin(dim=1, keepdim=True)
    soft_best = -torch.logsumexp(-predicted / scale, dim=1, keepdim=True)
    topk = F.smooth_l1_loss((soft_best - best_truth) / scale, torch.zeros_like(soft_best))
    total = (
        weights.exact_delta_tfv * exact
        + weights.group_centered * centred
        + weights.listwise_rank * listwise
        + weights.topk_regret_surrogate * topk
    )
    return total, {
        "loss": float(total.detach()),
        "exact_delta_tfv": float(exact.detach()),
        "group_centered": float(centred.detach()),
        "listwise_rank": float(listwise.detach()),
        "topk_regret_surrogate": float(topk.detach()),
        "tfv_mae_m3": float((predicted.detach() - truth.detach()).abs().mean()),
        "response_ratio": float(predicted.detach().abs().mean() / truth.detach().abs().mean().clamp_min(1.0)),
    }


def hydraulic_loss_components_v50(
    output,
    batch: ResponseGroupBatchV41,
    *,
    state_scale: torch.Tensor,
    flow_scale: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Independent diagnostic hydraulic loss; it never receives value gradients."""

    state = torch.as_tensor(state_scale, device=output.delta_states_physical.device, dtype=output.delta_states_physical.dtype).reshape(1, 1, 1, 1, -1).clamp_min(1e-6)
    flow = torch.as_tensor(flow_scale, device=output.delta_flows_physical.device, dtype=output.delta_flows_physical.dtype).reshape(1, 1, 1, -1).clamp_min(1e-6)
    true_state = batch.true_delta_states_physical
    true_flow = batch.true_delta_flows_physical
    state_loss = F.smooth_l1_loss(output.delta_states_physical / state, true_state / state)
    flow_loss = F.smooth_l1_loss(output.delta_flows_physical / flow, true_flow / flow)
    total = state_loss + flow_loss
    return total, {
        "loss": float(total.detach()),
        "delta_state": float(state_loss.detach()),
        "delta_flow": float(flow_loss.detach()),
    }


def deterministic_group_split_v50(grouped_pairs: Any, *, holdout_fraction: float = 0.2) -> tuple[Any, Any]:
    """Split whole rainfall/event groups deterministically, never individual rows."""

    if not 0.0 < holdout_fraction < 0.5:
        raise ValueError("holdout_fraction must lie in (0, 0.5)")
    fit_names: list[str] = []
    holdout_names: list[str] = []
    for name in sorted(grouped_pairs):
        # Hash the rainfall/event identity, not outcomes or model metrics.
        digest = int(hashlib.sha256(name.split("::", 1)[-1].encode("utf-8")).hexdigest()[:8], 16)
        if (digest % 10000) < int(round(holdout_fraction * 10000)):
            holdout_names.append(name)
        else:
            fit_names.append(name)
    if not fit_names or not holdout_names:
        names = sorted(grouped_pairs)
        pivot = max(1, int(round(len(names) * (1.0 - holdout_fraction))))
        fit_names = names[:pivot]
        holdout_names = names[pivot:]
    if hasattr(grouped_pairs, "subset"):
        return grouped_pairs.subset(fit_names), grouped_pairs.subset(holdout_names)
    return ({name: grouped_pairs[name] for name in fit_names}, {name: grouped_pairs[name] for name in holdout_names})


def select_mechanism_mini_pack_v50(grouped_pairs: Any, *, per_source: int = 3) -> list[str]:
    """Select forcing/state-only deterministic representatives, never by outcomes."""

    selected: list[str] = []
    for source in ("D2", "D3"):
        names = sorted(name for name in grouped_pairs if name.startswith(source + "::"))
        if not names:
            continue
        positions = np.linspace(0, len(names) - 1, num=min(per_source, len(names)), dtype=int)
        selected.extend(names[int(index)] for index in positions)
    return list(dict.fromkeys(selected))


def _group_names(grouped_pairs: Any) -> list[str]:
    return sorted(grouped_pairs)


def _group_pairs(grouped_pairs: Any, name: str) -> list[Any]:
    return grouped_pairs[name]


def _value_batches(grouped_pairs: Any, device: torch.device) -> dict[str, ResponseGroupBatchV41]:
    """Compatibility helper for small eager subsets only."""
    return {
        name: stack_response_group_v41(_group_pairs(grouped_pairs, name), device)
        for name in _group_names(grouped_pairs)
    }


def _source_scale(scales: CounterfactualDeltaScalesV41, source: str) -> float:
    return float(scales.by_source[source.upper()].tfv_scale_m3)


def train_control_value_v50(
    *,
    model: StateConditionedCandidateManifoldModelV50,
    fit_groups: dict[str, list[Any]],
    d2_anchor_groups: dict[str, list[Any]],
    normalization: ResponseNormalizationV4,
    scales: CounterfactualDeltaScalesV41,
    graph: Any,
    stage_a_epochs: int = 6,
    stage_b_epochs: int = 8,
    learning_rate: float = 2e-3,
    device: str = "cuda",
    seed: int = 42,
) -> dict[str, Any]:
    """Train value-only stages A/B with source- and group-balanced exact TFV targets."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(target).float()
    model.freeze_reference()
    model.set_value_trainable(True)
    prepared = prepare_graph_v41(model.reference_backbone, graph, normalization, target)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-5)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    def run_epoch(epoch: int, names: list[str], *, anchors: list[str] | None = None) -> dict[str, Any]:
        optimizer.zero_grad(set_to_none=True)
        ordered = list(names)
        np.random.default_rng(seed + epoch).shuffle(ordered)
        if anchors:
            ordered.extend(anchors)
        losses: list[float] = []
        components: dict[str, list[float]] = {}
        for name in ordered:
            source_groups = d2_anchor_groups if name in d2_anchor_groups and name not in fit_groups else fit_groups
            batch = stack_response_group_v41(_group_pairs(source_groups, name), target)
            source = batch.source_kind.upper()
            output = model.forward_value_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind=source,
            )
            loss, metrics = value_loss_components_v50(
                output.direct_delta_tfv_m3,
                batch.true_delta_tfv_m3,
                source_scale=_source_scale(scales, source),
            )
            # Equal group weighting: average after all group losses have contributed.
            (loss / max(len(ordered), 1)).backward()
            losses.append(float(loss.detach()))
            for key, value in metrics.items():
                components.setdefault(key, []).append(float(value))
        optimizer.step()
        return {
            "epoch": int(epoch),
            "groups": len(ordered),
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "components": {key: float(np.mean(values)) for key, values in components.items()},
        }

    d2_names = _group_names(d2_anchor_groups)
    for epoch in range(1, int(stage_a_epochs) + 1):
        history.append({"stage": "A_D2_sensitivity", **run_epoch(epoch, d2_names)})
    for offset in range(1, int(stage_b_epochs) + 1):
        epoch = int(stage_a_epochs) + offset
        names = _group_names(fit_groups)
        anchors = d2_names[: max(1, min(len(d2_names), len(names)))]
        history.append({"stage": "B_joint_action", **run_epoch(epoch, names, anchors=anchors)})
    return {
        "contract": TRAINING_CONTRACT_V50,
        "value_loss_contract": VALUE_LOSS_CONTRACT_V50,
        "trajectory_loss_contract": TRAJECTORY_LOSS_CONTRACT_V50,
        "reference_frozen": True,
        "hydraulic_gradient_into_value": False,
        "stage_a_epochs": int(stage_a_epochs),
        "stage_b_epochs": int(stage_b_epochs),
        "fit_groups": len(fit_groups),
        "d2_anchor_groups": len(d2_anchor_groups),
        "history": history,
        "wall_seconds": float(time.perf_counter() - started),
    }


def train_hydraulic_response_v50(
    *,
    model: StateConditionedCandidateManifoldModelV50,
    groups: dict[str, list[Any]],
    normalization: ResponseNormalizationV4,
    scales: CounterfactualDeltaScalesV41,
    graph: Any,
    epochs: int = 4,
    learning_rate: float = 2e-3,
    device: str = "cuda",
    seed: int = 42,
) -> dict[str, Any]:
    """Train only the hydraulic head on an independent optimizer/objective."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(target).float()
    model.freeze_reference()
    model.set_hydraulic_trainable(True)
    prepared = prepare_graph_v41(model.reference_backbone, graph, normalization, target)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=1e-5)
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(epochs) + 1):
        optimizer.zero_grad(set_to_none=True)
        losses: list[float] = []
        names = _group_names(groups)
        for name in names:
            batch = stack_response_group_v41(_group_pairs(groups, name), target)
            output = model.forward_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind=batch.source_kind,
            )
            source = batch.source_kind.upper()
            loss, metrics = hydraulic_loss_components_v50(
                output, batch,
                state_scale=(scales.by_source[source].state_scale if source in scales.by_source else torch.ones(model.state_dim)),
                flow_scale=(scales.by_source[source].flow_scale if source in scales.by_source else torch.ones(model.actuator_count)),
            )
            (loss / max(len(names), 1)).backward()
            losses.append(float(loss.detach()))
        optimizer.step()
        history.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else float("nan"), "groups": len(names)})
    return {
        "contract": TRAJECTORY_LOSS_CONTRACT_V50,
        "value_gradient_received": False,
        "epochs": int(epochs),
        "groups": len(groups),
        "history": history,
    }


def evaluate_value_groups_v50(
    *,
    model: StateConditionedCandidateManifoldModelV50,
    grouped_pairs: dict[str, list[Any]],
    normalization: ResponseNormalizationV4,
    graph: Any,
    device: str = "cuda",
) -> list[dict[str, Any]]:
    target = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    model.to(target).eval()
    prepared = prepare_graph_v41(model.reference_backbone, graph, normalization, target)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for name in _group_names(grouped_pairs):
            batch = stack_response_group_v41(_group_pairs(grouped_pairs, name), target)
            output = model.forward_value_group(
                batch.initial_state, batch.rainfall, batch.reference_settings,
                batch.candidate_settings, batch.previous_actuator_flow, prepared,
                batch.elapsed_seconds, source_kind=batch.source_kind,
            )
            rows.append(group_metrics_v41(
                predicted=output.direct_delta_tfv_m3[0].cpu().numpy(),
                truth=batch.true_delta_tfv_m3[0].cpu().numpy(),
                group=name,
                source_kind=batch.source_kind,
            ))
    return rows


def aggregate_metrics_v50(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"groups": 0}
    return {
        "groups": len(rows),
        "rank": float(np.nanmean([row["rank"] for row in rows])),
        "pairwise": float(np.nanmean([row["pairwise"] for row in rows])),
        "sign": float(np.nanmean([row["sign"] for row in rows])),
        "top1": int(sum(bool(row["top1"]) for row in rows)),
        "mean_regret_m3": float(np.mean([row["regret_m3"] for row in rows])),
        "max_regret_m3": float(np.max([row["regret_m3"] for row in rows])),
        "spread_ratio": float(np.nanmean([row["spread_ratio"] for row in rows])),
        "tfv_mae_m3": float(np.mean([row["mae_m3"] for row in rows])),
    }


__all__ = [
    "TRAINING_CONTRACT_V50",
    "TRAJECTORY_LOSS_CONTRACT_V50",
    "VALUE_LOSS_CONTRACT_V50",
    "ValueLossWeightsV50",
    "aggregate_metrics_v50",
    "deterministic_group_split_v50",
    "evaluate_value_groups_v50",
    "hydraulic_loss_components_v50",
    "listwise_pl_loss_v50",
    "select_mechanism_mini_pack_v50",
    "train_control_value_v50",
    "train_hydraulic_response_v50",
    "value_loss_components_v50",
]
