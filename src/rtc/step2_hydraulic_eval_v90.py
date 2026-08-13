"""Evaluation for V9 signed counterfactual hydraulic effects and D2 localization."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
import torch

from .step2_control_response_v90 import DirectHydraulicEffectSurrogateV90, PreparedStaticV80
from .step2_train_response_v60 import InputNormalizationV60, V60TrainCache
from .step2_train_response_v70 import TargetScalesV70
from .step2_v90_contract import DirectHydraulicEffectLossContractV90, LEVEL_C


# Fixed diagnostic-only slices.  They are not model-selection knobs and are
# deliberately expressed as elapsed prediction end-times, matching the H360 cache.
HORIZON_BUCKETS_V90 = (
    ("0_30_min", 0.0, 30.0),
    ("30_120_min", 30.0, 120.0),
    ("120_360_min", 120.0, 360.0),
)


def _event_balanced(records: Sequence[dict[str, Any]], key: str) -> float:
    by_event: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = float(record.get(key, float("nan")))
        if np.isfinite(value):
            by_event[str(record["event"])].append(value)
    values = [float(np.mean(items)) for items in by_event.values() if items]
    return float(np.mean(values)) if values else float("nan")


def _effect_record(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    scale: float | np.ndarray,
    active_fraction: float,
    prefix: str,
) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if predicted.shape != truth.shape:
        raise ValueError("V9 effect metric prediction/truth shape mismatch")
    try:
        scale_array = np.broadcast_to(
            np.maximum(np.asarray(scale, dtype=np.float64), 1e-12), truth.shape
        )
    except ValueError as exc:
        raise ValueError("V9 effect metric scale cannot broadcast to target shape") from exc
    error = predicted - truth
    mse = float(np.mean(np.square(error)))
    zero_mse = float(np.mean(np.square(truth)))
    truth_abs = float(np.mean(np.abs(truth)))
    predicted_abs = float(np.mean(np.abs(predicted)))
    # Flow targets have a distinct RMS scale for every actuator.  Using a global
    # median changes which physical effects are called active, so all sparse metrics
    # must normalize elementwise before thresholding.
    active = np.abs(truth) / scale_array >= float(active_fraction)
    active_mse = float(np.mean(np.square(error[active]))) if active.any() else float("nan")
    active_zero = float(np.mean(np.square(truth[active]))) if active.any() else float("nan")
    active_sign = (
        float(np.mean(np.sign(predicted[active]) == np.sign(truth[active])))
        if active.any()
        else float("nan")
    )
    return {
        f"{prefix}_rmse": float(np.sqrt(mse)),
        f"{prefix}_response_ratio": predicted_abs / max(truth_abs, 1e-12),
        f"{prefix}_skill_vs_zero": 1.0 - mse / max(zero_mse, 1e-12),
        f"{prefix}_active_skill_vs_zero": (
            1.0 - active_mse / max(active_zero, 1e-12)
            if np.isfinite(active_mse)
            else float("nan")
        ),
        f"{prefix}_active_sign": active_sign,
        f"{prefix}_active_fraction": float(active.mean()),
    }


@dataclass
class _PooledEffectStatistics:
    """Sufficient physical-error statistics for one event and one channel.

    Candidate-level ratios are undefined when a valid counterfactual happens to
    have zero true effect.  Pool the numerator and denominator within an event
    before taking skill/response ratios, then event-balance those quantities.
    This keeps a real zero-effect candidate from creating an arbitrary ``1e-12``
    denominator that dominates every other candidate in its rainfall event.
    """

    count: int = 0
    squared_error: float = 0.0
    squared_truth: float = 0.0
    absolute_prediction: float = 0.0
    absolute_truth: float = 0.0
    active_count: int = 0
    active_squared_error: float = 0.0
    active_squared_truth: float = 0.0
    active_sign_correct: int = 0

    def update(
        self,
        predicted: np.ndarray,
        truth: np.ndarray,
        *,
        scale: float | np.ndarray,
        active_fraction: float,
    ) -> None:
        predicted = np.asarray(predicted, dtype=np.float64)
        truth = np.asarray(truth, dtype=np.float64)
        if predicted.shape != truth.shape:
            raise ValueError("V9 pooled effect prediction/truth shape mismatch")
        try:
            scale_array = np.broadcast_to(
                np.maximum(np.asarray(scale, dtype=np.float64), 1e-12), truth.shape
            )
        except ValueError as exc:
            raise ValueError("V9 pooled effect scale cannot broadcast to target shape") from exc
        if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
            raise ValueError("V9 pooled effect metrics require finite physical tensors")
        error = predicted - truth
        self.count += int(truth.size)
        self.squared_error += float(np.square(error).sum())
        self.squared_truth += float(np.square(truth).sum())
        self.absolute_prediction += float(np.abs(predicted).sum())
        self.absolute_truth += float(np.abs(truth).sum())
        active = np.abs(truth) / scale_array >= float(active_fraction)
        if active.any():
            self.active_count += int(active.sum())
            self.active_squared_error += float(np.square(error[active]).sum())
            self.active_squared_truth += float(np.square(truth[active]).sum())
            self.active_sign_correct += int(
                (np.sign(predicted[active]) == np.sign(truth[active])).sum()
            )


def _summarize_pooled_event_effects(
    by_event: dict[str, _PooledEffectStatistics], *, prefix: str
) -> dict[str, float]:
    """Return event-balanced metrics from pooled physical sufficient statistics."""

    def average(values: list[float]) -> float:
        finite = [value for value in values if np.isfinite(value)]
        return float(np.mean(finite)) if finite else float("nan")

    rows = [stats for _, stats in sorted(by_event.items()) if stats.count]
    rmse = [math.sqrt(stats.squared_error / stats.count) for stats in rows]
    response = [
        stats.absolute_prediction / stats.absolute_truth
        for stats in rows
        if stats.absolute_truth > 1e-12
    ]
    skill = [
        1.0 - stats.squared_error / stats.squared_truth
        for stats in rows
        if stats.squared_truth > 1e-12
    ]
    active_skill = [
        1.0 - stats.active_squared_error / stats.active_squared_truth
        for stats in rows
        if stats.active_squared_truth > 1e-12
    ]
    active_sign = [
        stats.active_sign_correct / stats.active_count
        for stats in rows
        if stats.active_count > 0
    ]
    active_fraction = [stats.active_count / stats.count for stats in rows]
    return {
        f"{prefix}_rmse": average(rmse),
        f"{prefix}_response_ratio": average(response),
        f"{prefix}_skill_vs_zero": average(skill),
        f"{prefix}_active_skill_vs_zero": average(active_skill),
        f"{prefix}_active_sign": average(active_sign),
        f"{prefix}_active_fraction": average(active_fraction),
        f"{prefix}_events_with_truth_support": float(len(skill)),
        f"{prefix}_events_total": float(len(rows)),
    }


def _pooled_event_effect_metrics(
    records: Sequence[tuple[str, np.ndarray, np.ndarray, float | np.ndarray]],
    *,
    active_fraction: float,
    prefix: str,
) -> dict[str, float]:
    """Small testable adapter for event-pooled physical effect semantics."""

    by_event: dict[str, _PooledEffectStatistics] = {}
    for event, predicted, truth, scale in records:
        by_event.setdefault(str(event), _PooledEffectStatistics()).update(
            predicted, truth, scale=scale, active_fraction=active_fraction
        )
    return _summarize_pooled_event_effects(by_event, prefix=prefix)


def _accumulate_event_effect(
    storage: dict[tuple[str | None, str], dict[str, _PooledEffectStatistics]],
    *,
    bucket: str | None,
    prefix: str,
    event: str,
    predicted: np.ndarray,
    truth: np.ndarray,
    scale: float | np.ndarray,
    active_fraction: float,
) -> None:
    by_event = storage.setdefault((bucket, prefix), {})
    by_event.setdefault(event, _PooledEffectStatistics()).update(
        predicted, truth, scale=scale, active_fraction=active_fraction
    )


def _adjacency(edge_index: np.ndarray, node_count: int) -> list[list[int]]:
    adjacency = [[] for _ in range(node_count)]
    for src, dst in np.asarray(edge_index, dtype=np.int64).T:
        src_i, dst_i = int(src), int(dst)
        adjacency[src_i].append(dst_i)
        adjacency[dst_i].append(src_i)
    return adjacency


def _distance_vector(
    adjacency: list[list[int]],
    source: int,
    cache: dict[int, np.ndarray],
) -> np.ndarray:
    if source in cache:
        return cache[source]
    distance = np.full(len(adjacency), np.inf, dtype=np.float64)
    distance[int(source)] = 0.0
    queue = deque([int(source)])
    while queue:
        node = queue.popleft()
        for nxt in adjacency[node]:
            if not np.isfinite(distance[nxt]):
                distance[nxt] = distance[node] + 1.0
                queue.append(nxt)
    cache[int(source)] = distance
    return distance


def _graph_effect_medoid(
    effect_mass: np.ndarray,
    *,
    adjacency: list[list[int]],
    distance_cache: dict[int, np.ndarray],
    support_k: int = 20,
) -> int | None:
    """Weighted graph medoid of the strongest effect mass.

    Using the strongest 20 nodes makes the metric robust to tiny dense surrogate noise
    while measuring a real graph centroid rather than merely the peak node.
    """
    mass = np.asarray(effect_mass, dtype=np.float64).reshape(-1)
    positive = np.flatnonzero(mass > 0.0)
    if positive.size == 0:
        return None
    k = min(int(support_k), int(positive.size))
    support = positive[np.argpartition(mass[positive], -k)[-k:]]
    weights = mass[support]
    weights = weights / max(float(weights.sum()), 1e-12)
    best_node: int | None = None
    best_cost = float("inf")
    for candidate in support.tolist():
        distances = _distance_vector(adjacency, int(candidate), distance_cache)[support]
        finite = np.isfinite(distances)
        if not finite.all():
            continue
        cost = float(np.sum(weights * distances))
        if cost < best_cost:
            best_cost = cost
            best_node = int(candidate)
    return best_node


def _top_overlap(
    predicted: np.ndarray,
    truth: np.ndarray,
    k: int,
    *,
    support_threshold: float = 0.0,
) -> float:
    """Top-K overlap only where a meaningful effect exists.

    ``argpartition`` over an all-zero target is non-deterministic with respect to
    semantic node identity: arbitrary zero ties must never become apparent
    localization skill.  Empty truth support is therefore not applicable; a real
    truth support with empty prediction support scores zero.
    """
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    truth = np.asarray(truth, dtype=np.float64).reshape(-1)
    if predicted.shape != truth.shape:
        raise ValueError("V9 Top-K prediction/truth shape mismatch")
    if not np.isfinite(predicted).all() or not np.isfinite(truth).all():
        return float("nan")
    if int(k) <= 0:
        return float("nan")
    threshold = max(float(support_threshold), 0.0)
    truth_support = np.flatnonzero(np.abs(truth) > threshold)
    if truth_support.size == 0:
        return float("nan")
    predicted_support = np.flatnonzero(np.abs(predicted) > threshold)
    if predicted_support.size == 0:
        return 0.0
    selected = min(int(k), int(truth_support.size))
    truth_top = truth_support[
        np.argpartition(np.abs(truth[truth_support]), -selected)[-selected:]
    ]
    predicted_selected = min(selected, int(predicted_support.size))
    predicted_top = predicted_support[
        np.argpartition(np.abs(predicted[predicted_support]), -predicted_selected)[
            -predicted_selected:
        ]
    ]
    return len(set(predicted_top.tolist()) & set(truth_top.tolist())) / float(selected)


def _localization_record(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    scale: float,
    active_fraction: float,
    adjacency: list[list[int]],
    distance_cache: dict[int, np.ndarray],
    prefix: str,
) -> dict[str, float]:
    p = np.mean(np.abs(predicted), axis=0)
    t = np.mean(np.abs(truth), axis=0)
    threshold = float(active_fraction) * max(float(scale), 1e-12)
    p_active = p > threshold
    t_active = t > threshold
    tp = int(np.logical_and(p_active, t_active).sum())
    precision = tp / max(int(p_active.sum()), 1)
    recall = tp / max(int(t_active.sum()), 1)

    p_centroid = _graph_effect_medoid(
        p, adjacency=adjacency, distance_cache=distance_cache
    )
    t_centroid = _graph_effect_medoid(
        t, adjacency=adjacency, distance_cache=distance_cache
    )
    if p_centroid is None or t_centroid is None:
        centroid_distance = float("nan")
    else:
        centroid_distance = float(
            _distance_vector(adjacency, p_centroid, distance_cache)[t_centroid]
        )
        if not np.isfinite(centroid_distance):
            centroid_distance = float("nan")

    return {
        f"{prefix}_top10_overlap": _top_overlap(
            p, t, 10, support_threshold=threshold
        ),
        f"{prefix}_top20_overlap": _top_overlap(
            p, t, 20, support_threshold=threshold
        ),
        f"{prefix}_active_node_precision": float(precision),
        f"{prefix}_active_node_recall": float(recall),
        f"{prefix}_effect_centroid_graph_distance_hops": centroid_distance,
    }


def _timing_record(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    retained_indices: np.ndarray,
    scale: float | np.ndarray,
    active_fraction: float,
    prefix: str,
) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if predicted.shape != truth.shape or predicted.ndim < 1:
        raise ValueError("V9 timing prediction/truth shape mismatch")
    try:
        scale_array = np.broadcast_to(
            np.maximum(np.asarray(scale, dtype=np.float64), 1e-12), truth.shape
        )
    except ValueError as exc:
        raise ValueError("V9 timing scale cannot broadcast to target shape") from exc
    reduce_dims = tuple(range(1, predicted.ndim))
    p = np.mean(np.abs(predicted), axis=reduce_dims)
    t = np.mean(np.abs(truth), axis=reduce_dims)
    minutes = (np.asarray(retained_indices, dtype=np.float64) + 1.0) * 5.0
    peak_error = abs(
        float(minutes[int(np.argmax(p))]) - float(minutes[int(np.argmax(t))])
    )
    # Peak timing stays a separate total-response-mass diagnostic.  Sparse onset
    # instead uses a fixed normalized spatial maximum, so one physically active
    # flooded/storage node cannot be erased by 931 inactive nodes.
    normalized_predicted = np.abs(predicted) / scale_array
    normalized_truth = np.abs(truth) / scale_array
    if reduce_dims:
        p_onset = np.max(normalized_predicted, axis=reduce_dims)
        t_onset = np.max(normalized_truth, axis=reduce_dims)
    else:  # scalar one-dimensional time series
        p_onset = normalized_predicted
        t_onset = normalized_truth
    t_idx = np.flatnonzero(t_onset >= float(active_fraction))
    p_idx = np.flatnonzero(p_onset >= float(active_fraction))
    if t_idx.size:
        truth_onset = float(minutes[int(t_idx[0])])
        predicted_onset = (
            float(minutes[int(p_idx[0])]) if p_idx.size else float(minutes[-1] + 5.0)
        )
        onset_error = abs(predicted_onset - truth_onset)
    else:
        onset_error = float("nan")
    return {
        f"{prefix}_peak_effect_timing_error_min": peak_error,
        f"{prefix}_response_onset_timing_error_min": onset_error,
    }


def _horizon_bucket_masks(retained_indices: np.ndarray) -> dict[str, np.ndarray]:
    """Return the fixed exhaustive H360 diagnostic partition."""
    indices = np.asarray(retained_indices, dtype=np.int64).reshape(-1)
    minutes = (indices.astype(np.float64) + 1.0) * 5.0
    masks = {
        name: (minutes > start) & (minutes <= end)
        for name, start, end in HORIZON_BUCKETS_V90
    }
    membership = np.zeros(indices.shape, dtype=np.int64)
    for mask in masks.values():
        membership += mask.astype(np.int64)
    if not np.all(membership == 1):
        raise ValueError("V9 retained horizon does not partition into fixed H360 buckets")
    return masks


def _bucket_effect_records(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    retained_indices: np.ndarray,
    scale: float | np.ndarray,
    active_fraction: float,
    prefix: str,
) -> dict[str, dict[str, float]]:
    masks = _horizon_bucket_masks(retained_indices)
    return {
        label: _effect_record(
            predicted[mask],
            truth[mask],
            scale=scale,
            active_fraction=active_fraction,
            prefix=prefix,
        )
        for label, mask in masks.items()
    }


def _actuator_type(graph: Any, actuator_index: int) -> str:
    names = tuple(graph.actuator_physics_feature_names)
    row = np.asarray(graph.actuator_physics[actuator_index])
    for feature, label in (
        ("is_pump", "pump"),
        ("is_orifice", "orifice"),
        ("is_weir", "weir"),
    ):
        if feature in names and row[names.index(feature)] > 0.5:
            return label
    return "other"


def _single_changed_actuator(batch, candidate_index: int) -> int | None:
    delta = batch.candidate_settings[:, candidate_index] - batch.reference_settings
    active = torch.any(delta.abs() > 1e-8, dim=(0, 1)).nonzero(as_tuple=False).flatten()
    return int(active.item()) if active.numel() == 1 else None


def evaluate_hydraulic_effect_v90(
    model: DirectHydraulicEffectSurrogateV90,
    cache: V60TrainCache,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    prepared: PreparedStaticV80,
    scales: TargetScalesV70,
    *,
    graph: Any,
    device: torch.device | str,
    contract: DirectHydraulicEffectLossContractV90 = DirectHydraulicEffectLossContractV90(),
) -> dict[str, Any]:
    """Event-balanced signed-effect metrics plus spatial/temporal localization."""
    target = torch.device(device)
    model.to(target).eval()
    adjacency = _adjacency(np.asarray(graph.edge_index), len(graph.node_ids))
    distance_cache: dict[int, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    effect_prefixes = (
        "delta_depth_m",
        "delta_flood_m3s",
        "delta_storage_m3",
        "delta_managed_flow_m3s",
    )
    pooled_effects: dict[tuple[str | None, str], dict[str, _PooledEffectStatistics]] = {}
    pooled_by_type: dict[str, dict[tuple[str | None, str], dict[str, _PooledEffectStatistics]]] = defaultdict(dict)
    pooled_by_actuator: dict[str, dict[tuple[str | None, str], dict[str, _PooledEffectStatistics]]] = defaultdict(dict)

    with torch.no_grad():
        for name in names:
            batch = cache.batch(name, normalization, target)
            kwargs = {}
            if model.conditioning_level == LEVEL_C:
                kwargs = {
                    "oracle_reference_states_physical": batch.true_reference_states,
                    "oracle_reference_flows_physical": batch.true_reference_flows,
                }
            output = model(
                batch.initial_state,
                batch.rainfall,
                batch.reference_settings,
                batch.candidate_settings,
                batch.previous_actuator_flow,
                prepared,
                **kwargs,
            )
            indices = output.horizon_indices
            true_reference = batch.true_reference_states.index_select(1, indices)[:, None]
            true_reference = true_reference.expand_as(output.reference_states_physical)
            true_state_delta = (
                batch.true_candidate_states.index_select(2, indices) - true_reference
            )
            true_reference_flow = batch.true_reference_flows.index_select(1, indices)[:, None]
            true_reference_flow = true_reference_flow.expand_as(output.reference_flows_physical)
            true_flow_delta = (
                batch.true_candidate_flows.index_select(2, indices) - true_reference_flow
            )
            event = f"{cache.entry(name).rainfall_group}::{cache.entry(name).event_id}"
            retained_indices = indices.detach().cpu().numpy()
            bucket_masks = _horizon_bucket_masks(retained_indices)

            for candidate in range(output.raw_delta_states_physical.shape[1]):
                predicted_state = (
                    output.raw_delta_states_physical[0, candidate]
                    .detach().cpu().numpy().astype(np.float64)
                )
                truth_state = (
                    true_state_delta[0, candidate]
                    .detach().cpu().numpy().astype(np.float64)
                )
                predicted_flow = (
                    output.raw_delta_flows_physical[0, candidate]
                    .detach().cpu().numpy().astype(np.float64)
                )
                truth_flow = (
                    true_flow_delta[0, candidate]
                    .detach().cpu().numpy().astype(np.float64)
                )
                record: dict[str, Any] = {"event": event, "group": name}
                actuator = _single_changed_actuator(batch, candidate)
                actuator_type: str | None = None
                actuator_id: str | None = None
                if actuator is not None:
                    record["actuator_index"] = actuator
                    actuator_id = str(graph.actuator_ids[actuator])
                    actuator_type = _actuator_type(graph, actuator)
                    record["actuator_id"] = actuator_id
                    record["actuator_type"] = actuator_type

                def accumulate(
                    bucket: str | None,
                    prefix: str,
                    predicted: np.ndarray,
                    truth: np.ndarray,
                    scale: float | np.ndarray,
                ) -> None:
                    kwargs = {
                        "bucket": bucket,
                        "prefix": prefix,
                        "event": event,
                        "predicted": predicted,
                        "truth": truth,
                        "scale": scale,
                        "active_fraction": contract.active_effect_fraction,
                    }
                    _accumulate_event_effect(pooled_effects, **kwargs)
                    if actuator_type is not None:
                        _accumulate_event_effect(pooled_by_type[actuator_type], **kwargs)
                    if actuator_id is not None:
                        _accumulate_event_effect(pooled_by_actuator[actuator_id], **kwargs)

                for channel, label in (
                    (0, "depth_m"),
                    (2, "flood_m3s"),
                    (3, "storage_m3"),
                ):
                    channel_scale = float(scales.state_delta_scale[channel])
                    predicted = predicted_state[..., channel]
                    truth = truth_state[..., channel]
                    prefix = f"delta_{label}"
                    record.update(_effect_record(
                        predicted, truth,
                        scale=channel_scale,
                        active_fraction=contract.active_effect_fraction,
                        prefix=prefix,
                    ))
                    accumulate(None, prefix, predicted, truth, channel_scale)
                    record.update(_localization_record(
                        predicted, truth,
                        scale=channel_scale,
                        active_fraction=contract.active_effect_fraction,
                        adjacency=adjacency,
                        distance_cache=distance_cache,
                        prefix=prefix,
                    ))
                    record.update(_timing_record(
                        predicted, truth,
                        retained_indices=retained_indices,
                        scale=channel_scale,
                        active_fraction=contract.active_effect_fraction,
                        prefix=prefix,
                    ))
                    for bucket, metrics in _bucket_effect_records(
                        predicted,
                        truth,
                        retained_indices=retained_indices,
                        scale=channel_scale,
                        active_fraction=contract.active_effect_fraction,
                        prefix=prefix,
                    ).items():
                        record.update({f"{bucket}__{key}": value for key, value in metrics.items()})
                        mask = bucket_masks[bucket]
                        accumulate(bucket, prefix, predicted[mask], truth[mask], channel_scale)

                # Preserve raw physical flow metrics while using the matching scale of
                # each individual actuator for sparse active/sign diagnostics.
                flow_scale = np.asarray(scales.flow_delta_scale, dtype=np.float64)[None, :]
                record.update(_effect_record(
                    predicted_flow, truth_flow,
                    scale=flow_scale,
                    active_fraction=contract.active_effect_fraction,
                    prefix="delta_managed_flow_m3s",
                ))
                accumulate(None, "delta_managed_flow_m3s", predicted_flow, truth_flow, flow_scale)
                record.update(_timing_record(
                    predicted_flow, truth_flow,
                    retained_indices=retained_indices,
                    scale=flow_scale,
                    active_fraction=contract.active_effect_fraction,
                    prefix="delta_managed_flow_m3s",
                ))
                for bucket, metrics in _bucket_effect_records(
                    predicted_flow,
                    truth_flow,
                    retained_indices=retained_indices,
                    scale=flow_scale,
                    active_fraction=contract.active_effect_fraction,
                    prefix="delta_managed_flow_m3s",
                ).items():
                    record.update({f"{bucket}__{key}": value for key, value in metrics.items()})
                    mask = bucket_masks[bucket]
                    accumulate(
                        bucket,
                        "delta_managed_flow_m3s",
                        predicted_flow[mask],
                        truth_flow[mask],
                        flow_scale,
                    )
                records.append(record)

    excluded = {"event", "group", "actuator_id", "actuator_type", "actuator_index"}
    metric_names = sorted({
        key for record in records for key, value in record.items()
        if key not in excluded and "__" not in key and isinstance(value, (int, float))
    })
    overall = {key: _event_balanced(records, key) for key in metric_names}
    for prefix in effect_prefixes:
        overall.update(_summarize_pooled_event_effects(
            pooled_effects.get((None, prefix), {}), prefix=prefix
        ))

    horizon_buckets: dict[str, dict[str, float]] = {}
    for bucket, _, _ in HORIZON_BUCKETS_V90:
        prefix = bucket + "__"
        bucket_metric_names = sorted({
            key for record in records for key, value in record.items()
            if key.startswith(prefix) and isinstance(value, (int, float))
        })
        horizon_buckets[bucket] = {
            key[len(prefix):]: _event_balanced(records, key)
            for key in bucket_metric_names
        }
        for effect_prefix in effect_prefixes:
            horizon_buckets[bucket].update(_summarize_pooled_event_effects(
                pooled_effects.get((bucket, effect_prefix), {}), prefix=effect_prefix
            ))

    by_type: dict[str, dict[str, float]] = {}
    for actuator_type in ("pump", "orifice", "weir"):
        subset = [record for record in records if record.get("actuator_type") == actuator_type]
        if subset:
            by_type[actuator_type] = {
                key: _event_balanced(subset, key) for key in metric_names
            }
            for prefix in effect_prefixes:
                by_type[actuator_type].update(_summarize_pooled_event_effects(
                    pooled_by_type[actuator_type].get((None, prefix), {}), prefix=prefix
                ))

    by_actuator: dict[str, dict[str, float]] = {}
    actuator_ids = sorted({
        str(record["actuator_id"]) for record in records if "actuator_id" in record
    })
    for actuator_id in actuator_ids:
        subset = [record for record in records if record.get("actuator_id") == actuator_id]
        by_actuator[actuator_id] = {
            key: _event_balanced(subset, key) for key in metric_names
        }
        for prefix in effect_prefixes:
            by_actuator[actuator_id].update(_summarize_pooled_event_effects(
                pooled_by_actuator[actuator_id].get((None, prefix), {}), prefix=prefix
            ))

    return {
        "conditioning_level": model.conditioning_level,
        "groups": len(names),
        "records": len(records),
        "events": len({record["event"] for record in records}),
        "scientific_primary": "event_balanced_raw_signed_candidate_minus_reference_effect",
        "centroid_definition": "top20_effect_mass_weighted_graph_medoid",
        "onset_timing_definition": "max_spatial_abs_effect_normalized_by_trainfit_scale",
        "horizon_bucket_contract": {
            "units": "prediction end-time minutes",
            "buckets": [
                {"name": name, "start_exclusive_min": start, "end_inclusive_min": end}
                for name, start, end in HORIZON_BUCKETS_V90
            ],
        },
        "overall": overall,
        "horizon_buckets": horizon_buckets,
        "by_actuator_type": by_type,
        "by_actuator_identity": by_actuator,
    }


PRIMARY_SKILL_KEYS = (
    "delta_depth_m_skill_vs_zero",
    "delta_flood_m3s_skill_vs_zero",
    "delta_storage_m3_skill_vs_zero",
    "delta_managed_flow_m3s_skill_vs_zero",
)


def decide_state_sufficiency_v90(
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply the fixed Train-only A/B/C development decision margins."""
    def skills(level: str) -> np.ndarray:
        overall = results[level]["overall"]
        return np.asarray(
            [float(overall.get(key, np.nan)) for key in PRIMARY_SKILL_KEYS],
            dtype=np.float64,
        )

    a = skills("A_BOUNDARY")
    b = skills("B_PREDICTED_REFERENCE_TRAJECTORY")
    c = skills("C_ORACLE_REFERENCE_TRAJECTORY")
    if not np.all(np.isfinite(np.concatenate((a, b, c)))):
        return {
            "decision": "INCONCLUSIVE_MISSING_METRICS",
            "next_step": "repair/evaluate existing D2 ladder only",
        }

    mean_a, mean_b, mean_c = float(a.mean()), float(b.mean()), float(c.mean())
    if np.all(c <= 0.0):
        decision = "MARKOV_INSUFFICIENCY_SUPPORTED"
        next_step = (
            "audit existing 60-min pre-action history cache; do not run new SWMM yet"
        )
    else:
        b_beats_a = int(np.sum(b > a)) >= 3 and mean_b - mean_a >= 0.05
        c_beats_b = int(np.sum(c > b)) >= 3 and mean_c - mean_b >= 0.05
        if mean_c > mean_a + 1e-12:
            closure = (mean_b - mean_a) / (mean_c - mean_a)
        else:
            closure = 1.0 if mean_b >= mean_c - 0.05 else 0.0
        b_near_c = closure >= 0.70 or mean_b >= mean_c - 0.05

        if b_beats_a and b_near_c:
            decision = "PREDICTED_REFERENCE_TRAJECTORY_SUFFICIENT"
            next_step = (
                "train formal V9 Level B on D2=4 then D3=10 with frozen 0.75/0.25 mixture"
            )
        elif c_beats_b:
            decision = "REFERENCE_HYDRAULIC_ACCURACY_PRIMARY_BOTTLENECK"
            next_step = (
                "improve frozen-reference Hydraulic accuracy before adding history or new SWMM"
            )
        else:
            decision = "INCONCLUSIVE_EXISTING_INFORMATION"
            next_step = (
                "audit existing pre-action history and current-link-flow recoverability "
                "before any new SWMM"
            )

    return {
        "decision": decision,
        "mean_primary_skill": {"A": mean_a, "B": mean_b, "C": mean_c},
        "per_channel_skill": {
            "A": a.tolist(),
            "B": b.tolist(),
            "C": c.tolist(),
        },
        "next_step": next_step,
    }


__all__ = [
    "PRIMARY_SKILL_KEYS",
    "decide_state_sufficiency_v90",
    "evaluate_hydraulic_effect_v90",
]
