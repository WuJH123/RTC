"""Evaluation for V9 signed counterfactual hydraulic effects and D2 localization."""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Sequence

import numpy as np
import torch

from .step2_control_response_v90 import DirectHydraulicEffectSurrogateV90, PreparedStaticV80
from .step2_train_response_v60 import InputNormalizationV60, V60TrainCache
from .step2_train_response_v70 import TargetScalesV70
from .step2_v90_contract import DirectHydraulicEffectLossContractV90, LEVEL_C


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
    scale: float,
    active_fraction: float,
    prefix: str,
) -> dict[str, float]:
    error = predicted - truth
    mse = float(np.mean(np.square(error)))
    zero_mse = float(np.mean(np.square(truth)))
    truth_abs = float(np.mean(np.abs(truth)))
    predicted_abs = float(np.mean(np.abs(predicted)))
    active = np.abs(truth) >= float(active_fraction) * max(float(scale), 1e-12)
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
    }


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


def _top_overlap(predicted: np.ndarray, truth: np.ndarray, k: int) -> float:
    k = min(int(k), int(predicted.size))
    if k <= 0:
        return float("nan")
    p = set(np.argpartition(np.abs(predicted), -k)[-k:].tolist())
    t = set(np.argpartition(np.abs(truth), -k)[-k:].tolist())
    return len(p & t) / float(k)


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
    p_active = p >= threshold
    t_active = t >= threshold
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
        f"{prefix}_top10_overlap": _top_overlap(p, t, 10),
        f"{prefix}_top20_overlap": _top_overlap(p, t, 20),
        f"{prefix}_active_node_precision": float(precision),
        f"{prefix}_active_node_recall": float(recall),
        f"{prefix}_effect_centroid_graph_distance_hops": centroid_distance,
    }


def _timing_record(
    predicted: np.ndarray,
    truth: np.ndarray,
    *,
    retained_indices: np.ndarray,
    scale: float,
    active_fraction: float,
    prefix: str,
) -> dict[str, float]:
    reduce_dims = tuple(range(1, predicted.ndim))
    p = np.mean(np.abs(predicted), axis=reduce_dims)
    t = np.mean(np.abs(truth), axis=reduce_dims)
    minutes = (np.asarray(retained_indices, dtype=np.float64) + 1.0) * 5.0
    peak_error = abs(
        float(minutes[int(np.argmax(p))]) - float(minutes[int(np.argmax(t))])
    )
    threshold = float(active_fraction) * max(float(scale), 1e-12)
    t_idx = np.flatnonzero(t >= threshold)
    p_idx = np.flatnonzero(p >= threshold)
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
                if actuator is not None:
                    record["actuator_index"] = actuator
                    record["actuator_id"] = str(graph.actuator_ids[actuator])
                    record["actuator_type"] = _actuator_type(graph, actuator)

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

                flow_scale = float(np.median(scales.flow_delta_scale))
                record.update(_effect_record(
                    predicted_flow, truth_flow,
                    scale=flow_scale,
                    active_fraction=contract.active_effect_fraction,
                    prefix="delta_managed_flow_m3s",
                ))
                record.update(_timing_record(
                    predicted_flow, truth_flow,
                    retained_indices=retained_indices,
                    scale=flow_scale,
                    active_fraction=contract.active_effect_fraction,
                    prefix="delta_managed_flow_m3s",
                ))
                records.append(record)

    excluded = {"event", "group", "actuator_id", "actuator_type", "actuator_index"}
    metric_names = sorted({
        key for record in records for key, value in record.items()
        if key not in excluded and isinstance(value, (int, float))
    })
    overall = {key: _event_balanced(records, key) for key in metric_names}

    by_type: dict[str, dict[str, float]] = {}
    for actuator_type in ("pump", "orifice", "weir"):
        subset = [record for record in records if record.get("actuator_type") == actuator_type]
        if subset:
            by_type[actuator_type] = {
                key: _event_balanced(subset, key) for key in metric_names
            }

    by_actuator: dict[str, dict[str, float]] = {}
    actuator_ids = sorted({
        str(record["actuator_id"]) for record in records if "actuator_id" in record
    })
    for actuator_id in actuator_ids:
        subset = [record for record in records if record.get("actuator_id") == actuator_id]
        by_actuator[actuator_id] = {
            key: _event_balanced(subset, key) for key in metric_names
        }

    return {
        "conditioning_level": model.conditioning_level,
        "groups": len(names),
        "records": len(records),
        "events": len({record["event"] for record in records}),
        "scientific_primary": "event_balanced_raw_signed_candidate_minus_reference_effect",
        "centroid_definition": "top20_effect_mass_weighted_graph_medoid",
        "overall": overall,
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
