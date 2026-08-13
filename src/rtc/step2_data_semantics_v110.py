"""Mechanism-aware data semantics for Project7 Step2 V11.

This module extracts *training/diagnostic* descriptors from the frozen D2/D3
counterfactual groups. Target-side descriptors use authoritative SWMM outcomes
and therefore must never be supplied as online MPC inputs. Causal descriptors
(action exposure and current reference-state phase) are computed separately so
that the boundary is explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .step2_control_response_v60 import PreparedStaticV60
from .step2_control_response_v110 import (
    STATE_EFFECT_CHANNELS_V110,
    ActuatorNodeRelationsV110,
)
from .step2_hydraulic_objective_v110 import EffectScalesV110
from .step2_train_response_v60 import V60TrainCache
from .step2_v110_contract import HydraulicHorizonV110

V110_DATA_SEMANTICS_CONTRACT = "PROJECT7_STEP2_V110_MECHANISM_DATA_SEMANTICS_V1"


@dataclass(frozen=True)
class CandidateMechanismRecordV110:
    source_kind: str
    group_name: str
    event_id: str
    rainfall_group: str
    checkpoint_id: str
    candidate_index: int
    changed_actuator_count: int
    action_l1_exposure: float
    action_l2_exposure: float
    max_abs_setting_delta: float
    first_change_min: float
    last_change_min: float
    active_duration_min: float
    action_transition_count: int
    action_reversal_count: int
    peak_simultaneous_changed_actuators: int
    reference_depth_fill_mean: float
    reference_depth_fill_p90: float
    reference_near_surcharge_fraction: float
    reference_flood_active_fraction: float
    reference_storage_fill_mean: float
    reference_storage_near_capacity_fraction: float
    reference_net_inflow_m3s_mean: float
    hydraulic_effect_energy: float
    onset_min: float
    peak_response_min: float
    remote_effect_fraction_gt8: float
    delta_tfv_m3: float


def _finite_or_zero(value: np.ndarray, denominator: np.ndarray, *, eps: float = 1.0e-9) -> np.ndarray:
    return np.divide(value, denominator, out=np.zeros_like(value, dtype=np.float64), where=np.abs(denominator) > eps)


def action_sequence_descriptors_v110(
    reference_settings: np.ndarray,
    candidate_settings: np.ndarray,
    *,
    model_step_seconds: int = 300,
    epsilon: float = 1.0e-7,
) -> dict[str, np.ndarray]:
    """Causal action-sequence descriptors for [C,H,A] candidate schedules."""
    reference = np.asarray(reference_settings, dtype=np.float64)
    candidate = np.asarray(candidate_settings, dtype=np.float64)
    if reference.ndim != 2 or candidate.ndim != 3:
        raise ValueError("V11 action descriptors require [H,A] reference and [C,H,A] candidates")
    if candidate.shape[1:] != reference.shape:
        raise ValueError("V11 action descriptor candidate/reference shapes differ")
    delta = candidate - reference[None]
    changed = np.abs(delta) > float(epsilon)
    any_step = changed.any(axis=2)
    any_actuator = changed.any(axis=1)
    count = any_actuator.sum(axis=1).astype(np.int64)
    l1 = np.abs(delta).sum(axis=(1, 2))
    l2 = np.sqrt(np.square(delta).sum(axis=(1, 2)))
    max_abs = np.abs(delta).max(axis=(1, 2), initial=0.0)
    first = np.zeros(candidate.shape[0], dtype=np.float64)
    last = np.zeros(candidate.shape[0], dtype=np.float64)
    duration = np.zeros(candidate.shape[0], dtype=np.float64)
    peak = changed.sum(axis=2).max(axis=1, initial=0).astype(np.int64)
    for i in range(candidate.shape[0]):
        locations = np.flatnonzero(any_step[i])
        if locations.size:
            first[i] = (locations[0] + 1) * model_step_seconds / 60.0
            last[i] = (locations[-1] + 1) * model_step_seconds / 60.0
            duration[i] = locations.size * model_step_seconds / 60.0
    transitions = (np.abs(np.diff(delta, axis=1)) > float(epsilon)).sum(axis=(1, 2)).astype(np.int64)
    reversals = ((delta[:, 1:] * delta[:, :-1]) < -(float(epsilon) ** 2)).sum(axis=(1, 2)).astype(np.int64)
    return {
        "changed_actuator_count": count,
        "action_l1_exposure": l1,
        "action_l2_exposure": l2,
        "max_abs_setting_delta": max_abs,
        "first_change_min": first,
        "last_change_min": last,
        "active_duration_min": duration,
        "action_transition_count": transitions,
        "action_reversal_count": reversals,
        "peak_simultaneous_changed_actuators": peak,
    }


def reference_phase_descriptors_v110(
    initial_state: np.ndarray,
    prepared: PreparedStaticV60,
) -> dict[str, float]:
    """Current-state phase descriptors; no future outcome is accessed."""
    state = np.asarray(initial_state, dtype=np.float64)
    if state.ndim != 2 or state.shape[1] != 6:
        raise ValueError("V11 reference phase requires raw [N,6] state")
    max_depth = prepared.max_depth_m.detach().cpu().numpy().astype(np.float64)
    capacity = prepared.storage_capacity_m3.detach().cpu().numpy().astype(np.float64)
    depth_fill = _finite_or_zero(np.maximum(state[:, 0], 0.0), np.maximum(max_depth, 1.0e-6))
    storage_mask = capacity > 1.0e-6
    storage_fill = np.zeros_like(capacity)
    storage_fill[storage_mask] = np.maximum(state[storage_mask, 3], 0.0) / capacity[storage_mask]
    return {
        "reference_depth_fill_mean": float(np.mean(depth_fill)),
        "reference_depth_fill_p90": float(np.quantile(depth_fill, 0.90)),
        "reference_near_surcharge_fraction": float(np.mean(depth_fill >= 0.90)),
        "reference_flood_active_fraction": float(np.mean(state[:, 2] > 1.0e-8)),
        "reference_storage_fill_mean": float(np.mean(storage_fill[storage_mask])) if storage_mask.any() else 0.0,
        "reference_storage_near_capacity_fraction": float(np.mean(storage_fill[storage_mask] >= 0.90)) if storage_mask.any() else 0.0,
        "reference_net_inflow_m3s_mean": float(np.mean(state[:, 4] - state[:, 5])),
    }


def _min_undirected_distance_for_changed(
    relations: ActuatorNodeRelationsV110,
    changed_actuators: np.ndarray,
) -> np.ndarray:
    changed = np.asarray(changed_actuators, dtype=bool).reshape(-1)
    if changed.shape != (relations.actuator_count,):
        raise ValueError("V11 changed-actuator mask does not align with relations")
    if not changed.any():
        return np.full(relations.node_count, np.inf, dtype=np.float64)
    pair = relations.pair_features.detach().cpu().numpy().astype(np.float64)
    proximity = np.maximum(pair[changed, :, 2], pair[changed, :, 3])
    best = proximity.max(axis=0)
    distance = np.full(best.shape, np.inf, dtype=np.float64)
    reachable = best > 0.0
    distance[reachable] = 1.0 / best[reachable] - 1.0
    return distance


def response_mechanism_descriptors_v110(
    *,
    reference_states: np.ndarray,
    candidate_states: np.ndarray,
    reference_flows: np.ndarray,
    candidate_flows: np.ndarray,
    changed_actuators: np.ndarray,
    relations: ActuatorNodeRelationsV110,
    scales: EffectScalesV110,
    horizon: HydraulicHorizonV110 = HydraulicHorizonV110(),
) -> dict[str, float]:
    """Target-side labels used only for TrainFit diagnostics/stratification."""
    indices = np.asarray(horizon.indices(), dtype=np.int64)
    ref_state = np.asarray(reference_states, dtype=np.float64)[indices]
    cand_state = np.asarray(candidate_states, dtype=np.float64)[indices]
    ref_flow = np.asarray(reference_flows, dtype=np.float64)[indices]
    cand_flow = np.asarray(candidate_flows, dtype=np.float64)[indices]
    delta_state = cand_state - ref_state
    delta_state5 = delta_state[..., list(STATE_EFFECT_CHANNELS_V110)]
    delta_flow = cand_flow - ref_flow
    state_scale = np.maximum(np.asarray(scales.state_magnitude_scale, dtype=np.float64), 1.0e-8)
    flow_scale = np.maximum(np.asarray(scales.flow_magnitude_scale, dtype=np.float64), 1.0e-8)
    normalized_state = np.abs(delta_state5) / state_scale[None]
    normalized_flow = np.abs(delta_flow) / flow_scale[None]
    time_energy = normalized_state.mean(axis=(1, 2)) + normalized_flow.mean(axis=1)
    total_energy = float(time_energy.sum())

    threshold_state = np.asarray(scales.state_active_threshold, dtype=np.float64)
    threshold_flow = np.asarray(scales.flow_active_threshold, dtype=np.float64)
    active_time = (np.abs(delta_state5) >= threshold_state[None]).any(axis=(1, 2))
    active_time |= (np.abs(delta_flow) >= threshold_flow[None]).any(axis=1)
    minutes = np.asarray(horizon.response_minutes(), dtype=np.float64)
    onset = float(minutes[np.flatnonzero(active_time)[0]]) if active_time.any() else 0.0
    peak = float(minutes[int(np.argmax(time_energy))]) if time_energy.size else 0.0

    node_energy = normalized_state.sum(axis=(0, 2))
    distance = _min_undirected_distance_for_changed(relations, changed_actuators)
    remote = distance > 8.0
    remote_fraction = float(node_energy[remote].sum() / max(node_energy.sum(), 1.0e-12))
    return {
        "hydraulic_effect_energy": total_energy,
        "onset_min": onset,
        "peak_response_min": peak,
        "remote_effect_fraction_gt8": remote_fraction,
    }


def mechanism_records_v110(
    cache: V60TrainCache,
    names: Sequence[str],
    prepared: PreparedStaticV60,
    relations: ActuatorNodeRelationsV110,
    scales: EffectScalesV110,
    *,
    horizon: HydraulicHorizonV110 = HydraulicHorizonV110(),
) -> list[CandidateMechanismRecordV110]:
    """Extract one auditable record per non-reference branch."""
    records: list[CandidateMechanismRecordV110] = []
    for name in names:
        entry = cache.entry(name)
        arrays, ref = entry.arrays, entry.reference_index
        candidates = [i for i in entry.indices if i != ref]
        reference_settings = np.asarray(arrays["settings"][ref], dtype=np.float64)
        candidate_settings = np.asarray(arrays["settings"][candidates], dtype=np.float64)
        action = action_sequence_descriptors_v110(reference_settings, candidate_settings)
        phase = reference_phase_descriptors_v110(
            np.asarray(arrays["initial_state"][ref], dtype=np.float64), prepared
        )
        ref_tfv = float(np.asarray(arrays["exact_node_flood_volume_m3"][ref], dtype=np.float64).sum())
        for local, candidate_index in enumerate(candidates):
            changed_actuators = (
                np.abs(candidate_settings[local] - reference_settings) > 1.0e-7
            ).any(axis=0)
            response = response_mechanism_descriptors_v110(
                reference_states=np.asarray(arrays["target_states"][ref]),
                candidate_states=np.asarray(arrays["target_states"][candidate_index]),
                reference_flows=np.asarray(arrays["target_actuator_flows"][ref]),
                candidate_flows=np.asarray(arrays["target_actuator_flows"][candidate_index]),
                changed_actuators=changed_actuators,
                relations=relations,
                scales=scales,
                horizon=horizon,
            )
            candidate_tfv = float(
                np.asarray(arrays["exact_node_flood_volume_m3"][candidate_index], dtype=np.float64).sum()
            )
            records.append(
                CandidateMechanismRecordV110(
                    source_kind=entry.source_kind,
                    group_name=name,
                    event_id=entry.event_id,
                    rainfall_group=entry.rainfall_group,
                    checkpoint_id=entry.checkpoint_id,
                    candidate_index=int(candidate_index),
                    delta_tfv_m3=candidate_tfv - ref_tfv,
                    **{key: (int(value[local]) if key in {
                        "changed_actuator_count", "action_transition_count",
                        "action_reversal_count", "peak_simultaneous_changed_actuators"
                    } else float(value[local])) for key, value in action.items()},
                    **phase,
                    **response,
                )
            )
    return records


def _q(values: Sequence[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"p10": float("nan"), "median": float("nan"), "p90": float("nan")}
    return {
        "p10": float(np.quantile(x, 0.10)),
        "median": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
    }


def summarize_mechanism_records_v110(
    records: Sequence[CandidateMechanismRecordV110],
) -> dict[str, Any]:
    if not records:
        raise ValueError("V11 mechanism summary requires records")
    result: dict[str, Any] = {
        "contract": V110_DATA_SEMANTICS_CONTRACT,
        "record_count": len(records),
        "source_counts": {},
        "event_count": len({record.event_id for record in records}),
        "rainfall_group_count": len({record.rainfall_group for record in records}),
        "checkpoint_count": len({record.checkpoint_id for record in records}),
    }
    for source in sorted({record.source_kind for record in records}):
        subset = [record for record in records if record.source_kind == source]
        result["source_counts"][source] = len(subset)
    numeric = (
        "changed_actuator_count", "action_l1_exposure", "max_abs_setting_delta",
        "first_change_min", "active_duration_min", "reference_depth_fill_p90",
        "reference_near_surcharge_fraction", "reference_storage_fill_mean",
        "hydraulic_effect_energy", "onset_min", "peak_response_min",
        "remote_effect_fraction_gt8", "delta_tfv_m3",
    )
    result["quantiles"] = {
        field: _q([float(getattr(record, field)) for record in records]) for field in numeric
    }
    result["training_use"] = {
        "causal_online_features": [
            "action exposure/timing", "current reference hydraulic phase",
            "static actuator-node graph relations",
        ],
        "trainfit_only_stratification_labels": [
            "response onset/peak", "remote-effect fraction", "authoritative delta-TFV magnitude",
        ],
        "forbidden_online_inputs": [
            "future SWMM response", "future realized flooding", "target-side onset/peak labels",
        ],
    }
    return result


__all__ = [
    "CandidateMechanismRecordV110",
    "V110_DATA_SEMANTICS_CONTRACT",
    "action_sequence_descriptors_v110",
    "mechanism_records_v110",
    "reference_phase_descriptors_v110",
    "response_mechanism_descriptors_v110",
    "summarize_mechanism_records_v110",
]
