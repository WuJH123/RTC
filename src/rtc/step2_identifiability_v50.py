"""Train-only action identifiability and candidate-manifold diagnostics for V5.0.

This module is intentionally read-only with respect to SWMM and validation data.  It
operates on the rebuildable Train cache and on the frozen runtime controller contract.
The outputs are diagnostics, not model-selection thresholds.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .project7_contract import (
    CONTROL_UPDATE_SECONDS,
    MAX_SETTING_DELTA_PER_UPDATE,
    MODEL_STEP_SECONDS,
    PREDICTION_HORIZON_SECONDS,
)
from .step2_counterfactual import counterfactual_groups, reference_index
from .step2_training_cache import load_step2_training_cache


IDENTIFIABILITY_CONTRACT_V50 = "STEP2_ACTION_IDENTIFIABILITY_AUDIT_V50_TRAIN_ONLY"
MPC_MANIFOLD_CONTRACT_V50 = "MPC_CANDIDATE_MANIFOLD_CONTRACT_V1"


def block_action_matrix_v50(action: Any, *, control_blocks: int = 36) -> np.ndarray:
    """Collapse model-step settings into the frozen control-block representation.

    ``action`` may be ``[H, A]`` or ``[C, H, A]``.  V5 never invents a new action
    grid: the operation only exposes the existing 72 model steps as 36 two-step
    blocks under the frozen 5-minute/10-minute timing contract.
    """

    is_torch = hasattr(action, "detach")
    values = np.asarray(action.detach().cpu() if is_torch else action, dtype=np.float64)
    if values.ndim not in (2, 3):
        raise ValueError("action must have shape [H,A] or [C,H,A]")
    horizon = values.shape[-2]
    if horizon % int(control_blocks) != 0:
        raise ValueError("model horizon must divide into control blocks")
    steps = horizon // int(control_blocks)
    shape = values.shape[:-2] + (int(control_blocks), steps, values.shape[-1])
    result = values.reshape(shape).mean(axis=-2)
    if is_torch:
        import torch

        return torch.as_tensor(result, dtype=action.dtype, device=action.device)
    return result


def effective_rank_v50(matrix: Any) -> dict[str, Any]:
    """Return numerical and energy effective ranks of a sample-by-feature matrix."""

    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if values.shape[0] == 0 or values.shape[1] == 0:
        return {
            "samples": int(values.shape[0]),
            "features": int(values.shape[1]),
            "rank": 0,
            "rank_90": 0,
            "rank_95": 0,
            "rank_99": 0,
            "singular_values": [],
        }
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    tolerance = np.finfo(np.float64).eps * max(values.shape) * max(float(singular[0]), 1.0)
    numerical_rank = int(np.count_nonzero(singular > tolerance))
    energy = np.square(singular)
    total = float(energy.sum())
    if total <= np.finfo(np.float64).tiny:
        ranks = {"rank_90": 0, "rank_95": 0, "rank_99": 0}
    else:
        cumulative = np.cumsum(energy) / total
        ranks = {
            f"rank_{int(q * 100)}": int(np.searchsorted(cumulative, q) + 1)
            for q in (0.90, 0.95, 0.99)
        }
    return {
        "samples": int(values.shape[0]),
        "features": int(values.shape[1]),
        "rank": numerical_rank,
        **ranks,
        "singular_values": [float(value) for value in singular[: min(32, singular.size)]],
    }


def mpc_candidate_manifold_contract_v50() -> dict[str, Any]:
    """Return the production-intended continuous candidate support, read-only."""

    horizon_steps = int(PREDICTION_HORIZON_SECONDS // MODEL_STEP_SECONDS)
    block_steps = int(CONTROL_UPDATE_SECONDS // MODEL_STEP_SECONDS)
    if horizon_steps % block_steps:
        raise ValueError("frozen runtime horizon is not divisible by update cadence")
    return {
        "contract": MPC_MANIFOLD_CONTRACT_V50,
        "candidate_family": "continuous_all_actuator_projected_gradient",
        "actuator_count": 109,
        "model_horizon_steps": horizon_steps,
        "control_block_steps": block_steps,
        "control_blocks": int(horizon_steps // block_steps),
        "setting_bounds": [0.0, 1.0],
        "max_setting_delta_per_update": float(MAX_SETTING_DELTA_PER_UPDATE),
        "candidate_count_per_decision": "continuous_optimization_not_finite_pack",
        "identity_policy": "all_109_actuators_retained",
        "temporal_policy": "piecewise_constant_control_blocks_with_sequential_projection",
        "validation_or_final_used": False,
    }


def _percentiles(values: list[float] | np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {name: float("nan") for name in ("min", "p25", "median", "p75", "p90", "p95", "max")}
    q = np.percentile(array, [0, 25, 50, 75, 90, 95, 100])
    return {name: float(value) for name, value in zip(("min", "p25", "median", "p75", "p90", "p95", "max"), q, strict=True)}


def _nearest_neighbor_distance(values: np.ndarray) -> dict[str, float]:
    if values.shape[0] < 2:
        return {"median": float("nan"), "p90": float("nan"), "max": float("nan")}
    centered = np.asarray(values, dtype=np.float64)
    norms = np.square(centered).sum(axis=1, keepdims=True)
    distances = norms + norms.T - 2.0 * centered @ centered.T
    np.fill_diagonal(distances, np.inf)
    nearest = np.sqrt(np.maximum(np.min(distances, axis=1), 0.0) / max(values.shape[1], 1))
    return {"median": float(np.median(nearest)), "p90": float(np.percentile(nearest, 90)), "max": float(np.max(nearest))}


def _actuator_type(physics: np.ndarray, names: tuple[str, ...], index: int) -> str:
    fields = {name: float(physics[index, pos]) for pos, name in enumerate(names)}
    for name, label in (("is_pump", "pump"), ("is_orifice", "orifice"), ("is_weir", "weir"), ("is_outlet", "outlet")):
        if fields.get(name, 0.0) > 0.5:
            return label
    return "other"


def _source_summary(actions: np.ndarray, reference_settings: np.ndarray) -> dict[str, Any]:
    if actions.size == 0:
        return {"candidate_count": 0}
    delta = np.asarray(actions, dtype=np.float64)
    active = np.abs(delta) > 1e-6
    changed_per_candidate = active.any(axis=1).sum(axis=1)
    duration = active.any(axis=2).sum(axis=1)
    return {
        "candidate_count": int(delta.shape[0]),
        "active_actuator_count": _percentiles(changed_per_candidate),
        "action_l1": _percentiles(np.abs(delta).sum(axis=(1, 2))),
        "action_l2": _percentiles(np.sqrt(np.square(delta).sum(axis=(1, 2)))),
        "action_linf": _percentiles(np.abs(delta).max(axis=(1, 2))),
        "action_duration_blocks": _percentiles(duration),
        "changed_control_blocks": _percentiles(active.any(axis=2).sum(axis=1)),
        "all_zero_fraction": float(np.mean(~active.any(axis=(1, 2)))),
        "reference_setting_range": _percentiles(reference_settings.reshape(-1)),
    }


def scan_train_action_identifiability_v50(
    cache_manifest: str | Path,
    *,
    actuator_physics: np.ndarray | None = None,
    actuator_physics_feature_names: tuple[str, ...] | None = None,
    control_blocks: int = 36,
) -> dict[str, Any]:
    """Scan all D2/D3 Train-cache groups without loading target trajectories."""

    cache = load_step2_training_cache(Path(cache_manifest))
    source_actions: dict[str, list[np.ndarray]] = defaultdict(list)
    source_refs: dict[str, list[np.ndarray]] = defaultdict(list)
    source_states: dict[str, list[np.ndarray]] = defaultdict(list)
    group_rank: dict[str, dict[str, Any]] = {}
    storm_groups: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    coaction: dict[str, np.ndarray] = {}
    temporal_samples: dict[str, list[np.ndarray]] = defaultdict(list)
    change_points: dict[str, list[int]] = defaultdict(list)

    for item in cache["shards"]:
        arrays = item["arrays"]
        for group, indices in counterfactual_groups(arrays).items():
            source = group.split("::", 1)[0].upper()
            if source not in {"D2", "D3"}:
                continue
            ref = reference_index(arrays, indices)
            reference = np.asarray(arrays["settings"][ref], dtype=np.float64)
            candidates = [index for index in indices if index != ref]
            if not candidates:
                continue
            model_actions = np.stack([arrays["settings"][index] - reference for index in candidates], axis=0)
            block_actions = block_action_matrix_v50(model_actions, control_blocks=control_blocks)
            flat = block_actions.reshape(block_actions.shape[0], -1)
            rank = effective_rank_v50(flat)
            group_rank[group] = {"source_kind": source, "candidate_count": len(candidates), **rank}
            storm_groups[str(group.split("::")[1])] += 1
            candidate_counts[source] += len(candidates)
            source_actions[source].append(block_actions)
            reference_blocks = block_action_matrix_v50(reference, control_blocks=control_blocks)
            source_refs[source].append(np.repeat(reference_blocks[None], len(candidates), axis=0))
            source_states[source].append(np.asarray(arrays["initial_state"][ref], dtype=np.float64).reshape(-1))
            if source not in coaction:
                actuator_count = model_actions.shape[-1]
                coaction[source] = np.zeros((actuator_count, actuator_count), dtype=np.int64)
            changed = np.abs(block_actions).any(axis=1)  # [C,A]
            coaction[source] += changed.T.astype(np.int64) @ changed.astype(np.int64)
            for candidate_delta in block_actions:
                active = np.abs(candidate_delta) > 1e-6
                changed_blocks = np.flatnonzero(active.any(axis=1))
                change_points[source].extend(np.diff(changed_blocks).tolist())
                for actuator in np.flatnonzero(active.any(axis=0)):
                    temporal_samples[source].append(candidate_delta[:, actuator])
                    if actuator_physics is not None and actuator_physics_feature_names is not None:
                        type_counts[source][_actuator_type(actuator_physics, actuator_physics_feature_names, int(actuator))] += 1

    source_report: dict[str, Any] = {}
    for source in ("D2", "D3"):
        actions = np.concatenate(source_actions[source], axis=0) if source_actions[source] else np.empty((0, control_blocks, 0))
        refs = np.concatenate(source_refs[source], axis=0) if source_refs[source] else np.empty((0, control_blocks, 0))
        flat = actions.reshape(actions.shape[0], -1) if actions.size else np.empty((0, 0))
        global_rank = effective_rank_v50(flat)
        group_values = list(group_rank.values())
        rank_95 = [int(value["rank_95"]) for value in group_values if value["source_kind"] == source]
        rank_99 = [int(value["rank_99"]) for value in group_values if value["source_kind"] == source]
        co = coaction.get(source, np.zeros((0, 0), dtype=np.int64))
        temporal = np.stack(temporal_samples[source], axis=0) if temporal_samples[source] else np.empty((0, control_blocks))
        temporal_rank = effective_rank_v50(temporal)
        source_report[source] = {
            "groups": int(sum(1 for value in group_values if value["source_kind"] == source)),
            "candidates": int(candidate_counts[source]),
            "group_candidate_counts": sorted({int(value["candidate_count"]) for value in group_values if value["source_kind"] == source}),
            "group_effective_rank_95": _percentiles(rank_95),
            "group_effective_rank_99": _percentiles(rank_99),
            "global_action_rank": global_rank,
            "nearest_neighbor_distance": _nearest_neighbor_distance(flat) if flat.size else {},
            "action_distribution": _source_summary(actions, refs),
            "temporal_effective_rank": temporal_rank,
            "change_point_gap_blocks": _percentiles(change_points[source]),
            "actuator_type_frequency": dict(type_counts[source]),
            "coaction": {
                "pairs_never_jointly_changed": int(np.triu(co == 0, k=1).sum()) if co.size else 0,
                "pairs_seen_once": int(np.triu(co == 1, k=1).sum()) if co.size else 0,
                "pairs_seen_at_least_5": int(np.triu(co >= 5, k=1).sum()) if co.size else 0,
                "pairs_seen_at_least_10": int(np.triu(co >= 10, k=1).sum()) if co.size else 0,
                "matrix_shape": list(co.shape),
            },
            "state_action_disentanglement": {
                "state_nearest_neighbor": _nearest_neighbor_distance(np.stack(source_states[source])) if len(source_states[source]) > 1 else {},
                "action_nearest_neighbor": _nearest_neighbor_distance(flat) if flat.size else {},
                "interpretation": "reference initial state and candidate action distances are diagnostics; no outcome-based selection",
            },
        }

    manifold = mpc_candidate_manifold_contract_v50()
    d3_actions = np.concatenate(source_actions.get("D3", []), axis=0) if source_actions.get("D3") else np.empty((0, control_blocks, 109))
    d3_refs = np.concatenate(source_refs.get("D3", []), axis=0) if source_refs.get("D3") else np.empty((0, control_blocks, 109))
    if d3_actions.size:
        # The production manifold is continuous, so support is checked against
        # bounds and the sequential per-block delta contract rather than a finite
        # candidate list.
        candidate = d3_refs + d3_actions
        previous = np.concatenate((d3_refs[:, :1], candidate[:, :-1]), axis=1)
        in_bounds = (candidate >= -1e-6).all(axis=(1, 2)) & (candidate <= 1.0 + 1e-6).all(axis=(1, 2))
        in_rate = (np.abs(candidate - previous) <= float(MAX_SETTING_DELTA_PER_UPDATE) + 1e-6).all(axis=(1, 2))
        out_support = ~(in_bounds & in_rate)
        changed_fraction = (np.abs(d3_actions).any(axis=1).mean(axis=0)).tolist()
        manifold_report = {
            **manifold,
            "observed_candidates": int(d3_actions.shape[0]),
            "in_support_fraction": float(np.mean(~out_support)),
            "out_of_support_fraction": float(np.mean(out_support)),
            "nearest_neighbor_distance": _nearest_neighbor_distance(d3_actions.reshape(d3_actions.shape[0], -1)),
            "per_actuator_changed_fraction": [float(value) for value in changed_fraction],
            "temporal_pattern_coverage": {
                "observed_control_block_patterns": int(d3_actions.shape[1]),
                "finite_candidate_cap": False,
                "coverage_definition": "continuous support and observed Train action density proxy",
            },
        }
    else:
        manifold_report = {**manifold, "observed_candidates": 0, "in_support_fraction": 0.0, "out_of_support_fraction": 1.0}

    d3 = source_report.get("D3", {})
    local95 = float(d3.get("group_effective_rank_95", {}).get("median", 0.0))
    global95 = float(d3.get("global_action_rank", {}).get("rank_95", 0.0))
    feature_dim = float(d3.get("global_action_rank", {}).get("features", control_blocks * 109))
    never_pairs = int(d3.get("coaction", {}).get("pairs_never_jointly_changed", 0))
    total_pairs = 109 * 108 // 2
    if not d3.get("candidates"):
        verdict = "INADEQUATE"
    elif local95 < 4.0 or global95 / max(feature_dim, 1.0) < 0.05:
        verdict = "INADEQUATE"
    elif local95 < 8.0 or never_pairs / max(total_pairs, 1) > 0.50:
        verdict = "MARGINAL"
    else:
        verdict = "ADEQUATE"

    return {
        "contract": IDENTIFIABILITY_CONTRACT_V50,
        "boundary": {"swmm_launched": False, "validation_accessed": False, "final_accessed": False},
        "cache_manifest": str(Path(cache_manifest).resolve()),
        "cache_source_manifest_sha256": str(cache.get("source_manifest_sha256", "")),
        "d3_group_count": int(source_report.get("D3", {}).get("groups", 0)),
        "d3_candidate_count": int(source_report.get("D3", {}).get("candidates", 0)),
        "groups_per_storm": dict(storm_groups),
        "source": source_report,
        "group_rank": group_rank,
        "mpc_candidate_manifold": manifold_report,
        "current_d3_identifiability": verdict,
        "decision_rule": "deterministic local/global action-rank and support census; no outcome-based threshold tuning",
        "d3_v2_authorized": verdict == "INADEQUATE",
    }


__all__ = [
    "IDENTIFIABILITY_CONTRACT_V50",
    "MPC_MANIFOLD_CONTRACT_V50",
    "block_action_matrix_v50",
    "effective_rank_v50",
    "mpc_candidate_manifold_contract_v50",
    "scan_train_action_identifiability_v50",
]
