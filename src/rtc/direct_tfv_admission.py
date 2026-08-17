"""Optimizer-aware one-sided admission calibration for Project7 Direct-TFV MPC.

The current scientific failure is continuous-optimizer selection-induced optimism: cached D3
candidates can look safe while L-BFGS-B selects extremal negative predictions that are false-
beneficial in exact same-prefix H360 SWMM replay.

The admission margin combines two Development-only evidence sources without overstating their
statistical strength:

1. rainfall-disjoint D3 HOLD-reference residuals use a finite-sample one-sided split-conformal upper
   bound whose independent calibration unit is the rainfall group, not individual correlated
   candidate branches. Each calibration rainfall group contributes its worst true-minus-predicted
   residual;
2. the small exact optimizer-replay sample uses the maximum observed optimism residual. With only a
   few optimizer plans it would be invalid to claim formal conformal coverage.

The final margin is the more conservative component. The online objective remains system-wide
cumulative TFV only. No PFV/peak/energy objective, baseline imitation, future realised rainfall or
online SWMM call is introduced. Optimizer replay used for calibration cannot also be claimed as
independent post-calibration validation evidence.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .step2_tfv_value_training import (
    _branch_indices,
    _candidate_change_counts,
    _forward_candidates,
    _graph_tensors,
)
from .step2_train_response_v60 import InputNormalizationV60


DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT = (
    "PROJECT7_DIRECT_TFV_OPTIMIZER_AWARE_ONE_SIDED_ADMISSION_V1"
)
DIRECT_TFV_ADMISSION_COVERAGE = 0.90
DIRECT_TFV_ADMISSION_SPLIT_SEED = 4217
DIRECT_TFV_REPLAY_CONTRACT = "PROJECT7_DIRECT_TFV_SAME_PREFIX_H360_AUTHORITATIVE_SWMM_REPLAY_V1"


def _stable_key(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{text}".encode("utf-8")).hexdigest()


def _minimum_conformal_sample_size(coverage: float) -> int:
    if not 0.5 < float(coverage) < 1.0:
        raise ValueError("one-sided admission coverage must lie in (0.5,1)")
    ratio = float(coverage) / (1.0 - float(coverage))
    return int(math.ceil(ratio - 1.0e-12))


def split_d3_holdout_for_admission(
    cache: Any,
    names: Sequence[str],
    *,
    coverage: float = DIRECT_TFV_ADMISSION_COVERAGE,
    seed: int = DIRECT_TFV_ADMISSION_SPLIT_SEED,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Create rainfall-disjoint D3 calibration/audit sets with valid finite-sample coverage."""

    by_group: dict[str, list[str]] = {}
    for name in names:
        entry = cache.entry(name)
        group = str(entry.rainfall_group)
        by_group.setdefault(group, []).append(str(name))
    groups = sorted(by_group, key=lambda value: _stable_key(value, seed))
    minimum = _minimum_conformal_sample_size(coverage)
    if len(groups) < minimum + 1:
        raise ValueError(
            f"{coverage:.3f} rainfall-group conformal calibration plus an independent D3 audit "
            f"needs at least {minimum + 1} rainfall groups; got {len(groups)}"
        )
    calibration_group_count = max(minimum, len(groups) // 2)
    calibration_group_count = min(calibration_group_count, len(groups) - 1)
    calibration_groups = set(groups[:calibration_group_count])
    audit_groups = set(groups[calibration_group_count:])
    calibration = sorted(name for group in calibration_groups for name in by_group[group])
    audit = sorted(name for group in audit_groups for name in by_group[group])
    if not calibration or not audit:
        raise ValueError("D3 admission calibration/audit split must both be non-empty")
    overlap = calibration_groups & audit_groups
    if overlap:
        raise RuntimeError(f"D3 admission rainfall groups overlap: {sorted(overlap)}")
    return calibration, audit, {
        "split_seed": int(seed),
        "requested_conformal_coverage": float(coverage),
        "minimum_calibration_rainfall_groups": int(minimum),
        "calibration_group_count": len(calibration_groups),
        "audit_group_count": len(audit_groups),
        "calibration_rainfall_groups": sorted(calibration_groups),
        "audit_rainfall_groups": sorted(audit_groups),
        "rainfall_group_overlap_count": 0,
    }


def _one_sided_conformal_upper(values: Sequence[float], coverage: float) -> float:
    """Finite-sample one-sided split-conformal upper quantile without invalid rank clipping."""

    data = np.asarray(tuple(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size <= 0:
        raise ValueError("cannot calibrate Direct-TFV admission from an empty residual set")
    minimum = _minimum_conformal_sample_size(coverage)
    if int(data.size) < minimum:
        raise ValueError(
            f"{coverage:.3f} one-sided conformal coverage needs at least {minimum} independent "
            f"residual units; got {int(data.size)}"
        )
    ordered = np.sort(data)
    rank = int(math.ceil((data.size + 1) * float(coverage) - 1.0e-12))
    if rank < 1 or rank > int(data.size):
        raise ValueError("requested conformal coverage has no finite order statistic")
    return float(ordered[rank - 1])


def _score_group(
    model: Any,
    *,
    cache: Any,
    name: str,
    normalization: InputNormalizationV60,
    graph_tensors: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = cache.batch(name, normalization, device)
    indices = _branch_indices(batch, mode="all")
    if int(indices.numel()) <= 0:
        raise ValueError(f"{name}: D3 admission group has no changed candidate")
    with torch.no_grad():
        output = _forward_candidates(model, batch, indices, graph_tensors=graph_tensors)
    prediction = output.total_delta_tfv_m3.detach().cpu().numpy().astype(np.float64)
    truth = batch.true_delta_tfv_m3[0].index_select(0, indices).detach().cpu().numpy().astype(np.float64)
    change_count = (
        _candidate_change_counts(batch)
        .index_select(0, indices)
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    return prediction, truth, change_count


def _optimizer_replay_residuals(
    report: Mapping[str, Any],
    *,
    density_floor: int,
) -> tuple[list[float], list[float], dict[str, Any]]:
    if str(report.get("contract", "")) != DIRECT_TFV_REPLAY_CONTRACT:
        raise ValueError("Direct-TFV admission requires the exact same-prefix H360 replay contract")
    if report.get("development_only") is not True or report.get("swmm_launched") is not True:
        raise ValueError("optimizer replay calibration must be authoritative Development SWMM evidence")
    if int(report.get("prefix_mismatch_count", -1)) != 0:
        raise ValueError("optimizer replay calibration contains prefix mismatches")
    rows = report.get("results")
    if not isinstance(rows, list) or len(rows) < 4:
        raise ValueError("optimizer replay calibration requires at least four exact H360 plans")

    residuals: list[float] = []
    dense: list[float] = []
    event_ids: set[str] = set()
    plan_hashes: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("optimizer replay result is not an object")
        for key in (
            "predicted_delta_tfv_m3",
            "true_delta_tfv_h360_m3",
            "active_facility_count",
            "plan_sha256",
        ):
            if key not in row:
                raise ValueError(f"optimizer replay result lacks {key}")
        for key in (
            "prefix_state_max_abs_difference",
            "prefix_target_max_abs_difference",
            "prefix_current_max_abs_difference",
            "prefix_statistics_max_abs_difference",
            "hold_reference_target_max_abs_difference",
        ):
            if float(row.get(key, 0.0)) != 0.0:
                raise ValueError(f"optimizer replay calibration has nonzero {key}")
        if abs(float(row.get("candidate_branch_routing_error_pct", 0.0))) > 1.0e-9:
            raise ValueError("optimizer replay candidate branch has routing error")
        if abs(float(row.get("hold_branch_routing_error_pct", 0.0))) > 1.0e-9:
            raise ValueError("optimizer replay HOLD branch has routing error")

        predicted = float(row["predicted_delta_tfv_m3"])
        truth = float(row["true_delta_tfv_h360_m3"])
        residual = truth - predicted
        if not math.isfinite(residual):
            raise ValueError("optimizer replay residual is non-finite")
        residuals.append(residual)
        if int(row["active_facility_count"]) >= int(density_floor):
            dense.append(residual)
        event_ids.add(str(row.get("event_id", "UNKNOWN")))
        plan_hash = str(row["plan_sha256"])
        if plan_hash in plan_hashes:
            raise ValueError("optimizer replay calibration repeats a plan SHA")
        plan_hashes.add(plan_hash)
    return residuals, dense, {
        "optimizer_replay_count": len(residuals),
        "optimizer_replay_dense_count": len(dense),
        "optimizer_replay_event_ids": sorted(event_ids),
        "optimizer_replay_plan_sha256": sorted(plan_hashes),
        "optimizer_replay_prefix_mismatch_count": 0,
    }


def derive_direct_tfv_admission_calibration(
    model: Any,
    *,
    cache: Any,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    action_support: Mapping[str, Any],
    optimizer_replay_report: Mapping[str, Any],
    coverage: float = DIRECT_TFV_ADMISSION_COVERAGE,
) -> dict[str, Any]:
    """Combine rainfall-group conformal residual maxima with empirical optimizer maxima."""

    if str(action_support.get("joint_density_reference_semantics", "")) != "HOLD_REFERENCE_ONLY":
        raise ValueError("current Direct-TFV admission requires HOLD-reference joint-density support")
    density_floor = max(
        2,
        int(math.ceil(float(action_support.get("joint_changed_facility_count_q90", 2.0)))),
    )
    static = _graph_tensors(graph, device)
    group_global_max: dict[str, float] = {}
    group_dense_max: dict[str, float] = {}
    branches = dense_branches = 0
    model.eval()
    for name in names:
        prediction, truth, change_count = _score_group(
            model,
            cache=cache,
            name=name,
            normalization=normalization,
            graph_tensors=static,
            device=device,
        )
        rainfall_group = str(cache.entry(name).rainfall_group)
        residual = truth - prediction
        finite = np.isfinite(residual)
        if bool(np.any(finite)):
            worst = float(np.max(residual[finite]))
            group_global_max[rainfall_group] = max(
                worst,
                group_global_max.get(rainfall_group, -math.inf),
            )
        dense = finite & (change_count >= density_floor)
        if bool(np.any(dense)):
            worst_dense = float(np.max(residual[dense]))
            group_dense_max[rainfall_group] = max(
                worst_dense,
                group_dense_max.get(rainfall_group, -math.inf),
            )
            dense_branches += int(np.sum(dense))
        branches += len(prediction)
    if not group_global_max or branches <= 0:
        raise ValueError("D3 admission calibration has no usable rainfall-group residual maxima")

    replay_residuals, replay_dense, replay_meta = _optimizer_replay_residuals(
        optimizer_replay_report,
        density_floor=density_floor,
    )
    group_values = list(group_global_max.values())
    d3_global_q = _one_sided_conformal_upper(group_values, coverage)
    minimum = _minimum_conformal_sample_size(coverage)
    dense_group_values = list(group_dense_max.values())
    d3_dense_q = (
        _one_sided_conformal_upper(dense_group_values, coverage)
        if len(dense_group_values) >= minimum
        else d3_global_q
    )

    replay_global_max = max(replay_residuals)
    replay_dense_max = max(replay_dense) if replay_dense else replay_global_max
    global_margin = max(0.0, d3_global_q, replay_global_max)
    dense_margin = max(global_margin, d3_dense_q, replay_dense_max)
    return {
        "contract": DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT,
        "development_only": True,
        "reference_semantics": "HOLD_ACTIVE_TARGET_H360",
        "d3_conformal_coverage": float(coverage),
        "d3_conformal_unit": "RAINFALL_GROUP_MAX_TRUE_MINUS_PREDICTED_RESIDUAL",
        "residual_definition": "true_delta_tfv_m3_minus_predicted_delta_tfv_m3",
        "admission_rule": "predicted_delta_tfv_m3 + one_sided_residual_margin_m3 < 0",
        "density_floor_changed_facilities": int(density_floor),
        "global_margin_m3": float(global_margin),
        "dense_margin_m3": float(dense_margin),
        "d3_rainfall_group_residual_conformal_upper_m3": float(d3_global_q),
        "d3_dense_rainfall_group_residual_conformal_upper_m3": float(d3_dense_q),
        "d3_dense_conformal_used": bool(len(dense_group_values) >= minimum),
        "d3_calibration_rainfall_group_count": int(len(group_values)),
        "d3_dense_calibration_rainfall_group_count": int(len(dense_group_values)),
        "d3_calibration_branches": int(branches),
        "d3_dense_calibration_branches": int(dense_branches),
        "optimizer_replay_rule": "MAX_OBSERVED_TRUE_MINUS_PREDICTED_RESIDUAL",
        "optimizer_replay_residual_max_m3": float(replay_global_max),
        "optimizer_replay_dense_residual_max_m3": float(replay_dense_max),
        "optimizer_replay_coverage_claimed": False,
        **replay_meta,
        "scientific_role": (
            "Rainfall-group maxima provide the finite-sample D3 model-error bound without branch "
            "pseudo-replication; exact optimizer replay contributes a conservative observed "
            "selection-shift maximum without a formal coverage claim"
        ),
        "independence_note": (
            "events listed in optimizer_replay_event_ids are calibration evidence and must not be "
            "counted as independent post-calibration validation events"
        ),
    }


def admission_margin_m3(calibration: Mapping[str, Any], active_facility_count: int) -> tuple[float, str]:
    if str(calibration.get("contract")) != DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT:
        raise ValueError("Direct-TFV Step3 received the wrong admission calibration contract")
    density_floor = int(calibration.get("density_floor_changed_facilities", 2))
    if int(active_facility_count) >= density_floor:
        return float(calibration["dense_margin_m3"]), "dense"
    return float(calibration["global_margin_m3"]), "global"


def evaluate_direct_tfv_admission(
    model: Any,
    *,
    cache: Any,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit the final margin on rainfall-disjoint D3 HOLD-reference cached groups."""

    static = _graph_tensors(graph, device)
    groups = acted = harmful = beneficial = 0
    selected_truth: list[float] = []
    selected_upper: list[float] = []
    model.eval()
    for name in names:
        prediction, truth, change_count = _score_group(
            model,
            cache=cache,
            name=name,
            normalization=normalization,
            graph_tensors=static,
            device=device,
        )
        selected = int(np.argmin(np.concatenate((np.zeros(1), prediction))))
        value = 0.0
        if selected > 0:
            index = selected - 1
            margin, _ = admission_margin_m3(calibration, int(change_count[index]))
            upper = float(prediction[index] + margin)
            selected_upper.append(upper)
            if upper < 0.0:
                acted += 1
                value = float(truth[index])
        selected_truth.append(value)
        beneficial += int(value < -1.0)
        harmful += int(value > 1.0)
        groups += 1
    if groups <= 0:
        raise ValueError("D3 admission audit has no usable groups")
    return {
        "groups": int(groups),
        "acted_fraction": float(acted / groups),
        "selected_beneficial_fraction": float(beneficial / groups),
        "selected_harmful_fraction": float(harmful / groups),
        "selected_true_delta_tfv_m3": float(np.mean(selected_truth)),
        "admission_upper_bound_m3_min": float(np.min(selected_upper)) if selected_upper else 0.0,
        "admission_upper_bound_m3_max": float(np.max(selected_upper)) if selected_upper else 0.0,
    }


__all__ = [
    "DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT",
    "DIRECT_TFV_ADMISSION_COVERAGE",
    "DIRECT_TFV_ADMISSION_SPLIT_SEED",
    "DIRECT_TFV_REPLAY_CONTRACT",
    "_minimum_conformal_sample_size",
    "_one_sided_conformal_upper",
    "admission_margin_m3",
    "derive_direct_tfv_admission_calibration",
    "evaluate_direct_tfv_admission",
    "split_d3_holdout_for_admission",
]
