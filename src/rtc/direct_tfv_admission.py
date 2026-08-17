"""One-sided HOLD-relative admission calibration for Project7 Direct-TFV MPC.

The continuous Step3 optimizer selects an extremal minimum from a large differentiable action
space. Even when the value model is approximately unbiased on cached candidates, this selection can
amplify optimistic residuals (optimizer's curse). The authoritative Development H360 replay exposed
exactly this failure mode: strong negative predictions can be useful, while weak negative extrema can
be false-beneficial.

This module keeps the scientific objective unchanged (system-wide cumulative TFV only). It derives a
one-sided residual margin from a rainfall-disjoint subset of the existing D3 HOLD-reference holdout
and admits an action only when

    predicted_delta_tfv + calibrated_upper_residual_margin < 0.

The remaining D3 HOLD-reference groups are retained as an admission audit. No Validation/Final data,
PFV/peak objective, rule baseline, future realised rainfall or SWMM call is introduced online.
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
    "PROJECT7_DIRECT_TFV_D3_HOLD_ONE_SIDED_ADMISSION_V1"
)
DIRECT_TFV_ADMISSION_COVERAGE = 0.90
DIRECT_TFV_ADMISSION_SPLIT_SEED = 4217


def _stable_key(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{text}".encode("utf-8")).hexdigest()


def split_d3_holdout_for_admission(
    cache: Any,
    names: Sequence[str],
    *,
    seed: int = DIRECT_TFV_ADMISSION_SPLIT_SEED,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Split D3 holdout by rainfall group so calibration and audit never share rainfall forcing."""

    by_group: dict[str, list[str]] = {}
    for name in names:
        entry = cache.entry(name)
        group = str(entry.rainfall_group)
        by_group.setdefault(group, []).append(str(name))
    groups = sorted(by_group, key=lambda value: _stable_key(value, seed))
    if len(groups) < 2:
        raise ValueError("D3 admission calibration requires at least two rainfall groups")
    calibration_group_count = max(1, len(groups) // 2)
    calibration_groups = set(groups[:calibration_group_count])
    audit_groups = set(groups[calibration_group_count:])
    calibration = sorted(
        name for group in calibration_groups for name in by_group[group]
    )
    audit = sorted(name for group in audit_groups for name in by_group[group])
    if not calibration or not audit:
        raise ValueError("D3 admission calibration/audit split must both be non-empty")
    overlap = calibration_groups & audit_groups
    if overlap:
        raise RuntimeError(f"D3 admission rainfall groups overlap: {sorted(overlap)}")
    return calibration, audit, {
        "split_seed": int(seed),
        "calibration_group_count": len(calibration_groups),
        "audit_group_count": len(audit_groups),
        "calibration_rainfall_groups": sorted(calibration_groups),
        "audit_rainfall_groups": sorted(audit_groups),
        "rainfall_group_overlap_count": 0,
    }


def _one_sided_conformal_upper(values: Sequence[float], coverage: float) -> float:
    """Finite-sample one-sided split-conformal upper quantile."""

    if not 0.5 < float(coverage) < 1.0:
        raise ValueError("one-sided admission coverage must lie in (0.5,1)")
    data = np.asarray(tuple(values), dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size <= 0:
        raise ValueError("cannot calibrate Direct-TFV admission from an empty residual set")
    ordered = np.sort(data)
    rank = int(math.ceil((data.size + 1) * float(coverage)))
    rank = min(max(rank, 1), int(data.size))
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
    truth = (
        batch.true_delta_tfv_m3[0]
        .index_select(0, indices)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    change_count = (
        _candidate_change_counts(batch)
        .index_select(0, indices)
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )
    return prediction, truth, change_count


def derive_direct_tfv_admission_calibration(
    model: Any,
    *,
    cache: Any,
    names: Sequence[str],
    normalization: InputNormalizationV60,
    graph: Any,
    device: torch.device,
    action_support: Mapping[str, Any],
    coverage: float = DIRECT_TFV_ADMISSION_COVERAGE,
) -> dict[str, Any]:
    """Calibrate optimism margins from D3 HOLD-reference candidates only."""

    if str(action_support.get("joint_density_reference_semantics", "")) != "HOLD_REFERENCE_ONLY":
        raise ValueError("current Direct-TFV admission requires HOLD-reference joint-density support")
    density_floor = max(
        2,
        int(math.ceil(float(action_support.get("joint_changed_facility_count_q90", 2.0)))),
    )
    static = _graph_tensors(graph, device)
    all_residuals: list[float] = []
    dense_residuals: list[float] = []
    selected_residuals: list[float] = []
    selected_dense_residuals: list[float] = []
    groups = branches = dense_branches = 0
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
        residual = truth - prediction
        finite = np.isfinite(residual)
        all_residuals.extend(residual[finite].tolist())
        dense = finite & (change_count >= density_floor)
        dense_residuals.extend(residual[dense].tolist())
        dense_branches += int(np.sum(dense))
        prediction_with_hold = np.concatenate((np.zeros(1, dtype=np.float64), prediction))
        selected = int(np.argmin(prediction_with_hold))
        if selected > 0:
            index = selected - 1
            value = float(residual[index])
            if math.isfinite(value):
                selected_residuals.append(value)
                if int(change_count[index]) >= density_floor:
                    selected_dense_residuals.append(value)
        groups += 1
        branches += len(prediction)

    if groups <= 0 or branches <= 0:
        raise ValueError("D3 admission calibration has no usable groups")
    global_branch_q = _one_sided_conformal_upper(all_residuals, coverage)
    selected_q = _one_sided_conformal_upper(
        selected_residuals if selected_residuals else all_residuals,
        coverage,
    )
    if dense_residuals:
        dense_branch_q = _one_sided_conformal_upper(dense_residuals, coverage)
    else:
        dense_branch_q = global_branch_q
    if selected_dense_residuals:
        selected_dense_q = _one_sided_conformal_upper(selected_dense_residuals, coverage)
    else:
        selected_dense_q = selected_q

    global_margin = max(0.0, global_branch_q, selected_q)
    dense_margin = max(global_margin, dense_branch_q, selected_dense_q)
    return {
        "contract": DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT,
        "development_only": True,
        "reference_semantics": "D3_HOLD_REFERENCE",
        "coverage": float(coverage),
        "residual_definition": "true_delta_tfv_m3_minus_predicted_delta_tfv_m3",
        "admission_rule": "predicted_delta_tfv_m3 + one_sided_residual_margin_m3 < 0",
        "density_floor_changed_facilities": int(density_floor),
        "global_margin_m3": float(global_margin),
        "dense_margin_m3": float(dense_margin),
        "global_branch_residual_quantile_m3": float(global_branch_q),
        "selected_residual_quantile_m3": float(selected_q),
        "dense_branch_residual_quantile_m3": float(dense_branch_q),
        "selected_dense_residual_quantile_m3": float(selected_dense_q),
        "calibration_groups": int(groups),
        "calibration_branches": int(branches),
        "dense_calibration_branches": int(dense_branches),
        "selected_candidate_residual_count": int(len(selected_residuals)),
        "selected_dense_residual_count": int(len(selected_dense_residuals)),
        "scientific_role": (
            "Correct selection-induced optimistic bias of the continuous TFV optimizer; this is a "
            "one-sided target-residual calibration, not a new objective or rule-baseline imitation"
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
    """Audit the calibrated admission rule on rainfall-disjoint D3 HOLD-reference groups."""

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
        "admission_upper_bound_m3_min": (
            float(np.min(selected_upper)) if selected_upper else 0.0
        ),
        "admission_upper_bound_m3_max": (
            float(np.max(selected_upper)) if selected_upper else 0.0
        ),
    }


__all__ = [
    "DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT",
    "DIRECT_TFV_ADMISSION_COVERAGE",
    "DIRECT_TFV_ADMISSION_SPLIT_SEED",
    "admission_margin_m3",
    "derive_direct_tfv_admission_calibration",
    "evaluate_direct_tfv_admission",
    "split_d3_holdout_for_admission",
]
