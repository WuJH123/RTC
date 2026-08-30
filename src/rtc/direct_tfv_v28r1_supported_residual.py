"""Supported-manifold residual fitting for Project7 V28R1 Development.

V28R1 keeps the frozen V27/Q27 surface and the V28 q95 execution path. It changes only the
residual learner: features that require an unavailable pre-q95 proposal are excluded from the
statistical correction, and residual strength is selected on Train leakage-group CV only.
Validation and Test never select ridge or shrinkage.
"""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np

from .direct_tfv_v28_residual_value import (
    V28_RESIDUAL_FEATURE_NAMES,
    V28ResidualValueModel,
    evaluate_v28_residual,
)

V28R1_SUPPORTED_RESIDUAL_CONTRACT = (
    "PROJECT7_STEP3_V28R1_SUPPORTED_MANIFOLD_RESIDUAL_PARITY_V1"
)
V28R1_RIDGE_GRID = (0.0, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0)
V28R1_SHRINKAGE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

# These columns are computable identically from an already-supported candidate at training and
# deployment. The other V28 columns encode raw->supported contraction geometry. Historical
# q95-exact rows do not carry the raw proposal that created them, so using those columns would
# create a train/deployment feature-distribution mismatch.
V28R1_SUPPORTED_FEATURE_NAMES = (
    "q27_supported_score_m3",
    "supported_first_move_l1",
    "changed_facility_count",
    "network_stress_q75",
    "rain_level",
    "strong_storm_blend",
    "candidate_family_probe_050",
    "candidate_family_probe_100",
    "candidate_family_hydraulic",
    "candidate_family_auto_rbc_shadow",
)
V28R1_SUPPORTED_FEATURE_INDICES = tuple(
    V28_RESIDUAL_FEATURE_NAMES.index(name) for name in V28R1_SUPPORTED_FEATURE_NAMES
)
V28R1_ZERO_WEIGHT_FEATURE_NAMES = tuple(
    name
    for index, name in enumerate(V28_RESIDUAL_FEATURE_NAMES)
    if index not in V28R1_SUPPORTED_FEATURE_INDICES
)


def _finite_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[0] == 0
        or array.shape[1] != len(V28_RESIDUAL_FEATURE_NAMES)
    ):
        raise ValueError(f"{name} must be [N,{len(V28_RESIDUAL_FEATURE_NAMES)}]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _finite_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def _fit_safe_ridge(
    features: np.ndarray,
    target: np.ndarray,
    ridge: float,
) -> V28ResidualValueModel:
    x_full = _finite_matrix(features, name="features")
    y = _finite_vector(target, name="target")
    if len(y) != x_full.shape[0]:
        raise ValueError("V28R1 residual fit arrays are not aligned")
    if float(ridge) < 0.0 or not np.isfinite(float(ridge)):
        raise ValueError("ridge must be finite and non-negative")

    x = x_full[:, V28R1_SUPPORTED_FEATURE_INDICES]
    mean_safe = x.mean(axis=0)
    scale_safe = np.maximum(x.std(axis=0), 1.0e-6)
    z = (x - mean_safe) / scale_safe
    design = np.column_stack((z, np.ones(len(z), dtype=np.float64)))
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ y
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]

    full_mean = np.zeros(len(V28_RESIDUAL_FEATURE_NAMES), dtype=np.float64)
    full_scale = np.ones(len(V28_RESIDUAL_FEATURE_NAMES), dtype=np.float64)
    full_weight = np.zeros(len(V28_RESIDUAL_FEATURE_NAMES), dtype=np.float64)
    for local, global_index in enumerate(V28R1_SUPPORTED_FEATURE_INDICES):
        full_mean[global_index] = mean_safe[local]
        full_scale[global_index] = scale_safe[local]
        full_weight[global_index] = beta[local]

    return V28ResidualValueModel(
        feature_mean=full_mean,
        feature_scale=full_scale,
        weight=full_weight,
        intercept=float(beta[-1]),
        q27_checkpoint_sha256="",
        ridge=float(ridge),
    )


def _shrink_model(
    model: V28ResidualValueModel,
    *,
    alpha: float,
    q27_checkpoint_sha256: str,
) -> V28ResidualValueModel:
    value = float(alpha)
    if not 0.0 <= value <= 1.0 or not np.isfinite(value):
        raise ValueError("residual shrinkage must lie in [0,1]")
    return V28ResidualValueModel(
        feature_mean=model.feature_mean.copy(),
        feature_scale=model.feature_scale.copy(),
        weight=model.weight * value,
        intercept=float(model.intercept) * value,
        q27_checkpoint_sha256=str(q27_checkpoint_sha256).lower(),
        ridge=float(model.ridge),
    )


def _group_folds(groups: Sequence[str], *, folds: int) -> tuple[int, dict[str, int]]:
    unique = sorted({str(value) for value in groups})
    if len(unique) < 2:
        raise ValueError("V28R1 requires at least two Train leakage groups")
    count = max(2, min(int(folds), len(unique)))
    ordered = sorted(
        unique,
        key=lambda value: hashlib.sha256(
            f"v28r1|{value}".encode("utf-8")
        ).hexdigest(),
    )
    return count, {group: index % count for index, group in enumerate(ordered)}


def fit_v28r1_supported_residual(
    *,
    train_features: np.ndarray,
    train_q27_scores_m3: np.ndarray,
    train_truth_m3: np.ndarray,
    train_groups: Sequence[str],
    train_units: Sequence[str],
    q27_checkpoint_sha256: str,
    cv_folds: int = 5,
    ridge_grid: Sequence[float] = V28R1_RIDGE_GRID,
    shrinkage_grid: Sequence[float] = V28R1_SHRINKAGE_GRID,
) -> tuple[V28ResidualValueModel, dict[str, Any]]:
    """Select ridge and shrinkage using Train leakage-group CV only.

    ``alpha=0`` is a real candidate and exactly recovers frozen Q27. Validation and Test are not
    accepted by this function, which makes it impossible for either split to tune the correction.
    """

    x = _finite_matrix(train_features, name="train_features")
    q27 = _finite_vector(train_q27_scores_m3, name="train_q27_scores_m3")
    truth = _finite_vector(train_truth_m3, name="train_truth_m3")
    if len(q27) != len(truth) or x.shape[0] != len(truth):
        raise ValueError("V28R1 Train arrays are not aligned")
    if len(train_groups) != len(truth) or len(train_units) != len(truth):
        raise ValueError("V28R1 Train identities are not aligned")

    residual = truth - q27
    fold_count, assignment = _group_folds(train_groups, folds=cv_folds)
    configurations: list[dict[str, Any]] = []

    for ridge in ridge_grid:
        for alpha in shrinkage_grid:
            reports: list[dict[str, Any]] = []
            for fold in range(fold_count):
                holdout = np.asarray(
                    [assignment[str(value)] == fold for value in train_groups],
                    dtype=bool,
                )
                if not holdout.any() or not (~holdout).any():
                    continue
                base = _fit_safe_ridge(x[~holdout], residual[~holdout], float(ridge))
                model = _shrink_model(
                    base,
                    alpha=float(alpha),
                    q27_checkpoint_sha256=q27_checkpoint_sha256,
                )
                reports.append(
                    evaluate_v28_residual(
                        model,
                        x[holdout],
                        q27[holdout],
                        truth[holdout],
                        [train_units[index] for index in np.flatnonzero(holdout)],
                    )
                )
            if not reports:
                continue
            configurations.append(
                {
                    "ridge": float(ridge),
                    "residual_shrinkage": float(alpha),
                    "fold_count": len(reports),
                    "mean_decision_regret_m3": float(
                        np.mean([r["decision"]["mean_regret_m3"] for r in reports])
                    ),
                    "mean_selected_true_delta_tfv_m3": float(
                        np.mean(
                            [
                                r["decision"]["mean_selected_true_delta_tfv_m3"]
                                for r in reports
                            ]
                        )
                    ),
                    "mean_pairwise_rank_accuracy": float(
                        np.mean(
                            [r["pairwise"]["pairwise_rank_accuracy"] for r in reports]
                        )
                    ),
                    "mean_q28_rmse_m3": float(
                        np.mean([r["q28_metrics"]["rmse_m3"] for r in reports])
                    ),
                    "mean_q28_spearman": float(
                        np.mean([r["q28_metrics"]["spearman"] for r in reports])
                    ),
                    "fold_reports": reports,
                }
            )

    if not configurations:
        raise RuntimeError("V28R1 Train CV produced no configuration")

    configurations.sort(
        key=lambda row: (
            float(row["mean_decision_regret_m3"]),
            -float(row["mean_pairwise_rank_accuracy"]),
            float(row["mean_q28_rmse_m3"]),
            float(row["residual_shrinkage"]),
            float(row["ridge"]),
        )
    )
    selected = configurations[0]
    fitted = _fit_safe_ridge(x, residual, float(selected["ridge"]))
    model = _shrink_model(
        fitted,
        alpha=float(selected["residual_shrinkage"]),
        q27_checkpoint_sha256=q27_checkpoint_sha256,
    )

    unsafe_weights = {
        name: float(model.weight[index])
        for index, name in enumerate(V28_RESIDUAL_FEATURE_NAMES)
        if index not in V28R1_SUPPORTED_FEATURE_INDICES
    }
    if any(abs(value) > 0.0 for value in unsafe_weights.values()):
        raise RuntimeError(
            "V28R1 assigned nonzero weight to a train/deployment-mismatched feature"
        )

    return model, {
        "contract": V28R1_SUPPORTED_RESIDUAL_CONTRACT,
        "selection_contract": (
            "TRAIN_LEAKAGE_GROUP_CV_ONLY_DECISION_REGRET_THEN_RANK_RMSE_V1"
        ),
        "cv_folds": int(fold_count),
        "configuration_count": len(configurations),
        "ranked_configurations": configurations,
        "selected_ridge": float(selected["ridge"]),
        "selected_residual_shrinkage": float(selected["residual_shrinkage"]),
        "supported_feature_names": list(V28R1_SUPPORTED_FEATURE_NAMES),
        "zero_weight_feature_names": list(V28R1_ZERO_WEIGHT_FEATURE_NAMES),
        "validation_used_for_model_selection": False,
        "test_used_for_model_selection": False,
    }


__all__ = [
    "V28R1_RIDGE_GRID",
    "V28R1_SHRINKAGE_GRID",
    "V28R1_SUPPORTED_FEATURE_INDICES",
    "V28R1_SUPPORTED_FEATURE_NAMES",
    "V28R1_SUPPORTED_RESIDUAL_CONTRACT",
    "V28R1_ZERO_WEIGHT_FEATURE_NAMES",
    "fit_v28r1_supported_residual",
]
