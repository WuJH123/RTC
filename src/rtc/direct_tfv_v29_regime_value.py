"""Regime-balanced supported-manifold value correction for Project7 V29.

V29 keeps frozen Q27 as the policy-return backbone and fits only a compact,
continuous residual on causal hydraulic-regime/action features that are
available with identical semantics in offline q95-supported truth and online
execution. Leakage groups are equal-weighted so repeated records from one
event/context cannot dominate the regression. Hyperparameters are selected
using Train-group CV only; alpha=0 is an exact Q27 fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .direct_tfv_v28_residual_value import V28_RESIDUAL_FEATURE_NAMES

V29_REGIME_VALUE_CONTRACT = "PROJECT7_STEP3_V29_REGIME_BALANCED_SUPPORTED_VALUE_V1"
V29_REGIME_CHECKPOINT_CONTRACT = "PROJECT7_STEP3_V29_REGIME_BALANCED_SUPPORTED_VALUE_CHECKPOINT_V1"
V29_RIDGE_GRID = (0.0, 1.0e-3, 1.0e-2, 1.0e-1, 1.0, 10.0, 100.0)
V29_SHRINKAGE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)

V29_FEATURE_NAMES = (
    "q27_supported_score_m3",
    "supported_first_move_l1",
    "changed_facility_count",
    "network_stress_q75",
    "network_stress_squared",
    "rain_level",
    "strong_storm_blend",
    "abs_q27_supported_score_m3",
    "q27_x_stress",
    "q27_x_strong_storm_blend",
    "first_move_l1_x_stress",
    "changed_facility_count_x_stress",
    "candidate_family_probe_050",
    "candidate_family_probe_100",
    "candidate_family_hydraulic",
    "candidate_family_auto_rbc_shadow",
)

_V28_INDEX = {name: V28_RESIDUAL_FEATURE_NAMES.index(name) for name in (
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
)}


def build_v29_regime_features(v28_features: np.ndarray) -> np.ndarray:
    """Map deployment-parity V28 coordinates into continuous regime features."""
    x = np.asarray(v28_features, dtype=np.float64).reshape(-1)
    if x.size != len(V28_RESIDUAL_FEATURE_NAMES) or not np.isfinite(x).all():
        raise ValueError("V29 requires one finite V28 residual feature vector")
    q27 = float(x[_V28_INDEX["q27_supported_score_m3"]])
    first = float(x[_V28_INDEX["supported_first_move_l1"]])
    changed = float(x[_V28_INDEX["changed_facility_count"]])
    stress = float(x[_V28_INDEX["network_stress_q75"]])
    rain = float(x[_V28_INDEX["rain_level"]])
    blend = float(x[_V28_INDEX["strong_storm_blend"]])
    family = (
        float(x[_V28_INDEX["candidate_family_probe_050"]]),
        float(x[_V28_INDEX["candidate_family_probe_100"]]),
        float(x[_V28_INDEX["candidate_family_hydraulic"]]),
        float(x[_V28_INDEX["candidate_family_auto_rbc_shadow"]]),
    )
    out = np.asarray(
        (
            q27,
            first,
            changed,
            stress,
            stress * stress,
            rain,
            blend,
            abs(q27),
            q27 * stress,
            q27 * blend,
            first * stress,
            changed * stress,
            *family,
        ),
        dtype=np.float64,
    )
    if out.shape != (len(V29_FEATURE_NAMES),) or not np.isfinite(out).all():
        raise RuntimeError("V29 regime feature contract drifted")
    return out


def _finite_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != len(V29_FEATURE_NAMES):
        raise ValueError(f"{name} must be [N,{len(V29_FEATURE_NAMES)}]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _finite_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def group_balanced_row_weights(groups: Sequence[str]) -> np.ndarray:
    """Give every leakage group equal total weight while preserving all rows."""
    labels = np.asarray([str(value) for value in groups], dtype=object)
    if labels.size == 0 or any(not value for value in labels):
        raise ValueError("V29 group-balanced weighting requires non-empty group labels")
    unique, counts = np.unique(labels, return_counts=True)
    count_by_group = dict(zip(unique.tolist(), counts.tolist(), strict=True))
    raw = np.asarray([1.0 / float(count_by_group[value]) for value in labels], dtype=np.float64)
    return raw / float(raw.mean())


@dataclass(frozen=True)
class V29RegimeValueModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weight: np.ndarray
    intercept: float
    q27_checkpoint_sha256: str
    ridge: float
    shrinkage: float

    @property
    def feature_width(self) -> int:
        return int(self.weight.size)

    def predict_residual_many(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.feature_width or not np.isfinite(x).all():
            raise ValueError("V29 residual feature shape/content is invalid")
        z = (x - self.feature_mean) / self.feature_scale
        result = z @ self.weight + float(self.intercept)
        if not np.isfinite(result).all():
            raise ValueError("V29 residual prediction is non-finite")
        return result

    def predict_residual_m3(self, features: np.ndarray) -> float:
        return float(self.predict_residual_many(features)[0])


def _fit_weighted_ridge(
    features: np.ndarray,
    residual_truth: np.ndarray,
    groups: Sequence[str],
    *,
    ridge: float,
    shrinkage: float,
    q27_checkpoint_sha256: str,
) -> V29RegimeValueModel:
    x = _finite_matrix(features, name="features")
    y = _finite_vector(residual_truth, name="residual_truth")
    if x.shape[0] != y.size or len(groups) != y.size:
        raise ValueError("V29 weighted ridge inputs are not aligned")
    if float(ridge) < 0.0 or not math.isfinite(float(ridge)):
        raise ValueError("V29 ridge must be finite and non-negative")
    alpha = float(shrinkage)
    if not 0.0 <= alpha <= 1.0 or not math.isfinite(alpha):
        raise ValueError("V29 shrinkage must lie in [0,1]")
    weights = group_balanced_row_weights(groups)
    mean = np.average(x, axis=0, weights=weights)
    var = np.average(np.square(x - mean), axis=0, weights=weights)
    scale = np.maximum(np.sqrt(np.maximum(var, 0.0)), 1.0e-6)
    z = (x - mean) / scale
    design = np.column_stack((z, np.ones(len(z), dtype=np.float64)))
    root_w = np.sqrt(weights)
    weighted_design = design * root_w[:, None]
    weighted_target = y * root_w
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    lhs = weighted_design.T @ weighted_design + penalty
    rhs = weighted_design.T @ weighted_target
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return V29RegimeValueModel(
        feature_mean=mean,
        feature_scale=scale,
        weight=beta[:-1] * alpha,
        intercept=float(beta[-1]) * alpha,
        q27_checkpoint_sha256=str(q27_checkpoint_sha256).lower(),
        ridge=float(ridge),
        shrinkage=alpha,
    )


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(prediction: np.ndarray, truth: np.ndarray) -> float:
    if len(prediction) < 2 or np.std(prediction) <= 1.0e-12 or np.std(truth) <= 1.0e-12:
        return 0.0
    value = float(np.corrcoef(_rank(prediction), _rank(truth))[0, 1])
    return value if math.isfinite(value) else 0.0


def _decision_groups(units: Sequence[str]) -> list[tuple[str, np.ndarray]]:
    labels = np.asarray([str(value) for value in units], dtype=object)
    return [(unit, labels == unit) for unit in sorted(set(labels.tolist()))]


def evaluate_v29_value(
    model: V29RegimeValueModel,
    features: np.ndarray,
    q27_scores_m3: np.ndarray,
    truth_m3: np.ndarray,
    decision_units: Sequence[str],
    leakage_groups: Sequence[str] | None = None,
) -> dict[str, Any]:
    x = _finite_matrix(features, name="features")
    q27 = _finite_vector(q27_scores_m3, name="q27_scores_m3")
    truth = _finite_vector(truth_m3, name="truth_m3")
    if x.shape[0] != len(q27) or len(q27) != len(truth) or len(decision_units) != len(truth):
        raise ValueError("V29 evaluation arrays are not aligned")
    residual = model.predict_residual_many(x)
    q29 = q27 + residual
    error = q29 - truth

    pair_count = 0
    pair_correct = 0
    selected_truth: list[float] = []
    oracle_truth: list[float] = []
    selected_group: list[str] = []
    false_hold = 0
    beneficial_query = 0
    action = 0
    harmful_action = 0
    top1 = 0
    group_labels = [str(v) for v in leakage_groups] if leakage_groups is not None else [""] * len(truth)
    if len(group_labels) != len(truth):
        raise ValueError("V29 leakage groups are not aligned")

    for _unit, mask in _decision_groups(decision_units):
        idx = np.flatnonzero(mask)
        local_pred = q29[idx]
        local_truth = truth[idx]
        selected_local = int(np.argmin(local_pred))
        selected_idx = int(idx[selected_local])
        oracle = float(min(0.0, float(np.min(local_truth))))
        oracle_local = int(np.argmin(np.minimum(local_truth, 0.0)))
        top1 += int(selected_local == oracle_local)
        is_action = float(local_pred[selected_local]) < 0.0
        value = float(local_truth[selected_local]) if is_action else 0.0
        action += int(is_action)
        harmful_action += int(is_action and value > 0.0)
        has_benefit = bool(np.min(local_truth) < 0.0)
        beneficial_query += int(has_benefit)
        false_hold += int((not is_action) and has_benefit)
        selected_truth.append(value)
        oracle_truth.append(oracle)
        selected_group.append(group_labels[selected_idx])

        for left in range(len(idx)):
            for right in range(left + 1, len(idx)):
                true_delta = float(local_truth[left] - local_truth[right])
                if abs(true_delta) <= 1.0e-6:
                    continue
                pair_count += 1
                pred_delta = float(local_pred[left] - local_pred[right])
                pair_correct += int((pred_delta < 0.0) == (true_delta < 0.0))

    selected_array = np.asarray(selected_truth, dtype=np.float64)
    oracle_array = np.asarray(oracle_truth, dtype=np.float64)
    regrets = selected_array - oracle_array
    macro_regret = float(np.mean(regrets)) if len(regrets) else 0.0
    macro_false_hold = float(false_hold / max(1, beneficial_query))
    if leakage_groups is not None and selected_group:
        group_values: list[float] = []
        group_false: list[float] = []
        sg = np.asarray(selected_group, dtype=object)
        for group in sorted(set(selected_group)):
            gm = sg == group
            group_values.append(float(np.mean(regrets[gm])))
            group_false.append(float(np.mean((selected_array[gm] == 0.0) & (oracle_array[gm] < 0.0))))
        macro_regret = float(np.mean(group_values))
        macro_false_hold = float(np.mean(group_false))

    return {
        "candidate": {
            "mae_m3": float(np.mean(np.abs(error))),
            "rmse_m3": float(np.sqrt(np.mean(np.square(error)))),
            "spearman": _spearman(q29, truth),
            "sign_accuracy": float(np.mean((q29 < 0.0) == (truth < 0.0))),
        },
        "pairwise": {
            "pair_count": int(pair_count),
            "pairwise_rank_accuracy": float(pair_correct / pair_count) if pair_count else 0.5,
        },
        "decision": {
            "query_count": int(len(selected_array)),
            "action_count": int(action),
            "hold_count": int(len(selected_array) - action),
            "harmful_action_count": int(harmful_action),
            "false_hold_count": int(false_hold),
            "beneficial_query_count": int(beneficial_query),
            "false_hold_rate": float(false_hold / max(1, beneficial_query)),
            "top1_accuracy": float(top1 / max(1, len(selected_array))),
            "mean_selected_true_delta_tfv_m3": float(selected_array.mean()) if len(selected_array) else 0.0,
            "mean_regret_m3": float(np.mean(regrets)) if len(regrets) else 0.0,
            "macro_group_regret_m3": macro_regret,
            "macro_group_false_hold_rate": macro_false_hold,
        },
    }


def _fold_assignment(groups: Sequence[str], *, folds: int, seed: int) -> tuple[int, dict[str, int]]:
    unique = sorted(set(str(value) for value in groups))
    if len(unique) < 2:
        raise ValueError("V29 requires at least two Train leakage groups")
    count = max(2, min(int(folds), len(unique)))
    ordered = sorted(unique, key=lambda value: hashlib.sha256(f"v29|{seed}|{value}".encode("utf-8")).hexdigest())
    return count, {group: index % count for index, group in enumerate(ordered)}


def fit_v29_regime_value(
    *,
    train_features: np.ndarray,
    train_q27_scores_m3: np.ndarray,
    train_truth_m3: np.ndarray,
    train_groups: Sequence[str],
    train_units: Sequence[str],
    q27_checkpoint_sha256: str,
    cv_folds: int = 5,
    seed: int = 42,
    ridge_grid: Sequence[float] = V29_RIDGE_GRID,
    shrinkage_grid: Sequence[float] = V29_SHRINKAGE_GRID,
) -> tuple[V29RegimeValueModel, dict[str, Any]]:
    """Select a regime correction with Train leakage-group CV only."""
    x = _finite_matrix(train_features, name="train_features")
    q27 = _finite_vector(train_q27_scores_m3, name="train_q27_scores_m3")
    truth = _finite_vector(train_truth_m3, name="train_truth_m3")
    if len(q27) != len(truth) or x.shape[0] != len(truth):
        raise ValueError("V29 Train arrays are not aligned")
    if len(train_groups) != len(truth) or len(train_units) != len(truth):
        raise ValueError("V29 Train identities are not aligned")
    residual_truth = truth - q27
    fold_count, assignment = _fold_assignment(train_groups, folds=cv_folds, seed=seed)
    rows: list[dict[str, Any]] = []
    groups_array = np.asarray([str(v) for v in train_groups], dtype=object)

    for ridge in ridge_grid:
        for alpha in shrinkage_grid:
            fold_reports: list[dict[str, Any]] = []
            for fold in range(fold_count):
                holdout = np.asarray([assignment[str(v)] == fold for v in train_groups], dtype=bool)
                if not holdout.any() or not (~holdout).any():
                    continue
                model = _fit_weighted_ridge(
                    x[~holdout],
                    residual_truth[~holdout],
                    groups_array[~holdout].tolist(),
                    ridge=float(ridge),
                    shrinkage=float(alpha),
                    q27_checkpoint_sha256=q27_checkpoint_sha256,
                )
                idx = np.flatnonzero(holdout)
                fold_reports.append(
                    evaluate_v29_value(
                        model,
                        x[holdout],
                        q27[holdout],
                        truth[holdout],
                        [train_units[i] for i in idx],
                        [train_groups[i] for i in idx],
                    )
                )
            if not fold_reports:
                continue
            rows.append(
                {
                    "ridge": float(ridge),
                    "shrinkage": float(alpha),
                    "fold_count": len(fold_reports),
                    "mean_macro_group_regret_m3": float(np.mean([r["decision"]["macro_group_regret_m3"] for r in fold_reports])),
                    "mean_macro_group_false_hold_rate": float(np.mean([r["decision"]["macro_group_false_hold_rate"] for r in fold_reports])),
                    "mean_pairwise_rank_accuracy": float(np.mean([r["pairwise"]["pairwise_rank_accuracy"] for r in fold_reports])),
                    "mean_candidate_rmse_m3": float(np.mean([r["candidate"]["rmse_m3"] for r in fold_reports])),
                    "fold_reports": fold_reports,
                }
            )
    if not rows:
        raise RuntimeError("V29 Train group CV produced no configurations")
    rows.sort(
        key=lambda row: (
            float(row["mean_macro_group_regret_m3"]),
            float(row["mean_macro_group_false_hold_rate"]),
            -float(row["mean_pairwise_rank_accuracy"]),
            float(row["mean_candidate_rmse_m3"]),
            float(row["shrinkage"]),
            float(row["ridge"]),
        )
    )
    selected = rows[0]
    model = _fit_weighted_ridge(
        x,
        residual_truth,
        train_groups,
        ridge=float(selected["ridge"]),
        shrinkage=float(selected["shrinkage"]),
        q27_checkpoint_sha256=q27_checkpoint_sha256,
    )
    stress = x[:, V29_FEATURE_NAMES.index("network_stress_q75")]
    return model, {
        "contract": V29_REGIME_VALUE_CONTRACT,
        "selection_contract": "TRAIN_GROUP_BALANCED_CV_MACRO_DECISION_REGRET_FALSE_HOLD_V1",
        "cv_folds": int(fold_count),
        "configuration_count": len(rows),
        "ranked_configurations": rows,
        "selected_ridge": float(selected["ridge"]),
        "selected_shrinkage": float(selected["shrinkage"]),
        "train_stress_q25": float(np.quantile(stress, 0.25)),
        "train_stress_q50": float(np.quantile(stress, 0.50)),
        "train_stress_q75": float(np.quantile(stress, 0.75)),
        "validation_used_for_model_selection": False,
        "test_used_for_model_selection": False,
        "return_period_used_as_feature": False,
        "event_duration_used_as_feature": False,
        "event_id_used_as_feature": False,
    }


def checkpoint_payload(
    model: V29RegimeValueModel,
    *,
    lineage: Mapping[str, Any],
    selection_report: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    test_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contract": V29_REGIME_CHECKPOINT_CONTRACT,
        "model_contract": V29_REGIME_VALUE_CONTRACT,
        "feature_names": list(V29_FEATURE_NAMES),
        "feature_mean": model.feature_mean.tolist(),
        "feature_scale": model.feature_scale.tolist(),
        "weight": model.weight.tolist(),
        "intercept": float(model.intercept),
        "q27_checkpoint_sha256": model.q27_checkpoint_sha256,
        "ridge": float(model.ridge),
        "shrinkage": float(model.shrinkage),
        "lineage": dict(lineage),
        "selection_report": dict(selection_report),
        "validation_report": dict(validation_report),
        "test_report": dict(test_report),
        "development_only": True,
        "formal_evidence": False,
    }


def load_v29_regime_value_model(
    path: str,
    *,
    expected_q27_checkpoint_sha256: str,
    expected_dataset_manifest_sha256: str | None = None,
) -> tuple[V29RegimeValueModel, dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("contract") != V29_REGIME_CHECKPOINT_CONTRACT:
        raise ValueError("V29 runtime requires a V29 regime-value checkpoint")
    if tuple(str(v) for v in payload.get("feature_names", ())) != V29_FEATURE_NAMES:
        raise ValueError("V29 regime feature contract drifted")
    q27_sha = str(payload.get("q27_checkpoint_sha256", "")).lower()
    if q27_sha != str(expected_q27_checkpoint_sha256).lower():
        raise ValueError("V29 checkpoint is bound to another Q27 checkpoint")
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("V29 checkpoint lacks lineage")
    if expected_dataset_manifest_sha256 is not None and str(lineage.get("dataset_manifest_sha256", "")).lower() != str(expected_dataset_manifest_sha256).lower():
        raise ValueError("V29 checkpoint/dataset lineage mismatch")
    model = V29RegimeValueModel(
        feature_mean=_finite_vector(payload.get("feature_mean"), name="feature_mean"),
        feature_scale=np.maximum(_finite_vector(payload.get("feature_scale"), name="feature_scale"), 1.0e-6),
        weight=_finite_vector(payload.get("weight"), name="weight"),
        intercept=float(payload.get("intercept", 0.0)),
        q27_checkpoint_sha256=q27_sha,
        ridge=float(payload.get("ridge", 0.0)),
        shrinkage=float(payload.get("shrinkage", 0.0)),
    )
    if model.feature_width != len(V29_FEATURE_NAMES):
        raise ValueError("V29 checkpoint feature width is invalid")
    return model, payload


__all__ = [
    "V29_FEATURE_NAMES",
    "V29_REGIME_CHECKPOINT_CONTRACT",
    "V29_REGIME_VALUE_CONTRACT",
    "V29RegimeValueModel",
    "build_v29_regime_features",
    "checkpoint_payload",
    "evaluate_v29_value",
    "fit_v29_regime_value",
    "group_balanced_row_weights",
    "load_v29_regime_value_model",
]
