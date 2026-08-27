"""Decision-aware exact policy-return value model for Project7 Step3 V27.

V26 fitted pointwise exact-return magnitude and then used the fitted values to rank candidates within
one causal state.  V27 aligns training with deployment by augmenting pointwise return regression with
same-state pairwise return-difference equations.  Hyperparameters are screened by leakage-group CV
inside Train and finalized on Validation; Test never participates in fitting or model selection.

Runtime selection uses the *unclipped latent coordinate*.  The asinh/sinh transform is monotone and
maps physical HOLD=0 to latent zero, so ranking and ACTION/HOLD do not require the clipped m3 inverse
transform.  Clipped m3 values remain reporting-only diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .direct_tfv_v26_value_model import candidate_metrics, decision_metrics


V27_VALUE_MODEL_CONTRACT = "PROJECT7_STEP3_V27_DECISION_AWARE_PAIRWISE_EXACT_RETURN_RIDGE_V1"
V27_VALUE_CHECKPOINT_CONTRACT = "PROJECT7_STEP3_V27_DECISION_AWARE_VALUE_CHECKPOINT_V1"
V27_RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
V27_PAIRWISE_WEIGHT_GRID = (0.0, 0.5, 1.0, 2.0, 4.0)
V27_REPORT_LATENT_CLIP = 8.0


def _matrix(value: Any, *, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 2 or out.shape[0] < 1 or out.shape[1] < 1 or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite non-empty matrix")
    return out


def _vector(value: Any, *, name: str) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64).reshape(-1)
    if out.size < 1 or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a finite non-empty vector")
    return out


def _pair_indices(decision_units: Sequence[str]) -> list[tuple[int, int]]:
    grouped: dict[str, list[int]] = {}
    for index, unit in enumerate(decision_units):
        grouped.setdefault(str(unit), []).append(index)
    pairs: list[tuple[int, int]] = []
    for unit in sorted(grouped):
        local = grouped[unit]
        for left_pos in range(len(local)):
            for right_pos in range(left_pos + 1, len(local)):
                pairs.append((local[left_pos], local[right_pos]))
    return pairs


def pairwise_rank_accuracy(
    scores: np.ndarray,
    truth: np.ndarray,
    decision_units: Sequence[str],
    *,
    tie_tolerance_m3: float = 1.0e-6,
) -> dict[str, float | int]:
    pred = _vector(scores, name="scores")
    y = _vector(truth, name="truth")
    if pred.size != y.size or len(decision_units) != pred.size:
        raise ValueError("pairwise metric inputs are not aligned")
    correct = 0
    count = 0
    for i, j in _pair_indices(decision_units):
        true_delta = float(y[i] - y[j])
        if abs(true_delta) <= float(tie_tolerance_m3):
            continue
        predicted_delta = float(pred[i] - pred[j])
        correct += int((predicted_delta < 0.0) == (true_delta < 0.0))
        count += 1
    return {
        "pair_count": int(count),
        "pairwise_rank_accuracy": float(correct / count) if count else 0.5,
    }


def _group_folds(groups: Sequence[str], *, folds: int, seed: int) -> list[np.ndarray]:
    unique = sorted(set(str(value) for value in groups))
    if len(unique) < 2:
        raise ValueError("V27 group CV requires at least two Train leakage groups")
    k = min(max(2, int(folds)), len(unique))
    ordered = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest(),
    )
    assignment = {group: index % k for index, group in enumerate(ordered)}
    group_array = np.asarray([str(value) for value in groups], dtype=object)
    return [np.asarray([assignment[value] == fold for value in group_array], dtype=bool) for fold in range(k)]


@dataclass(frozen=True)
class FittedV27DecisionValueModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weight: np.ndarray
    intercept: float
    target_scale_m3: float
    ridge: float
    pairwise_weight: float

    @property
    def feature_width(self) -> int:
        return int(self.weight.size)

    def latent_numpy(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or int(x.shape[1]) != self.feature_width or not np.isfinite(x).all():
            raise ValueError("V27 feature shape/content is invalid")
        return ((x - self.feature_mean) / self.feature_scale) @ self.weight + self.intercept

    def predict_m3_numpy(self, features: np.ndarray) -> np.ndarray:
        latent = self.latent_numpy(features)
        return np.sinh(np.clip(latent, -V27_REPORT_LATENT_CLIP, V27_REPORT_LATENT_CLIP)) * float(
            self.target_scale_m3
        )

    def clip_hit_numpy(self, features: np.ndarray) -> np.ndarray:
        return np.abs(self.latent_numpy(features)) >= V27_REPORT_LATENT_CLIP


class V27DecisionValueModule(nn.Module):
    def __init__(self, fitted: FittedV27DecisionValueModel) -> None:
        super().__init__()
        self.register_buffer("feature_mean", torch.as_tensor(fitted.feature_mean, dtype=torch.float32))
        self.register_buffer("feature_scale", torch.as_tensor(fitted.feature_scale, dtype=torch.float32))
        self.register_buffer("weight", torch.as_tensor(fitted.weight, dtype=torch.float32))
        self.register_buffer("intercept", torch.tensor(float(fitted.intercept), dtype=torch.float32))
        self.register_buffer("target_scale_m3", torch.tensor(float(fitted.target_scale_m3), dtype=torch.float32))

    def latent_score(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim != 1 or int(feature.numel()) != int(self.weight.numel()):
            raise ValueError("V27 runtime feature width drifted")
        x = feature.to(dtype=self.weight.dtype, device=self.weight.device)
        return torch.dot((x - self.feature_mean) / self.feature_scale, self.weight) + self.intercept

    def reported_prediction_m3(self, feature: torch.Tensor) -> torch.Tensor:
        latent = self.latent_score(feature)
        return torch.sinh(torch.clamp(latent, -V27_REPORT_LATENT_CLIP, V27_REPORT_LATENT_CLIP)) * self.target_scale_m3

    def clip_hit(self, feature: torch.Tensor) -> bool:
        return bool(torch.abs(self.latent_score(feature)).detach().cpu() >= V27_REPORT_LATENT_CLIP)


def _fit_augmented_ridge(
    features: np.ndarray,
    truth_m3: np.ndarray,
    decision_units: Sequence[str],
    *,
    ridge: float,
    pairwise_weight: float,
    target_scale_m3: float,
) -> FittedV27DecisionValueModel:
    x = _matrix(features, name="features")
    y = _vector(truth_m3, name="truth_m3")
    if x.shape[0] != y.size or len(decision_units) != y.size:
        raise ValueError("V27 fit inputs are not aligned")
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1.0e-6)
    z = (x - mean) / scale
    transformed = np.arcsinh(y / float(target_scale_m3))

    point_design = np.column_stack((z, np.ones(len(z), dtype=np.float64)))
    rows = [point_design]
    targets = [transformed]
    pairs = _pair_indices(decision_units)
    if pairwise_weight > 0.0 and pairs:
        pair_design = np.empty((len(pairs), z.shape[1] + 1), dtype=np.float64)
        pair_target = np.empty(len(pairs), dtype=np.float64)
        for index, (left, right) in enumerate(pairs):
            pair_design[index, :-1] = z[left] - z[right]
            pair_design[index, -1] = 0.0
            pair_target[index] = transformed[left] - transformed[right]
        # Keep pairwise loss interpretable as a dataset-level weight rather than allowing decision
        # units with many candidates to dominate solely because they generate more pairs.
        pair_scale = math.sqrt(float(pairwise_weight) * len(z) / max(1, len(pairs)))
        rows.append(pair_design * pair_scale)
        targets.append(pair_target * pair_scale)

    design = np.vstack(rows)
    objective = np.concatenate(targets)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ objective
    try:
        beta = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return FittedV27DecisionValueModel(
        feature_mean=mean,
        feature_scale=scale,
        weight=beta[:-1],
        intercept=float(beta[-1]),
        target_scale_m3=float(target_scale_m3),
        ridge=float(ridge),
        pairwise_weight=float(pairwise_weight),
    )


def _evaluate(
    model: FittedV27DecisionValueModel,
    features: np.ndarray,
    truth: np.ndarray,
    decision_units: Sequence[str],
) -> dict[str, Any]:
    latent = model.latent_numpy(features)
    prediction = model.predict_m3_numpy(features)
    candidate = candidate_metrics(prediction, truth)
    decision = decision_metrics(latent, truth, decision_units)
    pairwise = pairwise_rank_accuracy(latent, truth, decision_units)
    return {
        "candidate": candidate,
        "decision": decision,
        "pairwise": pairwise,
        "latent_min": float(latent.min()),
        "latent_max": float(latent.max()),
        "latent_std": float(latent.std()),
        "report_clip_hit_count": int(model.clip_hit_numpy(features).sum()),
        "report_clip_hit_fraction": float(model.clip_hit_numpy(features).mean()),
    }


def fit_v27_decision_value_model(
    train_features: np.ndarray,
    train_truth: np.ndarray,
    train_decision_units: Sequence[str],
    train_leakage_groups: Sequence[str],
    validation_features: np.ndarray,
    validation_truth: np.ndarray,
    validation_decision_units: Sequence[str],
    *,
    seed: int = 42,
    cv_folds: int = 5,
    ridge_grid: Sequence[float] = V27_RIDGE_GRID,
    pairwise_weight_grid: Sequence[float] = V27_PAIRWISE_WEIGHT_GRID,
    validation_shortlist_size: int = 5,
) -> tuple[FittedV27DecisionValueModel, dict[str, Any]]:
    x_train = _matrix(train_features, name="train_features")
    y_train = _vector(train_truth, name="train_truth")
    x_val = _matrix(validation_features, name="validation_features")
    y_val = _vector(validation_truth, name="validation_truth")
    if x_train.shape[0] != y_train.size or len(train_decision_units) != y_train.size:
        raise ValueError("V27 Train arrays are not aligned")
    if len(train_leakage_groups) != y_train.size:
        raise ValueError("V27 Train leakage groups are not aligned")
    if x_val.shape[0] != y_val.size or len(validation_decision_units) != y_val.size:
        raise ValueError("V27 Validation arrays are not aligned")
    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError("V27 Train/Validation feature widths differ")

    target_scale = float(max(np.median(np.abs(y_train)), 1.0))
    folds = _group_folds(train_leakage_groups, folds=int(cv_folds), seed=int(seed))
    cv_rows: list[dict[str, Any]] = []
    for ridge_value in ridge_grid:
        for pair_weight_value in pairwise_weight_grid:
            ridge = float(ridge_value)
            pair_weight = float(pair_weight_value)
            fold_reports: list[dict[str, Any]] = []
            for fold_index, holdout in enumerate(folds):
                fit_mask = ~holdout
                if not bool(fit_mask.any()) or not bool(holdout.any()):
                    continue
                model = _fit_augmented_ridge(
                    x_train[fit_mask],
                    y_train[fit_mask],
                    [train_decision_units[i] for i in np.flatnonzero(fit_mask)],
                    ridge=ridge,
                    pairwise_weight=pair_weight,
                    target_scale_m3=target_scale,
                )
                report = _evaluate(
                    model,
                    x_train[holdout],
                    y_train[holdout],
                    [train_decision_units[i] for i in np.flatnonzero(holdout)],
                )
                report["fold"] = int(fold_index)
                fold_reports.append(report)
            if not fold_reports:
                continue
            cv_row = {
                "ridge": ridge,
                "pairwise_weight": pair_weight,
                "fold_count": len(fold_reports),
                "mean_decision_regret_m3": float(
                    np.mean([row["decision"]["mean_regret_m3"] for row in fold_reports])
                ),
                "mean_selected_true_delta_tfv_m3": float(
                    np.mean([row["decision"]["mean_selected_true_delta_tfv_m3"] for row in fold_reports])
                ),
                "mean_pairwise_rank_accuracy": float(
                    np.mean([row["pairwise"]["pairwise_rank_accuracy"] for row in fold_reports])
                ),
                "mean_candidate_rmse_m3": float(
                    np.mean([row["candidate"]["rmse_m3"] for row in fold_reports])
                ),
                "fold_reports": fold_reports,
            }
            cv_rows.append(cv_row)
    if not cv_rows:
        raise RuntimeError("V27 Train group CV produced no model configurations")

    ordered = sorted(
        cv_rows,
        key=lambda row: (
            float(row["mean_decision_regret_m3"]),
            float(row["mean_selected_true_delta_tfv_m3"]),
            -float(row["mean_pairwise_rank_accuracy"]),
            float(row["mean_candidate_rmse_m3"]),
            float(row["ridge"]),
            float(row["pairwise_weight"]),
        ),
    )
    shortlist = ordered[: min(max(1, int(validation_shortlist_size)), len(ordered))]
    validation_rows: list[dict[str, Any]] = []
    fitted_by_key: dict[tuple[float, float], FittedV27DecisionValueModel] = {}
    for cv_rank, row in enumerate(shortlist):
        key = (float(row["ridge"]), float(row["pairwise_weight"]))
        fitted = _fit_augmented_ridge(
            x_train,
            y_train,
            train_decision_units,
            ridge=key[0],
            pairwise_weight=key[1],
            target_scale_m3=target_scale,
        )
        fitted_by_key[key] = fitted
        report = _evaluate(fitted, x_val, y_val, validation_decision_units)
        validation_rows.append(
            {
                "cv_rank": int(cv_rank),
                "ridge": key[0],
                "pairwise_weight": key[1],
                "validation": report,
            }
        )
    selected_row = min(
        validation_rows,
        key=lambda row: (
            float(row["validation"]["decision"]["mean_regret_m3"]),
            float(row["validation"]["decision"]["mean_selected_true_delta_tfv_m3"]),
            -float(row["validation"]["pairwise"]["pairwise_rank_accuracy"]),
            float(row["validation"]["candidate"]["rmse_m3"]),
            int(row["cv_rank"]),
        ),
    )
    selected_key = (float(selected_row["ridge"]), float(selected_row["pairwise_weight"]))
    selected = fitted_by_key[selected_key]
    report = {
        "selection_contract": "TRAIN_GROUP_CV_SHORTLIST_THEN_VALIDATION_DECISION_REGRET_V1",
        "target_scale_m3": target_scale,
        "cv_folds": len(folds),
        "cv_configuration_count": len(cv_rows),
        "cv_ranked_configurations": ordered,
        "validation_shortlist": validation_rows,
        "selected_ridge": selected.ridge,
        "selected_pairwise_weight": selected.pairwise_weight,
        "validation_selected_metrics": selected_row["validation"],
        "test_used_for_training_or_model_selection": False,
        "scientific_metrics_block_runtime": False,
    }
    return selected, report


def checkpoint_payload(
    fitted: FittedV27DecisionValueModel,
    *,
    lineage: Mapping[str, Any],
    training_report: Mapping[str, Any],
    validation_report: Mapping[str, Any],
    test_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "contract": V27_VALUE_CHECKPOINT_CONTRACT,
        "model_contract": V27_VALUE_MODEL_CONTRACT,
        "feature_mean": fitted.feature_mean.tolist(),
        "feature_scale": fitted.feature_scale.tolist(),
        "weight": fitted.weight.tolist(),
        "intercept": float(fitted.intercept),
        "target_scale_m3": float(fitted.target_scale_m3),
        "ridge": float(fitted.ridge),
        "pairwise_weight": float(fitted.pairwise_weight),
        "lineage": dict(lineage),
        "training_report": dict(training_report),
        "validation_report": dict(validation_report),
        "test_report": dict(test_report),
        "runtime_ranking_uses_unclipped_latent": True,
        "reported_m3_clip": float(V27_REPORT_LATENT_CLIP),
        "development_only": True,
        "formal_evidence": False,
    }


def load_v27_value_model(
    path: str,
    *,
    device: torch.device,
    expected_lineage: Mapping[str, Any],
) -> tuple[V27DecisionValueModule, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("contract") != V27_VALUE_CHECKPOINT_CONTRACT:
        raise ValueError("V27 runtime requires a V27 decision-aware value checkpoint")
    lineage = payload.get("lineage")
    if not isinstance(lineage, dict):
        raise ValueError("V27 checkpoint lacks lineage")
    for key, expected in expected_lineage.items():
        actual = lineage.get(key)
        if isinstance(expected, str):
            if not isinstance(actual, str) or actual.lower() != expected.lower():
                raise ValueError(f"V27 checkpoint lineage mismatch: {key}")
        elif actual != expected:
            raise ValueError(f"V27 checkpoint lineage mismatch: {key}")
    fitted = FittedV27DecisionValueModel(
        feature_mean=_vector(payload["feature_mean"], name="feature_mean"),
        feature_scale=np.maximum(_vector(payload["feature_scale"], name="feature_scale"), 1.0e-6),
        weight=_vector(payload["weight"], name="weight"),
        intercept=float(payload["intercept"]),
        target_scale_m3=float(payload["target_scale_m3"]),
        ridge=float(payload["ridge"]),
        pairwise_weight=float(payload["pairwise_weight"]),
    )
    if fitted.feature_mean.size != fitted.weight.size or fitted.feature_scale.size != fitted.weight.size:
        raise ValueError("V27 checkpoint feature width mismatch")
    if fitted.target_scale_m3 <= 0.0 or not math.isfinite(fitted.target_scale_m3):
        raise ValueError("V27 checkpoint target scale is invalid")
    model = V27DecisionValueModule(fitted).to(device)
    model.eval()
    return model, payload


__all__ = [
    "FittedV27DecisionValueModel",
    "V27DecisionValueModule",
    "V27_PAIRWISE_WEIGHT_GRID",
    "V27_REPORT_LATENT_CLIP",
    "V27_RIDGE_GRID",
    "V27_VALUE_CHECKPOINT_CONTRACT",
    "V27_VALUE_MODEL_CONTRACT",
    "checkpoint_payload",
    "fit_v27_decision_value_model",
    "load_v27_value_model",
    "pairwise_rank_accuracy",
]
