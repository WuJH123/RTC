"""Train Step3 V18 on the existing 90-group development bank with zero new SWMM truth.

V18 reuses and freezes the development-validated V15 rank branch. The selected-action/HOLD problem is
fit as a small-sample boundary task: deterministic selected-candidate features are standardized using
Train only, compressed with Train-only PCA, and scored by an L2-regularized linear logistic model.
The ACTION/HOLD threshold is selected only from six-fold out-of-fold Train scores using decision error
and exact-return regret. Validation is one-shot after this Train-only design; Calibration truth stays
sealed unless Validation passes. No action/HOLD quota is imposed on held-out data.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from rtc.checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from rtc.direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
    DIRECT_TFV_POLICY_RETURN_ESTIMAND,
    sha256_file,
)
from rtc.direct_tfv_policy_return_query_margin_v17 import (
    LEGACY_V15_CHECKPOINT_CONTRACT,
    QueryConditionedPolicyReturnAdapterV17,
    import_v15_rank_state,
    rank_state_sha256,
)
from rtc.direct_tfv_policy_return_query_margin_v18 import (
    BoundaryPreprocessorV18,
    DIRECT_TFV_QUERY_MARGIN_V18_CHECKPOINT_CONTRACT,
    DIRECT_TFV_QUERY_MARGIN_V18_CONTRACT,
    LinearBoundaryCalibratorV18,
    PCA_COMPONENTS,
    build_boundary_feature_parts_v18,
)
from rtc.native_supervisory_control import load_native_supervisory_control
from rtc.production_cli import _load_graph
from build_step3_development_bank_current import STEP3_DEVELOPMENT_BANK_CONTRACT
from train_direct_tfv_policy_return_query_margin_current import _load_dataset, _rank_scores
from train_step3_query_margin_v2_current import _development_metadata, _features, _model_state_sha256


STEP3_V18_TRAINING_CONTRACT = (
    "PROJECT7_STEP3_V18_FROZEN_V15_RANK_TRAIN_OOF_REGULARIZED_BOUNDARY"
)
CV_FOLDS = 6
RIDGE_GRID = (0.01, 0.1, 1.0)
TRAIN_OOF_AUC_MIN = 0.60

DEVELOPMENT_ACCEPTANCE = {
    "selected_action_false_beneficial_fraction_max": 0.25,
    "selected_action_false_reject_fraction_max": 0.25,
    "hold_aware_decision_accuracy_min": 0.60,
    "within_query_pairwise_rank_accuracy_min": 0.60,
    "within_query_candidate_top1_accuracy_min": 0.55,
    "execute_all_collapse_forbidden_when_oracle_hold_exists": True,
    "hold_all_collapse_forbidden_when_oracle_action_exists": True,
}


def _query_inputs(
    rank_model: torch.nn.Module,
    normalization: Any,
    graph: Any,
    dataset: dict[str, Any],
    query: str,
    *,
    mask: np.ndarray,
    scale: float,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    indices = np.flatnonzero(dataset["queries"] == query)
    truth = torch.as_tensor(
        dataset["true_policy_return_delta_tfv_m3"][indices],
        dtype=torch.float32,
        device=device,
    ).reshape(-1)
    with torch.no_grad():
        raw = _rank_scores(
            rank_model,
            dataset,
            indices,
            graph=graph,
            normalization=normalization,
            device=device,
        )
        context, candidates = _features(
            rank_model,
            normalization,
            graph,
            dataset,
            indices,
            mask=mask,
            scale=scale,
            device=device,
        )
    return indices, truth, raw, context, candidates


def _rank_source_payload(
    path: str | Path,
    *,
    base_step2_sha256: str,
    train_sha256: str,
    validation_sha256: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or payload.get("contract") != LEGACY_V15_CHECKPOINT_CONTRACT:
        raise ValueError("V18 requires the V15 selection-consistent checkpoint as rank source")
    if str(payload.get("base_step2_sha256", "")).lower() != base_step2_sha256.lower():
        raise ValueError("V18 rank source uses another base Step2 checkpoint")
    if str(payload.get("train_dataset_sha256", "")).lower() != train_sha256.lower():
        raise ValueError("V18 rank source uses another development Train split")
    if str(payload.get("validation_dataset_sha256", "")).lower() != validation_sha256.lower():
        raise ValueError("V18 rank source uses another development Validation split")
    metrics = payload.get("validation_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("V18 rank source lacks validation metrics")
    if float(metrics.get("within_query_pairwise_rank_accuracy", 0.0)) < 0.90:
        raise ValueError("V18 rank source pairwise accuracy is below 0.90")
    if float(metrics.get("within_query_candidate_top1_accuracy", 0.0)) < 0.90:
        raise ValueError("V18 rank source top1 accuracy is below 0.90")
    return payload


def _make_rank_adapter(
    *,
    rank_model: torch.nn.Module,
    normalization: Any,
    graph: Any,
    train: dict[str, Any],
    rank_source: dict[str, Any],
    mask: np.ndarray,
    scale: float,
    device: torch.device,
) -> QueryConditionedPolicyReturnAdapterV17:
    first_query = sorted(set(train["queries"].tolist()))[0]
    indices = np.flatnonzero(train["queries"] == first_query)
    with torch.no_grad():
        context, candidates = _features(
            rank_model,
            normalization,
            graph,
            train,
            indices,
            mask=mask,
            scale=scale,
            device=device,
        )
    adapter = QueryConditionedPolicyReturnAdapterV17(
        target_scale_m3=scale,
        context_dim=int(context.numel()),
        candidate_dim=int(candidates.shape[1]),
    ).to(device)
    import_v15_rank_state(adapter, rank_source)
    adapter.eval()
    return adapter


def _collect_boundary_rows(
    rank_model: torch.nn.Module,
    rank_adapter: QueryConditionedPolicyReturnAdapterV17,
    dataset: dict[str, Any],
    *,
    graph: Any,
    normalization: Any,
    mask: np.ndarray,
    device: torch.device,
    scale: float,
) -> dict[str, Any]:
    dense_rows: list[np.ndarray] = []
    explicit_rows: list[np.ndarray] = []
    returns: list[float] = []
    queries: list[str] = []
    selected_sources: list[str] = []
    oracle_sources: list[str] = []
    rank_rows: list[np.ndarray] = []
    truth_rows: list[np.ndarray] = []

    with torch.no_grad():
        for query in sorted(set(dataset["queries"].tolist())):
            indices, truth, raw, context, candidates = _query_inputs(
                rank_model,
                normalization,
                graph,
                dataset,
                query,
                mask=mask,
                scale=scale,
                device=device,
            )
            out = rank_adapter(
                raw_rank_scores_m3=raw,
                context_features=context,
                candidate_features=candidates,
            )
            selected = int(out.selected_candidate_index.detach().cpu())
            oracle = int(torch.argmin(truth).detach().cpu())
            parts = build_boundary_feature_parts_v18(
                context_features=context,
                candidate_features=candidates,
                raw_rank_scores_m3=raw,
                rank_scores_normalized=out.rank_scores_normalized,
                selected_candidate_index=out.selected_candidate_index,
                target_scale_m3=scale,
            )
            dense_rows.append(parts.dense.detach().cpu().numpy().astype(np.float64))
            explicit_rows.append(parts.explicit.detach().cpu().numpy().astype(np.float64))
            returns.append(float(truth[selected].detach().cpu()))
            queries.append(query)
            selected_sources.append(str(dataset["sources"][indices[selected]]))
            oracle_sources.append(str(dataset["sources"][indices[oracle]]))
            rank_rows.append(out.rank_scores_normalized.detach().cpu().numpy().astype(np.float64))
            truth_rows.append(truth.detach().cpu().numpy().astype(np.float64))
    return {
        "dense": np.stack(dense_rows),
        "explicit": np.stack(explicit_rows),
        "returns": np.asarray(returns, dtype=np.float64),
        "queries": np.asarray(queries),
        "selected_sources": np.asarray(selected_sources),
        "oracle_sources": np.asarray(oracle_sources),
        "rank_rows": rank_rows,
        "truth_rows": truth_rows,
    }


def _fit_preprocessor(dense: np.ndarray, explicit: np.ndarray) -> dict[str, np.ndarray]:
    dense_mean = dense.mean(axis=0)
    dense_std = dense.std(axis=0)
    dense_std = np.where(dense_std > 1.0e-6, dense_std, 1.0)
    dense_z = (dense - dense_mean) / dense_std
    _u, _s, vh = np.linalg.svd(dense_z, full_matrices=False)
    k = min(PCA_COMPONENTS, max(1, dense.shape[0] - 1), dense.shape[1], vh.shape[0])
    components = vh[:k]
    explicit_mean = explicit.mean(axis=0)
    explicit_std = explicit.std(axis=0)
    explicit_std = np.where(explicit_std > 1.0e-6, explicit_std, 1.0)
    return {
        "dense_mean": dense_mean,
        "dense_std": dense_std,
        "components": components,
        "explicit_mean": explicit_mean,
        "explicit_std": explicit_std,
    }


def _transform(
    state: dict[str, np.ndarray], dense: np.ndarray, explicit: np.ndarray
) -> np.ndarray:
    dense_z = (dense - state["dense_mean"]) / state["dense_std"]
    projected = dense_z @ state["components"].T
    explicit_z = (explicit - state["explicit_mean"]) / state["explicit_std"]
    return np.concatenate((projected, explicit_z), axis=1)


def _fit_logistic(z: np.ndarray, hold: np.ndarray, ridge: float) -> tuple[np.ndarray, float]:
    x = np.concatenate((z, np.ones((z.shape[0], 1), dtype=np.float64)), axis=1)
    beta = np.zeros(x.shape[1], dtype=np.float64)
    reg = np.eye(x.shape[1], dtype=np.float64) * float(ridge)
    reg[-1, -1] = 0.0
    for _ in range(80):
        logits = np.clip(x @ beta, -30.0, 30.0)
        prob = 1.0 / (1.0 + np.exp(-logits))
        grad = (x.T @ (prob - hold)) / x.shape[0] + reg @ beta
        weight = np.clip(prob * (1.0 - prob), 1.0e-5, None)
        hessian = (x.T @ (x * weight[:, None])) / x.shape[0] + reg
        hessian += np.eye(x.shape[1]) * 1.0e-7
        step = np.linalg.solve(hessian, grad)
        beta -= step
        if float(np.linalg.norm(step)) < 1.0e-8:
            break
    return beta[:-1], float(beta[-1])


def _fit_magnitude(z: np.ndarray, returns: np.ndarray, scale: float, ridge: float) -> tuple[np.ndarray, float]:
    target = np.abs(np.arcsinh(returns / float(scale)))
    x = np.concatenate((z, np.ones((z.shape[0], 1), dtype=np.float64)), axis=1)
    reg = np.eye(x.shape[1], dtype=np.float64) * float(ridge)
    reg[-1, -1] = 0.0
    beta = np.linalg.solve(x.T @ x + reg + np.eye(x.shape[1]) * 1.0e-7, x.T @ target)
    return beta[:-1], float(beta[-1])


def _stratified_folds(returns: np.ndarray, queries: np.ndarray, seed: int) -> list[np.ndarray]:
    hold = [i for i, value in enumerate(returns) if value >= 0.0]
    action = [i for i, value in enumerate(returns) if value < 0.0]
    if min(len(hold), len(action)) < CV_FOLDS:
        raise ValueError("V18 Train needs at least one HOLD and ACTION query per OOF fold")
    folds: list[list[int]] = [[] for _ in range(CV_FOLDS)]
    for label, values in (("hold", hold), ("action", action)):
        ordered = sorted(
            values,
            key=lambda i: hashlib.sha256(
                f"v18|{seed}|{label}|{queries[i]}".encode("utf-8")
            ).hexdigest(),
        )
        for position, index in enumerate(ordered):
            folds[position % CV_FOLDS].append(index)
    return [np.asarray(sorted(values), dtype=np.int64) for values in folds]


def _auc_hold(scores: np.ndarray, returns: np.ndarray) -> float:
    hold = scores[returns >= 0.0]
    action = scores[returns < 0.0]
    if hold.size == 0 or action.size == 0:
        return 0.5
    wins = 0.0
    total = 0
    for left in hold:
        for right in action:
            wins += float(left > right) + 0.5 * float(left == right)
            total += 1
    return wins / max(1, total)


def _threshold_metrics(scores: np.ndarray, returns: np.ndarray, threshold: float) -> dict[str, float | bool]:
    execute = scores < float(threshold)
    hold_truth = returns >= 0.0
    action_truth = ~hold_truth
    n = max(1, returns.size)
    fb = float(np.sum(execute & hold_truth) / n)
    fr = float(np.sum((~execute) & action_truth) / n)
    regret = np.where(execute & hold_truth, returns, 0.0)
    regret += np.where((~execute) & action_truth, -returns, 0.0)
    predicted_hold = float(np.mean(~execute))
    oracle_hold = float(np.mean(hold_truth))
    return {
        "fb": fb,
        "fr": fr,
        "worst": max(fb, fr),
        "balanced": 0.5 * (fb + fr),
        "mean_selected_vs_hold_regret_m3": float(np.mean(regret)),
        "predicted_hold_fraction": predicted_hold,
        "oracle_hold_fraction": oracle_hold,
        "collapse": bool((hold_truth.any() and predicted_hold == 0.0) or (action_truth.any() and predicted_hold == 1.0)),
    }


def _choose_threshold(scores: np.ndarray, returns: np.ndarray) -> tuple[float, dict[str, Any]]:
    unique = np.unique(scores)
    candidates = [float(unique[0] - 1.0e-6), float(unique[-1] + 1.0e-6)]
    candidates.extend(float(0.5 * (left + right)) for left, right in zip(unique[:-1], unique[1:]))
    best: tuple[float, ...] | None = None
    best_threshold = 0.0
    best_metrics: dict[str, Any] | None = None
    for threshold in candidates:
        metrics = _threshold_metrics(scores, returns, threshold)
        key = (
            float(metrics["worst"]),
            float(metrics["balanced"]),
            float(metrics["mean_selected_vs_hold_regret_m3"]),
            abs(float(metrics["predicted_hold_fraction"]) - float(metrics["oracle_hold_fraction"])),
            abs(float(threshold)),
        )
        if best is None or key < best:
            best = key
            best_threshold = float(threshold)
            best_metrics = metrics
    if best_metrics is None:
        raise RuntimeError("V18 could not choose a Train-OOF decision threshold")
    return best_threshold, best_metrics


def _crossfit_boundary(rows: dict[str, Any], seed: int) -> dict[str, Any]:
    returns = rows["returns"]
    hold = (returns >= 0.0).astype(np.float64)
    folds = _stratified_folds(returns, rows["queries"], seed)
    candidates: list[dict[str, Any]] = []
    all_indices = np.arange(returns.size)
    for ridge in RIDGE_GRID:
        scores = np.full(returns.shape, np.nan, dtype=np.float64)
        for test_indices in folds:
            train_indices = np.setdiff1d(all_indices, test_indices)
            state = _fit_preprocessor(rows["dense"][train_indices], rows["explicit"][train_indices])
            z_train = _transform(state, rows["dense"][train_indices], rows["explicit"][train_indices])
            z_test = _transform(state, rows["dense"][test_indices], rows["explicit"][test_indices])
            weight, bias = _fit_logistic(z_train, hold[train_indices], ridge)
            scores[test_indices] = z_test @ weight + bias
        if not bool(np.isfinite(scores).all()):
            raise RuntimeError("V18 OOF boundary scores are non-finite")
        threshold, metrics = _choose_threshold(scores, returns)
        auc = _auc_hold(scores, returns)
        key = (
            float(metrics["worst"]),
            float(metrics["balanced"]),
            float(metrics["mean_selected_vs_hold_regret_m3"]),
            -float(auc),
            -float(ridge),
        )
        candidates.append(
            {
                "ridge": float(ridge),
                "threshold": float(threshold),
                "auc": float(auc),
                "metrics": metrics,
                "selection_key": list(key),
                "score_std": float(np.std(scores)),
                "score_min": float(np.min(scores)),
                "score_max": float(np.max(scores)),
            }
        )
    selected = min(candidates, key=lambda value: tuple(value["selection_key"]))
    return {"selected": selected, "candidates": candidates}


def _torch_preprocessor(state: dict[str, np.ndarray], device: torch.device) -> BoundaryPreprocessorV18:
    return BoundaryPreprocessorV18(
        dense_mean=torch.as_tensor(state["dense_mean"], dtype=torch.float32, device=device),
        dense_std=torch.as_tensor(state["dense_std"], dtype=torch.float32, device=device),
        components=torch.as_tensor(state["components"], dtype=torch.float32, device=device),
        explicit_mean=torch.as_tensor(state["explicit_mean"], dtype=torch.float32, device=device),
        explicit_std=torch.as_tensor(state["explicit_std"], dtype=torch.float32, device=device),
    )


def _evaluate(
    rank_model: torch.nn.Module,
    rank_adapter: QueryConditionedPolicyReturnAdapterV17,
    calibrator: LinearBoundaryCalibratorV18,
    dataset: dict[str, Any],
    *,
    graph: Any,
    normalization: Any,
    mask: np.ndarray,
    device: torch.device,
    scale: float,
) -> dict[str, Any]:
    rows = _collect_boundary_rows(
        rank_model,
        rank_adapter,
        dataset,
        graph=graph,
        normalization=normalization,
        mask=mask,
        device=device,
        scale=scale,
    )
    false_beneficial = false_reject = correct = top1 = 0
    pairwise_correct = pairwise_count = 0
    ranking_regrets: list[float] = []
    decision_regrets: list[float] = []
    scores: list[float] = []
    margins: list[float] = []
    predicted_hold = 0
    selected_source_counts: dict[str, int] = {}
    oracle_source_counts: dict[str, int] = {}

    for index, query in enumerate(rows["queries"].tolist()):
        indices, truth_tensor, raw, context, candidates = _query_inputs(
            rank_model,
            normalization,
            graph,
            dataset,
            query,
            mask=mask,
            scale=scale,
            device=device,
        )
        with torch.no_grad():
            rank_out = rank_adapter(
                raw_rank_scores_m3=raw,
                context_features=context,
                candidate_features=candidates,
            )
            parts = build_boundary_feature_parts_v18(
                context_features=context,
                candidate_features=candidates,
                raw_rank_scores_m3=raw,
                rank_scores_normalized=rank_out.rank_scores_normalized,
                selected_candidate_index=rank_out.selected_candidate_index,
                target_scale_m3=scale,
            )
            out = calibrator.predict(
                parts=parts,
                relative_rank_normalized=rank_out.relative_rank_normalized,
                selected_candidate_index=rank_out.selected_candidate_index,
            )
        truth = truth_tensor.detach().cpu().numpy().astype(float)
        selected = int(rank_out.selected_candidate_index.detach().cpu())
        oracle = int(np.argmin(truth))
        selected_truth = float(truth[selected])
        oracle_best = float(truth[oracle])
        execute = float(out.boundary_distance.detach().cpu()) < 0.0
        oracle_execute = oracle_best < 0.0
        selected_beneficial = selected_truth < 0.0
        realized = selected_truth if execute else 0.0
        oracle_value = min(0.0, oracle_best)
        false_beneficial += int(execute and not selected_beneficial)
        false_reject += int((not execute) and oracle_execute)
        correct += int(
            ((not execute) and (not oracle_execute))
            or (execute and oracle_execute and selected == oracle)
        )
        predicted_hold += int(not execute)
        top1 += int(selected == oracle)
        ranking_regrets.append(max(0.0, selected_truth - oracle_best))
        decision_regrets.append(max(0.0, realized - oracle_value))
        scores.append(float(out.hold_score.detach().cpu()))
        margins.append(float(out.query_best_margin_m3.detach().cpu()))
        selected_source = str(dataset["sources"][indices[selected]])
        oracle_source = str(dataset["sources"][indices[oracle]])
        selected_source_counts[selected_source] = selected_source_counts.get(selected_source, 0) + 1
        oracle_source_counts[oracle_source] = oracle_source_counts.get(oracle_source, 0) + 1
        rank_scores = rank_out.rank_scores_normalized.detach().cpu().numpy().astype(float)
        for left in range(len(rank_scores)):
            for right in range(left + 1, len(rank_scores)):
                if abs(truth[left] - truth[right]) <= 1.0:
                    continue
                pairwise_count += 1
                pairwise_correct += int(
                    np.sign(rank_scores[left] - rank_scores[right])
                    == np.sign(truth[left] - truth[right])
                )

    returns = rows["returns"]
    n = max(1, returns.size)
    fb = false_beneficial / n
    fr = false_reject / n
    score_array = np.asarray(scores, dtype=float)
    margin_array = np.asarray(margins, dtype=float)
    oracle_hold = float(np.mean(returns >= 0.0))
    predicted_hold_fraction = predicted_hold / n
    execute_all = bool(oracle_hold > 0.0 and predicted_hold == 0)
    hold_all = bool(oracle_hold < 1.0 and predicted_hold == n)
    return {
        "query_count": int(n),
        "selected_action_false_beneficial_fraction": fb,
        "selected_action_false_reject_fraction": fr,
        "selected_action_worst_side_error_fraction": max(fb, fr),
        "selected_action_balanced_error_fraction": 0.5 * (fb + fr),
        "hold_aware_decision_accuracy": correct / n,
        "within_query_pairwise_rank_accuracy": pairwise_correct / pairwise_count if pairwise_count else 1.0,
        "within_query_candidate_top1_accuracy": top1 / n,
        "mean_ranking_regret_before_hold_m3": float(np.mean(ranking_regrets)),
        "selected_action_mean_regret_m3": float(np.mean(decision_regrets)),
        "predicted_hold_fraction": predicted_hold_fraction,
        "oracle_hold_optimal_fraction": oracle_hold,
        "execute_all_collapse": execute_all,
        "hold_all_collapse": hold_all,
        "boundary_auc_hold_vs_action": _auc_hold(score_array, returns),
        "boundary_score_std": float(np.std(score_array)),
        "boundary_score_unique_count_1e8": int(np.unique(np.round(score_array, 8)).size),
        "boundary_score": {
            "min": float(np.min(score_array)),
            "q25": float(np.quantile(score_array, 0.25)),
            "median": float(np.median(score_array)),
            "q75": float(np.quantile(score_array, 0.75)),
            "max": float(np.max(score_array)),
        },
        "decision_threshold": float(calibrator.decision_threshold.detach().cpu()),
        "query_margin_selected_return_mae_m3": float(np.mean(np.abs(margin_array - returns))),
        "query_margin_m3": {
            "min": float(np.min(margin_array)),
            "q25": float(np.quantile(margin_array, 0.25)),
            "median": float(np.median(margin_array)),
            "q75": float(np.quantile(margin_array, 0.75)),
            "max": float(np.max(margin_array)),
        },
        "selected_candidate_source_counts": dict(sorted(selected_source_counts.items())),
        "oracle_best_candidate_source_counts": dict(sorted(oracle_source_counts.items())),
    }


def _accepted(metrics: dict[str, Any]) -> bool:
    t = DEVELOPMENT_ACCEPTANCE
    return bool(
        metrics["selected_action_false_beneficial_fraction"]
        <= t["selected_action_false_beneficial_fraction_max"]
        and metrics["selected_action_false_reject_fraction"]
        <= t["selected_action_false_reject_fraction_max"]
        and metrics["hold_aware_decision_accuracy"] >= t["hold_aware_decision_accuracy_min"]
        and metrics["within_query_pairwise_rank_accuracy"] >= t["within_query_pairwise_rank_accuracy_min"]
        and metrics["within_query_candidate_top1_accuracy"] >= t["within_query_candidate_top1_accuracy_min"]
        and not metrics["execute_all_collapse"]
        and not metrics["hold_all_collapse"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-step2", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--supervisory-control", required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--calibration-dataset", required=True)
    parser.add_argument("--rank-source-checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    train_meta = _development_metadata(args.train_dataset, "train")
    validation_meta = _development_metadata(args.validation_dataset, "validation")
    calibration_meta = _development_metadata(args.calibration_dataset, "calibration")
    if train_meta["groups"] & validation_meta["groups"]:
        raise ValueError("V18 Train/Validation rainfall groups overlap")
    if train_meta["groups"] & calibration_meta["groups"]:
        raise ValueError("V18 Train/Calibration rainfall groups overlap")
    if validation_meta["groups"] & calibration_meta["groups"]:
        raise ValueError("V18 Validation/Calibration rainfall groups overlap")

    train = _load_dataset(args.train_dataset, role="policy_return_train")
    validation = _load_dataset(args.validation_dataset, role="policy_return_validation")
    graph = _load_graph(args.graph)
    control, mask = load_native_supervisory_control(args.supervisory_control, actuator_ids=graph.actuator_ids)
    if str(control["supervisory_mask_sha256"]).lower() != train_meta["supervisory_mask_sha256"]:
        raise ValueError("V18 supervisory-control artifact differs from truth lineage")

    rank_model, normalization, _base = load_direct_tfv_runtime_checkpoint(
        args.base_step2, graph=graph, device=device
    )
    for parameter in rank_model.parameters():
        parameter.requires_grad_(False)
    rank_model.eval()
    step2_before = _model_state_sha256(rank_model)
    scale = float(rank_model.target_scale_m3.detach().cpu())

    rank_source = _rank_source_payload(
        args.rank_source_checkpoint,
        base_step2_sha256=sha256_file(args.base_step2),
        train_sha256=train_meta["sha256"],
        validation_sha256=validation_meta["sha256"],
    )
    rank_adapter = _make_rank_adapter(
        rank_model=rank_model,
        normalization=normalization,
        graph=graph,
        train=train,
        rank_source=rank_source,
        mask=mask,
        scale=scale,
        device=device,
    )
    rank_sha_before = rank_state_sha256(rank_adapter)

    train_rows = _collect_boundary_rows(
        rank_model,
        rank_adapter,
        train,
        graph=graph,
        normalization=normalization,
        mask=mask,
        device=device,
        scale=scale,
    )
    oof = _crossfit_boundary(train_rows, args.seed)
    selected_oof = oof["selected"]
    train_oof_supported = bool(
        float(selected_oof["auc"]) >= TRAIN_OOF_AUC_MIN
        and not bool(selected_oof["metrics"]["collapse"])
        and float(selected_oof["score_std"]) > 1.0e-6
    )

    state = _fit_preprocessor(train_rows["dense"], train_rows["explicit"])
    z_train = _transform(state, train_rows["dense"], train_rows["explicit"])
    hold_train = (train_rows["returns"] >= 0.0).astype(np.float64)
    ridge = float(selected_oof["ridge"])
    boundary_weight, boundary_bias = _fit_logistic(z_train, hold_train, ridge)
    magnitude_weight, magnitude_bias = _fit_magnitude(
        z_train, train_rows["returns"], scale, ridge
    )
    preprocessor = _torch_preprocessor(state, device)
    calibrator = LinearBoundaryCalibratorV18(
        preprocessor=preprocessor,
        boundary_weight=torch.as_tensor(boundary_weight, dtype=torch.float32, device=device),
        boundary_bias=boundary_bias,
        decision_threshold=float(selected_oof["threshold"]),
        magnitude_weight=torch.as_tensor(magnitude_weight, dtype=torch.float32, device=device),
        magnitude_bias=magnitude_bias,
        target_scale_m3=scale,
    ).to(device)
    calibrator.eval()

    validation_metrics = _evaluate(
        rank_model,
        rank_adapter,
        calibrator,
        validation,
        graph=graph,
        normalization=normalization,
        mask=mask,
        device=device,
        scale=scale,
    )
    validation_pass = bool(train_oof_supported and _accepted(validation_metrics))

    calibration_truth_loaded = False
    calibration_metrics: dict[str, Any] | None = None
    if validation_pass:
        calibration = _load_dataset(args.calibration_dataset, role="policy_return_calibration")
        calibration_truth_loaded = True
        calibration_metrics = _evaluate(
            rank_model,
            rank_adapter,
            calibrator,
            calibration,
            graph=graph,
            normalization=normalization,
            mask=mask,
            device=device,
            scale=scale,
        )

    rank_sha_after = rank_state_sha256(rank_adapter)
    if rank_sha_before != rank_sha_after:
        raise RuntimeError("V18 fitting mutated the frozen V15 rank branch")
    step2_after = _model_state_sha256(rank_model)
    if step2_before != step2_after:
        raise RuntimeError("V18 fitting mutated frozen Step2")

    dense_varying = float(np.mean(np.std(train_rows["dense"], axis=0) > 1.0e-6))
    explicit_varying = float(np.mean(np.std(train_rows["explicit"], axis=0) > 1.0e-6))
    payload = {
        "contract": DIRECT_TFV_QUERY_MARGIN_V18_CHECKPOINT_CONTRACT,
        "training_contract": STEP3_V18_TRAINING_CONTRACT,
        "query_margin_contract": DIRECT_TFV_QUERY_MARGIN_V18_CONTRACT,
        "development_only": True,
        "estimand": DIRECT_TFV_POLICY_RETURN_ESTIMAND,
        "action_encoding_contract": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
        "base_step2_sha256": sha256_file(args.base_step2),
        "base_step2_frozen": True,
        "base_step2_state_sha256_before_training": step2_before,
        "base_step2_state_sha256_after_training": step2_after,
        "step2_parameters_updated": False,
        "graph_sha256": sha256_file(args.graph),
        "supervisory_control_sha256": sha256_file(args.supervisory_control),
        "supervisory_mask_sha256": train_meta["supervisory_mask_sha256"],
        "supervisory_control_dimension": 82,
        "model_action_channel_count": 109,
        "split_contract": STEP3_DEVELOPMENT_BANK_CONTRACT,
        "train_dataset_sha256": train_meta["sha256"],
        "validation_dataset_sha256": validation_meta["sha256"],
        "calibration_dataset_sha256": calibration_meta["sha256"],
        "rank_source_checkpoint_sha256": sha256_file(args.rank_source_checkpoint),
        "rank_source_checkpoint_contract": LEGACY_V15_CHECKPOINT_CONTRACT,
        "rank_reused_from_v15": True,
        "rank_retrained_in_v18": False,
        "rank_branch_state_sha256_before_boundary": rank_sha_before,
        "rank_branch_state_sha256_after_boundary": rank_sha_after,
        "train_group_count": len(train_meta["groups"]),
        "validation_group_count": len(validation_meta["groups"]),
        "calibration_group_count": len(calibration_meta["groups"]),
        "boundary_model": "TRAIN_ONLY_STANDARDIZED_PCA_L2_LINEAR_LOGISTIC",
        "boundary_pca_components": int(state["components"].shape[0]),
        "boundary_dense_feature_varying_fraction": dense_varying,
        "boundary_explicit_feature_varying_fraction": explicit_varying,
        "boundary_cv_folds": CV_FOLDS,
        "boundary_ridge_grid": list(RIDGE_GRID),
        "boundary_oof_selection": oof,
        "train_oof_boundary_supported": train_oof_supported,
        "train_oof_auc_min": TRAIN_OOF_AUC_MIN,
        "decision_threshold_selected_from_train_oof_only": True,
        "validation_used_for_boundary_threshold_selection": False,
        "no_forced_action_or_hold_quota": True,
        "boundary_preprocessor_state": {
            key: value.astype(np.float32) for key, value in state.items()
        },
        "boundary_weight": boundary_weight.astype(np.float32),
        "boundary_bias": float(boundary_bias),
        "decision_threshold": float(selected_oof["threshold"]),
        "magnitude_weight": magnitude_weight.astype(np.float32),
        "magnitude_bias": float(magnitude_bias),
        "validation_metrics": validation_metrics,
        "development_acceptance": DEVELOPMENT_ACCEPTANCE,
        "development_validation_pass": validation_pass,
        "calibration_truth_loaded": calibration_truth_loaded,
        "calibration_raw_metrics": calibration_metrics,
        "online_swmm_called": False,
        "ready_for_pi1_development": False,
        "ready_for_policy_lock": False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    report = copy.deepcopy(payload)
    for key in (
        "boundary_preprocessor_state",
        "boundary_weight",
        "magnitude_weight",
    ):
        report.pop(key, None)
    report["checkpoint_sha256"] = sha256_file(out)
    out.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))

    if not validation_pass:
        raise RuntimeError(
            "V18 Step3 boundary failed Train-OOF/Validation development gates; Calibration stayed sealed"
        )


if __name__ == "__main__":
    main()
