"""Train Step3 V19 zero-anchored exact policy-return advantage on the existing Development bank.

This is a structural Step3 replacement, not another boundary-capacity tweak. The validated V15 rank
is imported and frozen. Every existing Train candidate record is used to learn the exact
candidate-minus-HOLD return, while OOF folds remain query-group disjoint so sibling candidates from
one hydraulic query can never leak across train/test folds.

The advantage model has no intercept and receives only action-to-HOLD effects plus low-rank
context×action interactions. Therefore candidate==HOLD/reference maps exactly to predicted return 0.
Validation is one-shot; Calibration truth stays sealed unless Validation passes. No SWMM is called.
"""
from __future__ import annotations

import argparse
from collections import Counter
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
from rtc.direct_tfv_policy_return_advantage_v19 import (
    ACTION_COMPONENTS,
    CONTEXT_COMPONENTS,
    DIRECT_TFV_ADVANTAGE_V19_CHECKPOINT_CONTRACT,
    DIRECT_TFV_ADVANTAGE_V19_CONTRACT,
    DIRECT_TFV_ADVANTAGE_V19_FEATURE_CONTRACT,
    EXPLICIT_INTERACTION_DIM,
    ZeroAnchoredAdvantagePartsV19,
    ZeroAnchoredAdvantagePreprocessorV19,
    ZeroAnchoredAdvantageRegressorV19,
    build_zero_anchored_advantage_parts_v19,
)
from rtc.direct_tfv_policy_return_query_margin_v17 import rank_state_sha256
from rtc.native_supervisory_control import load_native_supervisory_control
from rtc.production_cli import _load_graph
from build_step3_development_bank_current import STEP3_DEVELOPMENT_BANK_CONTRACT
from train_direct_tfv_policy_return_query_margin_current import _load_dataset, _rank_scores
from train_step3_query_margin_v2_current import (
    _development_metadata,
    _features,
    _model_state_sha256,
)
from train_step3_query_margin_v18_current import (
    _auc_hold,
    _make_rank_adapter,
    _rank_source_payload,
)


STEP3_V19_TRAINING_CONTRACT = (
    "PROJECT7_STEP3_V19_FROZEN_V15_RANK_ZERO_ANCHORED_ALL_RECORD_ADVANTAGE"
)
CV_FOLDS = 6
RIDGE_GRID = (0.01, 0.1, 1.0)
TRAIN_OOF_AUC_MIN = 0.60
ZERO_BOUNDARY = 0.0

DEVELOPMENT_ACCEPTANCE = {
    "selected_action_false_beneficial_fraction_max": 0.25,
    "selected_action_false_reject_fraction_max": 0.25,
    "hold_aware_decision_accuracy_min": 0.60,
    "within_query_pairwise_rank_accuracy_min": 0.60,
    "within_query_candidate_top1_accuracy_min": 0.55,
    "execute_all_collapse_forbidden_when_oracle_hold_exists": True,
    "hold_all_collapse_forbidden_when_oracle_action_exists": True,
}


def _collect_advantage_rows(
    rank_model: torch.nn.Module,
    rank_adapter: torch.nn.Module,
    dataset: dict[str, Any],
    *,
    graph: Any,
    normalization: Any,
    mask: np.ndarray,
    device: torch.device,
    scale: float,
) -> dict[str, Any]:
    contexts: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    explicit: list[np.ndarray] = []
    returns: list[float] = []
    queries: list[str] = []
    sources: list[str] = []
    selected_mask: list[bool] = []
    rank_scores: list[float] = []

    with torch.no_grad():
        for query in sorted(set(dataset["queries"].tolist())):
            indices = np.flatnonzero(dataset["queries"] == query)
            truth = torch.as_tensor(
                dataset["true_policy_return_delta_tfv_m3"][indices],
                dtype=torch.float32,
                device=device,
            ).reshape(-1)
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
            rank_out = rank_adapter(
                raw_rank_scores_m3=raw,
                context_features=context,
                candidate_features=candidates,
            )
            selected = int(rank_out.selected_candidate_index.detach().cpu())
            for local, row_index in enumerate(indices.tolist()):
                parts = build_zero_anchored_advantage_parts_v19(
                    context_features=context,
                    candidate_features=candidates[local],
                    raw_step2_score_m3=raw[local],
                    active_target=torch.as_tensor(
                        dataset["active_target"][row_index],
                        dtype=torch.float32,
                        device=device,
                    ),
                    candidate_target=torch.as_tensor(
                        dataset["candidate_target"][row_index],
                        dtype=torch.float32,
                        device=device,
                    ),
                    candidate_source=str(dataset["sources"][row_index]),
                    supervisory_mask=mask,
                    target_scale_m3=scale,
                )
                contexts.append(parts.context.detach().cpu().numpy().astype(np.float64))
                actions.append(parts.action_dense.detach().cpu().numpy().astype(np.float64))
                explicit.append(parts.explicit.detach().cpu().numpy().astype(np.float64))
                returns.append(float(truth[local].detach().cpu()))
                queries.append(query)
                sources.append(str(dataset["sources"][row_index]))
                selected_mask.append(local == selected)
                rank_scores.append(float(rank_out.rank_scores_normalized[local].detach().cpu()))

    result = {
        "context": np.stack(contexts),
        "action": np.stack(actions),
        "explicit": np.stack(explicit),
        "returns": np.asarray(returns, dtype=np.float64),
        "queries": np.asarray(queries),
        "sources": np.asarray(sources),
        "selected_mask": np.asarray(selected_mask, dtype=bool),
        "rank_scores": np.asarray(rank_scores, dtype=np.float64),
    }
    query_count = len(set(result["queries"].tolist()))
    if int(result["selected_mask"].sum()) != query_count:
        raise RuntimeError("V19 must have exactly one frozen-rank selected candidate per query")
    return result


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = values.std(axis=0)
    return np.where(std > 1.0e-6, std, 1.0)


def _rms_scale(values: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(np.square(values), axis=0))
    return np.where(rms > 1.0e-6, rms, 1.0)


def _fit_preprocessor(rows: dict[str, Any]) -> dict[str, np.ndarray]:
    context_mean = rows["context"].mean(axis=0)
    context_std = _safe_std(rows["context"])
    context_z = (rows["context"] - context_mean) / context_std
    _u, _s, context_vh = np.linalg.svd(context_z, full_matrices=False)
    context_k = min(
        CONTEXT_COMPONENTS,
        max(1, rows["context"].shape[0] - 1),
        rows["context"].shape[1],
        context_vh.shape[0],
    )
    context_components = context_vh[:context_k]

    # No centering is permitted for action blocks: zero action must remain exactly zero.
    action_scale = _rms_scale(rows["action"])
    action_z = rows["action"] / action_scale
    _u, _s, action_vh = np.linalg.svd(action_z, full_matrices=False)
    action_k = min(
        ACTION_COMPONENTS,
        max(1, rows["action"].shape[0] - 1),
        rows["action"].shape[1],
        action_vh.shape[0],
    )
    action_components = action_vh[:action_k]
    explicit_scale = _rms_scale(rows["explicit"])
    return {
        "context_mean": context_mean,
        "context_std": context_std,
        "context_components": context_components,
        "action_scale": action_scale,
        "action_components": action_components,
        "explicit_scale": explicit_scale,
    }


def _transform(state: dict[str, np.ndarray], rows: dict[str, Any]) -> np.ndarray:
    context_z = (rows["context"] - state["context_mean"]) / state["context_std"]
    context_low = context_z @ state["context_components"].T
    action_z = rows["action"] / state["action_scale"]
    action_low = action_z @ state["action_components"].T
    explicit_z = rows["explicit"] / state["explicit_scale"]
    context_action = np.einsum("ni,nj->nij", context_low, action_low).reshape(
        rows["returns"].size,
        -1,
    )
    context_explicit = np.einsum(
        "ni,nj->nij",
        context_low,
        explicit_z[:, :EXPLICIT_INTERACTION_DIM],
    ).reshape(rows["returns"].size, -1)
    design = np.concatenate(
        (
            action_low,
            explicit_z,
            context_action,
            context_explicit,
        ),
        axis=1,
    )
    if not bool(np.isfinite(design).all()):
        raise RuntimeError("V19 transformed design contains non-finite values")
    return design


def _sample_weights(returns: np.ndarray, queries: np.ndarray) -> np.ndarray:
    counts = Counter(str(value) for value in queries.tolist())
    base = np.asarray([1.0 / counts[str(value)] for value in queries.tolist()], dtype=np.float64)
    hold = returns >= 0.0
    action = ~hold
    if not bool(hold.any()) or not bool(action.any()):
        raise ValueError("V19 Train rows require both ACTION and HOLD exact-return signs")
    hold_mass = float(base[hold].sum())
    action_mass = float(base[action].sum())
    base[hold] *= 0.5 / hold_mass
    base[action] *= 0.5 / action_mass
    base *= returns.size / float(base.sum())
    return base


def _fit_ridge(
    design: np.ndarray,
    returns: np.ndarray,
    queries: np.ndarray,
    *,
    scale: float,
    ridge: float,
) -> np.ndarray:
    target = np.arcsinh(returns / float(scale))
    weights = _sample_weights(returns, queries)
    sqrt_w = np.sqrt(weights)[:, None]
    xw = design * sqrt_w
    yw = target * sqrt_w[:, 0]
    reg = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    return np.linalg.solve(xw.T @ xw + reg, xw.T @ yw)


def _selected_arrays(
    rows: dict[str, Any],
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = rows["selected_mask"]
    return coordinates[mask], rows["returns"][mask], rows["queries"][mask]


def _decision_metrics(coordinates: np.ndarray, returns: np.ndarray) -> dict[str, Any]:
    execute = coordinates < ZERO_BOUNDARY
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
        "collapse": bool(
            (hold_truth.any() and predicted_hold == 0.0)
            or (action_truth.any() and predicted_hold == 1.0)
        ),
    }


def _query_folds(rows: dict[str, Any], seed: int) -> list[np.ndarray]:
    selected_returns = {
        str(query): float(value)
        for query, value in zip(
            rows["queries"][rows["selected_mask"]].tolist(),
            rows["returns"][rows["selected_mask"]].tolist(),
        )
    }
    hold = [query for query, value in selected_returns.items() if value >= 0.0]
    action = [query for query, value in selected_returns.items() if value < 0.0]
    if min(len(hold), len(action)) < CV_FOLDS:
        raise ValueError("V19 Train needs at least one selected HOLD/ACTION query per OOF fold")
    folds: list[list[str]] = [[] for _ in range(CV_FOLDS)]
    for label, values in (("hold", hold), ("action", action)):
        ordered = sorted(
            values,
            key=lambda query: hashlib.sha256(
                f"v19|{seed}|{label}|{query}".encode("utf-8")
            ).hexdigest(),
        )
        for position, query in enumerate(ordered):
            folds[position % CV_FOLDS].append(query)
    return [np.asarray(sorted(values)) for values in folds]


def _subset(rows: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    return {
        key: value[mask]
        for key, value in rows.items()
        if isinstance(value, np.ndarray) and int(value.shape[0]) == int(mask.size)
    }


def _crossfit(rows: dict[str, Any], *, scale: float, seed: int) -> dict[str, Any]:
    folds = _query_folds(rows, seed)
    candidates: list[dict[str, Any]] = []
    for ridge in RIDGE_GRID:
        coordinates = np.full(rows["returns"].shape, np.nan, dtype=np.float64)
        for held_queries in folds:
            is_test = np.isin(rows["queries"], held_queries)
            is_train = ~is_test
            if set(rows["queries"][is_test].tolist()) & set(rows["queries"][is_train].tolist()):
                raise RuntimeError("V19 OOF leaked sibling candidates from one query")
            train_rows = _subset(rows, is_train)
            test_rows = _subset(rows, is_test)
            state = _fit_preprocessor(train_rows)
            z_train = _transform(state, train_rows)
            z_test = _transform(state, test_rows)
            weight = _fit_ridge(
                z_train,
                train_rows["returns"],
                train_rows["queries"],
                scale=scale,
                ridge=float(ridge),
            )
            coordinates[is_test] = z_test @ weight
        if not bool(np.isfinite(coordinates).all()):
            raise RuntimeError("V19 OOF coordinates are non-finite")
        selected_coordinates, selected_returns, _selected_queries = _selected_arrays(
            rows,
            coordinates,
        )
        metrics = _decision_metrics(selected_coordinates, selected_returns)
        auc = _auc_hold(selected_coordinates, selected_returns)
        predicted_all = np.sinh(np.clip(coordinates, -6.0, 6.0)) * float(scale)
        key = (
            float(metrics["worst"]),
            float(metrics["balanced"]),
            float(metrics["mean_selected_vs_hold_regret_m3"]),
            -float(auc),
            float(ridge),
        )
        candidates.append(
            {
                "ridge": float(ridge),
                "auc": float(auc),
                "score_std": float(np.std(selected_coordinates)),
                "score_min": float(np.min(selected_coordinates)),
                "score_max": float(np.max(selected_coordinates)),
                "selected_metrics": metrics,
                "all_record_return_mae_m3": float(
                    np.mean(np.abs(predicted_all - rows["returns"]))
                ),
                "selection_key": list(key),
            }
        )
    selected = min(candidates, key=lambda value: tuple(value["selection_key"]))
    return {
        "selected": selected,
        "candidates": candidates,
        "fold_query_counts": [int(len(values)) for values in folds],
        "query_group_disjoint": True,
    }


def _torch_preprocessor(
    state: dict[str, np.ndarray],
    device: torch.device,
) -> ZeroAnchoredAdvantagePreprocessorV19:
    return ZeroAnchoredAdvantagePreprocessorV19(
        context_mean=torch.as_tensor(state["context_mean"], dtype=torch.float32, device=device),
        context_std=torch.as_tensor(state["context_std"], dtype=torch.float32, device=device),
        context_components=torch.as_tensor(
            state["context_components"],
            dtype=torch.float32,
            device=device,
        ),
        action_scale=torch.as_tensor(state["action_scale"], dtype=torch.float32, device=device),
        action_components=torch.as_tensor(
            state["action_components"],
            dtype=torch.float32,
            device=device,
        ),
        explicit_scale=torch.as_tensor(state["explicit_scale"], dtype=torch.float32, device=device),
    )


def _predict_rows(
    regressor: ZeroAnchoredAdvantageRegressorV19,
    rows: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates: list[float] = []
    advantages: list[float] = []
    with torch.no_grad():
        for context, action, explicit in zip(
            rows["context"],
            rows["action"],
            rows["explicit"],
        ):
            parts = ZeroAnchoredAdvantagePartsV19(
                context=torch.as_tensor(context, dtype=torch.float32, device=device),
                action_dense=torch.as_tensor(action, dtype=torch.float32, device=device),
                explicit=torch.as_tensor(explicit, dtype=torch.float32, device=device),
            )
            out = regressor.predict(parts)
            coordinates.append(float(out.coordinate.detach().cpu()))
            advantages.append(float(out.advantage_m3.detach().cpu()))
    return np.asarray(coordinates), np.asarray(advantages)


def _evaluate(
    rows: dict[str, Any],
    regressor: ZeroAnchoredAdvantageRegressorV19,
) -> dict[str, Any]:
    device = regressor.weight.device
    coordinates, advantages = _predict_rows(regressor, rows, device=device)
    queries = sorted(set(rows["queries"].tolist()))
    false_beneficial = false_reject = correct = top1 = predicted_hold = 0
    pairwise_correct = pairwise_count = 0
    ranking_regrets: list[float] = []
    decision_regrets: list[float] = []
    selected_coordinates: list[float] = []
    selected_advantages: list[float] = []
    selected_truths: list[float] = []
    selected_sources: Counter[str] = Counter()
    oracle_sources: Counter[str] = Counter()

    for query in queries:
        idx = np.flatnonzero(rows["queries"] == query)
        local_selected = np.flatnonzero(rows["selected_mask"][idx])
        if local_selected.size != 1:
            raise RuntimeError("V19 evaluation lost one-selected-candidate-per-query invariant")
        selected = int(local_selected[0])
        truth = rows["returns"][idx]
        rank = rows["rank_scores"][idx]
        oracle = int(np.argmin(truth))
        selected_truth = float(truth[selected])
        oracle_best = float(truth[oracle])
        coordinate = float(coordinates[idx[selected]])
        predicted_advantage = float(advantages[idx[selected]])
        execute = coordinate < ZERO_BOUNDARY
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
        selected_coordinates.append(coordinate)
        selected_advantages.append(predicted_advantage)
        selected_truths.append(selected_truth)
        selected_sources[str(rows["sources"][idx[selected]])] += 1
        oracle_sources[str(rows["sources"][idx[oracle]])] += 1

        for left in range(len(rank)):
            for right in range(left + 1, len(rank)):
                if abs(float(truth[left]) - float(truth[right])) <= 1.0:
                    continue
                pairwise_count += 1
                pairwise_correct += int(
                    np.sign(rank[left] - rank[right])
                    == np.sign(truth[left] - truth[right])
                )

    selected_coordinate_array = np.asarray(selected_coordinates, dtype=np.float64)
    selected_truth_array = np.asarray(selected_truths, dtype=np.float64)
    selected_advantage_array = np.asarray(selected_advantages, dtype=np.float64)
    n = max(1, len(queries))
    fb = false_beneficial / n
    fr = false_reject / n
    oracle_hold = float(np.mean(selected_truth_array >= 0.0))
    predicted_hold_fraction = predicted_hold / n
    all_sign = np.mean((advantages < 0.0) == (rows["returns"] < 0.0))
    return {
        "query_count": int(n),
        "candidate_record_count": int(rows["returns"].size),
        "selected_action_false_beneficial_fraction": fb,
        "selected_action_false_reject_fraction": fr,
        "selected_action_worst_side_error_fraction": max(fb, fr),
        "selected_action_balanced_error_fraction": 0.5 * (fb + fr),
        "hold_aware_decision_accuracy": correct / n,
        "within_query_pairwise_rank_accuracy": (
            pairwise_correct / pairwise_count if pairwise_count else 1.0
        ),
        "within_query_candidate_top1_accuracy": top1 / n,
        "mean_ranking_regret_before_hold_m3": float(np.mean(ranking_regrets)),
        "selected_action_mean_regret_m3": float(np.mean(decision_regrets)),
        "predicted_hold_fraction": predicted_hold_fraction,
        "oracle_hold_optimal_fraction": oracle_hold,
        "execute_all_collapse": bool(oracle_hold > 0.0 and predicted_hold == 0),
        "hold_all_collapse": bool(oracle_hold < 1.0 and predicted_hold == n),
        "boundary_auc_hold_vs_action": _auc_hold(
            selected_coordinate_array,
            selected_truth_array,
        ),
        "boundary_score_std": float(np.std(selected_coordinate_array)),
        "boundary_score_unique_count_1e8": int(
            np.unique(np.round(selected_coordinate_array, 8)).size
        ),
        "boundary_score": {
            "min": float(np.min(selected_coordinate_array)),
            "q25": float(np.quantile(selected_coordinate_array, 0.25)),
            "median": float(np.median(selected_coordinate_array)),
            "q75": float(np.quantile(selected_coordinate_array, 0.75)),
            "max": float(np.max(selected_coordinate_array)),
        },
        "decision_threshold": ZERO_BOUNDARY,
        "selected_return_mae_m3": float(
            np.mean(np.abs(selected_advantage_array - selected_truth_array))
        ),
        "all_candidate_return_mae_m3": float(np.mean(np.abs(advantages - rows["returns"]))),
        "all_candidate_sign_accuracy": float(all_sign),
        "selected_candidate_source_counts": dict(sorted(selected_sources.items())),
        "oracle_best_candidate_source_counts": dict(sorted(oracle_sources.items())),
    }


def _accepted(metrics: dict[str, Any]) -> bool:
    threshold = DEVELOPMENT_ACCEPTANCE
    return bool(
        metrics["selected_action_false_beneficial_fraction"]
        <= threshold["selected_action_false_beneficial_fraction_max"]
        and metrics["selected_action_false_reject_fraction"]
        <= threshold["selected_action_false_reject_fraction_max"]
        and metrics["hold_aware_decision_accuracy"]
        >= threshold["hold_aware_decision_accuracy_min"]
        and metrics["within_query_pairwise_rank_accuracy"]
        >= threshold["within_query_pairwise_rank_accuracy_min"]
        and metrics["within_query_candidate_top1_accuracy"]
        >= threshold["within_query_candidate_top1_accuracy_min"]
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
    device = torch.device(
        args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )

    train_meta = _development_metadata(args.train_dataset, "train")
    validation_meta = _development_metadata(args.validation_dataset, "validation")
    calibration_meta = _development_metadata(args.calibration_dataset, "calibration")
    if train_meta["groups"] & validation_meta["groups"]:
        raise ValueError("V19 Train/Validation rainfall groups overlap")
    if train_meta["groups"] & calibration_meta["groups"]:
        raise ValueError("V19 Train/Calibration rainfall groups overlap")
    if validation_meta["groups"] & calibration_meta["groups"]:
        raise ValueError("V19 Validation/Calibration rainfall groups overlap")
    continuations = {
        train_meta["continuation_policy_sha256"],
        validation_meta["continuation_policy_sha256"],
        calibration_meta["continuation_policy_sha256"],
    }
    if len(continuations) != 1:
        raise ValueError("V19 development splits do not share one frozen continuation policy")

    train = _load_dataset(args.train_dataset, role="policy_return_train")
    validation = _load_dataset(args.validation_dataset, role="policy_return_validation")
    graph = _load_graph(args.graph)
    control, mask = load_native_supervisory_control(
        args.supervisory_control,
        actuator_ids=graph.actuator_ids,
    )
    if str(control["supervisory_mask_sha256"]).lower() != train_meta["supervisory_mask_sha256"]:
        raise ValueError("V19 supervisory-control artifact differs from truth lineage")

    rank_model, normalization, _base = load_direct_tfv_runtime_checkpoint(
        args.base_step2,
        graph=graph,
        device=device,
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

    train_rows = _collect_advantage_rows(
        rank_model,
        rank_adapter,
        train,
        graph=graph,
        normalization=normalization,
        mask=mask,
        device=device,
        scale=scale,
    )
    if int(train_rows["returns"].size) <= int(train_rows["selected_mask"].sum()):
        raise RuntimeError("V19 must train on all existing candidate records, not selected-only rows")

    oof = _crossfit(train_rows, scale=scale, seed=args.seed)
    selected_oof = oof["selected"]
    train_oof_supported = bool(
        float(selected_oof["auc"]) >= TRAIN_OOF_AUC_MIN
        and not bool(selected_oof["selected_metrics"]["collapse"])
        and float(selected_oof["score_std"]) > 1.0e-6
    )

    preprocessor_state = _fit_preprocessor(train_rows)
    z_train = _transform(preprocessor_state, train_rows)
    ridge = float(selected_oof["ridge"])
    advantage_weight = _fit_ridge(
        z_train,
        train_rows["returns"],
        train_rows["queries"],
        scale=scale,
        ridge=ridge,
    )
    preprocessor = _torch_preprocessor(preprocessor_state, device)
    regressor = ZeroAnchoredAdvantageRegressorV19(
        preprocessor=preprocessor,
        weight=torch.as_tensor(advantage_weight, dtype=torch.float32, device=device),
        target_scale_m3=scale,
    ).to(device)
    regressor.eval()

    validation_rows = _collect_advantage_rows(
        rank_model,
        rank_adapter,
        validation,
        graph=graph,
        normalization=normalization,
        mask=mask,
        device=device,
        scale=scale,
    )
    validation_metrics = _evaluate(validation_rows, regressor)
    validation_pass = bool(train_oof_supported and _accepted(validation_metrics))

    calibration_truth_loaded = False
    calibration_metrics: dict[str, Any] | None = None
    if validation_pass:
        calibration = _load_dataset(
            args.calibration_dataset,
            role="policy_return_calibration",
        )
        calibration_truth_loaded = True
        calibration_rows = _collect_advantage_rows(
            rank_model,
            rank_adapter,
            calibration,
            graph=graph,
            normalization=normalization,
            mask=mask,
            device=device,
            scale=scale,
        )
        calibration_metrics = _evaluate(calibration_rows, regressor)

    rank_sha_after = rank_state_sha256(rank_adapter)
    if rank_sha_before != rank_sha_after:
        raise RuntimeError("V19 fitting mutated the frozen V15 rank branch")
    step2_after = _model_state_sha256(rank_model)
    if step2_before != step2_after:
        raise RuntimeError("V19 fitting mutated frozen Step2")

    rank_source_state = rank_source.get("query_margin_state_dict")
    if not isinstance(rank_source_state, dict):
        raise ValueError("V19 V15 rank source lacks query_margin_state_dict")
    frozen_rank_state = {
        key: value.detach().cpu()
        for key, value in rank_source_state.items()
        if key.startswith("context_encoder.")
        or key.startswith("candidate_encoder.")
        or key.startswith("rank_adjustment.")
    }

    payload = {
        "contract": DIRECT_TFV_ADVANTAGE_V19_CHECKPOINT_CONTRACT,
        "training_contract": STEP3_V19_TRAINING_CONTRACT,
        "advantage_contract": DIRECT_TFV_ADVANTAGE_V19_CONTRACT,
        "feature_contract": DIRECT_TFV_ADVANTAGE_V19_FEATURE_CONTRACT,
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
        "continuation_policy_sha256": train_meta["continuation_policy_sha256"],
        "rank_source_checkpoint_sha256": sha256_file(args.rank_source_checkpoint),
        "rank_source_checkpoint_contract": rank_source.get("contract"),
        "rank_source_state_dict": frozen_rank_state,
        "rank_reused_from_v15": True,
        "rank_retrained_in_v19": False,
        "rank_branch_state_sha256_before_advantage": rank_sha_before,
        "rank_branch_state_sha256_after_advantage": rank_sha_after,
        "context_dim": int(rank_adapter.context_dim),
        "candidate_dim": int(rank_adapter.candidate_dim),
        "target_scale_m3": scale,
        "train_group_count": len(train_meta["groups"]),
        "validation_group_count": len(validation_meta["groups"]),
        "calibration_group_count": len(calibration_meta["groups"]),
        "train_candidate_record_count": int(train_rows["returns"].size),
        "train_selected_query_count": int(train_rows["selected_mask"].sum()),
        "uses_all_existing_candidate_records": True,
        "oof_query_group_disjoint": bool(oof["query_group_disjoint"]),
        "zero_anchor_identity": "candidate_equals_hold_implies_predicted_advantage_equals_zero",
        "numeric_boundary_m3": 0.0,
        "decision_threshold_coordinate": ZERO_BOUNDARY,
        "decision_threshold_tuned": False,
        "advantage_model": "NO_INTERCEPT_WEIGHTED_RIDGE_ASINH_LOW_RANK_CONTEXT_ACTION",
        "context_components": int(preprocessor_state["context_components"].shape[0]),
        "action_components": int(preprocessor_state["action_components"].shape[0]),
        "ridge_grid": list(RIDGE_GRID),
        "boundary_cv_folds": CV_FOLDS,
        "boundary_oof_selection": oof,
        "train_oof_boundary_supported": train_oof_supported,
        "train_oof_auc_min": TRAIN_OOF_AUC_MIN,
        "validation_used_for_model_selection": False,
        "no_forced_action_or_hold_quota": True,
        "advantage_preprocessor_state": {
            key: value.astype(np.float32) for key, value in preprocessor_state.items()
        },
        "advantage_weight": advantage_weight.astype(np.float32),
        "validation_metrics": validation_metrics,
        "development_acceptance": DEVELOPMENT_ACCEPTANCE,
        "development_validation_pass": validation_pass,
        "calibration_truth_loaded": calibration_truth_loaded,
        "calibration_raw_metrics": calibration_metrics,
        "online_swmm_called": False,
        "new_swmm_runs": 0,
        "ready_for_pi1_development": False,
        "ready_for_policy_lock": False,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    report = copy.deepcopy(payload)
    for key in (
        "advantage_preprocessor_state",
        "advantage_weight",
        "rank_source_state_dict",
    ):
        report.pop(key, None)
    report["checkpoint_sha256"] = sha256_file(out)
    out.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))

    if not train_oof_supported:
        raise RuntimeError(
            "V19 zero-anchored advantage lacks Train-OOF selected-action sign support"
        )
    if not validation_pass:
        raise RuntimeError(
            "V19 zero-anchored advantage failed one-shot Validation; Calibration stayed sealed"
        )


if __name__ == "__main__":
    main()
