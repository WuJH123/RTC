"""Train the accuracy-first Step3 critic on the existing 8:1:1 development bank.

This is a zero-SWMM DEVELOPMENT workflow. Step2 stays frozen. V16 trains Step3 in two stages:
(1) learn only the candidate-rank adapter on frozen Step2 scores/latents; (2) freeze ranking and learn
one shared signed margin coordinate for the already-selected candidate. The same scalar is used for
numeric policy-return regression and HOLD/ACTION classification, so an auxiliary classifier can no
longer disagree with the deployed numeric margin. A zero-preserving asinh target compresses extreme
TFV-return magnitudes without changing the sign or the HOLD=0 boundary.

Calibration truth remains sealed until the selected Validation checkpoint passes the development gate.
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
import torch.nn.functional as F

from rtc.checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from rtc.direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
    DIRECT_TFV_POLICY_RETURN_ESTIMAND,
    sha256_file,
)
from rtc.direct_tfv_policy_return_query_margin_v2 import (
    DIRECT_TFV_QUERY_MARGIN_V2_CHECKPOINT_CONTRACT,
    DIRECT_TFV_QUERY_MARGIN_V2_CONTRACT,
    DIRECT_TFV_QUERY_MARGIN_V2_FEATURE_CONTRACT,
    QUERY_MARGIN_V2_HIDDEN_DIM,
    QueryConditionedPolicyReturnAdapterV2,
    build_query_margin_v2_features,
    freeze_step2_for_step3,
)
from rtc.native_supervisory_control import load_native_supervisory_control
from rtc.production_cli import _load_graph
from train_direct_tfv_policy_return_query_margin_current import (
    _load_dataset,
    _rank_loss,
    _rank_scores,
)
from build_step3_development_bank_current import STEP3_DEVELOPMENT_BANK_CONTRACT


STEP3_V15_TRAINING_CONTRACT = (
    "PROJECT7_STEP3_ACCURACY_FIRST_LATENT_TRAINING_V4_TWO_STAGE_SHARED_BOUNDARY_ASINH"
)
RANK_ADAPTER_EPOCHS = 8
RANK_ADAPTER_LR = 2.0e-5
MARGIN_EPOCHS = 60
MARGIN_LR = 3.0e-4
MARGIN_TRANSFORM_REGRESSION_WEIGHT = 1.0
MARGIN_SHARED_BOUNDARY_BCE_WEIGHT = 1.0
HOLD_WEIGHT_CAP = 5.0

DEVELOPMENT_ACCEPTANCE = {
    "selected_action_false_beneficial_fraction_max": 0.25,
    "selected_action_false_reject_fraction_max": 0.25,
    "hold_aware_decision_accuracy_min": 0.60,
    "within_query_pairwise_rank_accuracy_min": 0.60,
    "within_query_candidate_top1_accuracy_min": 0.55,
    "execute_all_collapse_forbidden_when_oracle_hold_exists": True,
    "hold_all_collapse_forbidden_when_oracle_action_exists": True,
}


def _development_metadata(path: str | Path, expected_split: str) -> dict[str, Any]:
    """Read only split/lineage metadata; do not touch exact policy-return labels."""
    with np.load(path, allow_pickle=False) as data:
        if str(np.asarray(data["development_bank_contract"]).reshape(-1)[0]) != STEP3_DEVELOPMENT_BANK_CONTRACT:
            raise ValueError("V16 requires the current 8:1:1 development bank")
        if str(np.asarray(data["development_split"]).reshape(-1)[0]) != expected_split:
            raise ValueError(f"V16 dataset is not the expected {expected_split} split")
        groups = {str(value) for value in np.asarray(data["rainfall_group"]).reshape(-1).tolist()}
        continuation = str(np.asarray(data["continuation_policy_sha256"]).reshape(-1)[0]).lower()
        mask_sha = str(np.asarray(data["supervisory_mask_sha256"]).reshape(-1)[0]).lower()
    return {
        "groups": groups,
        "continuation_policy_sha256": continuation,
        "supervisory_mask_sha256": mask_sha,
        "sha256": sha256_file(path),
    }


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _parameters_sha256(parameters: list[torch.nn.Parameter]) -> str:
    digest = hashlib.sha256()
    for index, parameter in enumerate(parameters):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(str(index).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _features(
    rank_model: torch.nn.Module,
    normalization: Any,
    graph: Any,
    dataset: dict[str, Any],
    indices: np.ndarray,
    *,
    mask: np.ndarray,
    scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    first = int(indices[0])
    return build_query_margin_v2_features(
        step2_model=rank_model,
        normalization=normalization,
        graph=graph,
        current_state=torch.as_tensor(dataset["current_state"][first], dtype=torch.float32, device=device),
        rainfall_scenarios=torch.as_tensor(dataset["rainfall_scenarios"][first], dtype=torch.float32, device=device),
        previous_actuator_flow=torch.as_tensor(dataset["previous_actuator_flow"][first], dtype=torch.float32, device=device),
        active_target=torch.as_tensor(dataset["active_target"][first], dtype=torch.float32, device=device),
        candidate_targets=torch.as_tensor(dataset["candidate_target"][indices], dtype=torch.float32, device=device),
        base_step2_scores_m3=torch.as_tensor(dataset["base_step2_h10_score_m3"][indices], dtype=torch.float32, device=device),
        candidate_sources=[str(value) for value in dataset["sources"][indices].tolist()],
        supervisory_mask=mask,
        target_scale_m3=scale,
    )


def _selected_truth(output: Any, truth: torch.Tensor) -> torch.Tensor:
    index = output.selected_candidate_index.to(device=truth.device, dtype=torch.long).reshape(())
    selected = int(index.detach().cpu())
    if selected < 0 or selected >= int(truth.numel()):
        raise ValueError("V16 selected candidate index is out of range")
    return truth[index]


def _margin_loss(
    output: Any,
    truth: torch.Tensor,
    *,
    scale: float,
    hold_positive_weight: float,
) -> torch.Tensor:
    """Fit one shared signed scalar to the selected candidate's exact return."""
    selected_truth = _selected_truth(output, truth)
    target_coordinate = torch.asinh(selected_truth / float(scale))
    hold_target = (selected_truth >= 0.0).to(dtype=truth.dtype)

    # Balance minority HOLD queries for both magnitude and boundary learning.
    regression_weight = 1.0 + hold_target * (float(hold_positive_weight) - 1.0)
    transformed_regression = regression_weight * F.smooth_l1_loss(
        output.margin_coordinate,
        target_coordinate,
    )
    shared_boundary_bce = F.binary_cross_entropy_with_logits(
        output.margin_coordinate,
        hold_target,
        pos_weight=torch.as_tensor(
            float(hold_positive_weight),
            dtype=truth.dtype,
            device=truth.device,
        ),
    )
    return (
        MARGIN_TRANSFORM_REGRESSION_WEIGHT * transformed_regression
        + MARGIN_SHARED_BOUNDARY_BCE_WEIGHT * shared_boundary_bce
    )


def _evaluate(
    rank_model: torch.nn.Module,
    adapter: QueryConditionedPolicyReturnAdapterV2,
    dataset: dict[str, Any],
    *,
    graph: Any,
    normalization: Any,
    mask: np.ndarray,
    device: torch.device,
    scale: float,
) -> dict[str, Any]:
    rank_model.eval()
    adapter.eval()
    false_beneficial = false_reject = correct_decision = 0
    predicted_hold = oracle_hold = top1 = 0
    pairwise_correct = pairwise_count = 0
    selected_margin_boundary_correct = 0
    oracle_action_boundary_correct = 0
    executed_count = 0
    ranking_regrets: list[float] = []
    regrets: list[float] = []
    predictions: list[float] = []
    truths: list[float] = []
    margins: list[float] = []
    margin_coordinates: list[float] = []
    selected_truths: list[float] = []
    oracle_best_truths: list[float] = []
    transformed_selected_truths: list[float] = []
    selected_sources: Counter[str] = Counter()
    oracle_sources: Counter[str] = Counter()
    queries = sorted(set(dataset["queries"].tolist()))

    with torch.no_grad():
        for query in queries:
            indices = np.flatnonzero(dataset["queries"] == query)
            truth = dataset["true_policy_return_delta_tfv_m3"][indices].astype(float).reshape(-1)
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
            output = adapter(
                raw_rank_scores_m3=raw,
                context_features=context,
                candidate_features=candidates,
            )
            predicted = output.predicted_returns_m3.detach().cpu().numpy().astype(float)
            selected = int(output.selected_candidate_index.detach().cpu())
            if selected != int(np.argmin(predicted)):
                raise RuntimeError("V16 selected-candidate index differs from minimum predicted return")
            oracle = int(np.argmin(truth))
            selected_truth = float(truth[selected])
            oracle_best = float(truth[oracle])
            margin = float(output.query_best_margin_m3.detach().cpu())
            coordinate = float(output.margin_coordinate.detach().cpu())
            hold_logit = float(output.hold_logit.detach().cpu())
            if abs(coordinate - hold_logit) > 1.0e-7:
                raise RuntimeError("V16 HOLD logit drifted away from the deployed numeric margin coordinate")

            execute = margin < 0.0
            oracle_execute = oracle_best < 0.0
            selected_is_beneficial = selected_truth < 0.0
            realized = selected_truth if execute else 0.0
            oracle_value = min(0.0, oracle_best)
            false_beneficial += int(execute and not selected_is_beneficial)
            false_reject += int((not execute) and oracle_execute)
            executed_count += int(execute)
            predicted_hold += int(not execute)
            oracle_hold += int(not oracle_execute)
            selected_margin_boundary_correct += int(execute == selected_is_beneficial)
            oracle_action_boundary_correct += int(execute == oracle_execute)
            correct_decision += int(
                ((not execute) and (not oracle_execute))
                or (execute and oracle_execute and selected == oracle)
            )
            top1 += int(selected == oracle)
            ranking_regrets.append(max(0.0, selected_truth - oracle_best))
            regrets.append(max(0.0, realized - oracle_value))
            predictions.extend(predicted.tolist())
            truths.extend(truth.tolist())
            margins.append(margin)
            margin_coordinates.append(coordinate)
            selected_truths.append(selected_truth)
            oracle_best_truths.append(oracle_best)
            transformed_selected_truths.append(float(np.arcsinh(selected_truth / float(scale))))
            selected_sources[str(dataset["sources"][indices[selected]])] += 1
            oracle_sources[str(dataset["sources"][indices[oracle]])] += 1
            for left in range(len(predicted)):
                for right in range(left + 1, len(predicted)):
                    if abs(truth[left] - truth[right]) <= 1.0:
                        continue
                    pairwise_count += 1
                    pairwise_correct += int(
                        np.sign(predicted[left] - predicted[right])
                        == np.sign(truth[left] - truth[right])
                    )

    n = max(1, len(queries))
    prediction_array = np.asarray(predictions, dtype=float)
    truth_array = np.asarray(truths, dtype=float)
    informative = np.abs(truth_array) > 1.0
    margin_array = np.asarray(margins, dtype=float)
    coordinate_array = np.asarray(margin_coordinates, dtype=float)
    selected_truth_array = np.asarray(selected_truths, dtype=float)
    transformed_truth_array = np.asarray(transformed_selected_truths, dtype=float)
    oracle_best_array = np.asarray(oracle_best_truths, dtype=float)
    predicted_hold_fraction = predicted_hold / n
    oracle_hold_fraction = oracle_hold / n
    execute_all_collapse = bool(oracle_hold > 0 and predicted_hold == 0)
    hold_all_collapse = bool(oracle_hold < len(queries) and predicted_hold == len(queries))

    return {
        "query_count": len(queries),
        "selected_action_false_beneficial_fraction": false_beneficial / n,
        "selected_action_false_reject_fraction": false_reject / n,
        "selected_action_worst_side_error_fraction": max(false_beneficial / n, false_reject / n),
        "selected_action_balanced_error_fraction": 0.5 * (false_beneficial / n + false_reject / n),
        "executed_action_count": executed_count,
        "executed_action_false_beneficial_conditional_fraction": (
            false_beneficial / executed_count if executed_count else 0.0
        ),
        "held_query_false_reject_conditional_fraction": (
            false_reject / predicted_hold if predicted_hold else 0.0
        ),
        "hold_aware_decision_accuracy": correct_decision / n,
        "oracle_action_hold_boundary_accuracy": oracle_action_boundary_correct / n,
        "selected_candidate_margin_sign_accuracy": selected_margin_boundary_correct / n,
        "shared_numeric_margin_boundary_accuracy": selected_margin_boundary_correct / n,
        "within_query_pairwise_rank_accuracy": pairwise_correct / pairwise_count if pairwise_count else 1.0,
        "within_query_candidate_top1_accuracy": top1 / n,
        "mean_ranking_regret_before_hold_m3": float(np.mean(ranking_regrets)) if ranking_regrets else 0.0,
        "selected_action_mean_regret_m3": float(np.mean(regrets)) if regrets else 0.0,
        "predicted_hold_fraction": predicted_hold_fraction,
        "oracle_hold_optimal_fraction": oracle_hold_fraction,
        "execute_all_collapse": execute_all_collapse,
        "hold_all_collapse": hold_all_collapse,
        "aux_hold_classifier_accuracy": selected_margin_boundary_correct / n,
        "hold_logit_is_numeric_margin_coordinate": True,
        "query_margin_selected_return_mae_m3": (
            float(np.mean(np.abs(margin_array - selected_truth_array))) if margin_array.size else 0.0
        ),
        "query_margin_oracle_best_return_mae_m3": (
            float(np.mean(np.abs(margin_array - oracle_best_array))) if margin_array.size else 0.0
        ),
        "query_margin_asinh_coordinate_mae": (
            float(np.mean(np.abs(coordinate_array - transformed_truth_array)))
            if coordinate_array.size
            else 0.0
        ),
        "event_balanced_mae_m3": (
            float(np.mean(np.abs(prediction_array - truth_array))) if prediction_array.size else 0.0
        ),
        "event_balanced_sign_accuracy": (
            float(np.mean(np.sign(prediction_array[informative]) == np.sign(truth_array[informative])))
            if np.any(informative)
            else 1.0
        ),
        "selected_candidate_source_counts": dict(sorted(selected_sources.items())),
        "oracle_best_candidate_source_counts": dict(sorted(oracle_sources.items())),
        "query_margin_m3": {
            "min": float(np.min(margin_array)) if margin_array.size else 0.0,
            "q25": float(np.quantile(margin_array, 0.25)) if margin_array.size else 0.0,
            "median": float(np.median(margin_array)) if margin_array.size else 0.0,
            "q75": float(np.quantile(margin_array, 0.75)) if margin_array.size else 0.0,
            "max": float(np.max(margin_array)) if margin_array.size else 0.0,
        },
        "query_margin_coordinate": {
            "min": float(np.min(coordinate_array)) if coordinate_array.size else 0.0,
            "q25": float(np.quantile(coordinate_array, 0.25)) if coordinate_array.size else 0.0,
            "median": float(np.median(coordinate_array)) if coordinate_array.size else 0.0,
            "q75": float(np.quantile(coordinate_array, 0.75)) if coordinate_array.size else 0.0,
            "max": float(np.max(coordinate_array)) if coordinate_array.size else 0.0,
        },
        "selected_candidate_true_return_m3": {
            "min": float(np.min(selected_truth_array)) if selected_truth_array.size else 0.0,
            "median": float(np.median(selected_truth_array)) if selected_truth_array.size else 0.0,
            "max": float(np.max(selected_truth_array)) if selected_truth_array.size else 0.0,
        },
        "oracle_best_true_return_m3": {
            "min": float(np.min(oracle_best_array)) if oracle_best_array.size else 0.0,
            "median": float(np.median(oracle_best_array)) if oracle_best_array.size else 0.0,
            "max": float(np.max(oracle_best_array)) if oracle_best_array.size else 0.0,
        },
    }


def _selection_key(metrics: dict[str, Any]) -> tuple[float, ...]:
    collapse = bool(metrics["execute_all_collapse"] or metrics["hold_all_collapse"])
    return (
        float(collapse),
        float(metrics["selected_action_worst_side_error_fraction"]),
        float(metrics["selected_action_balanced_error_fraction"]),
        -float(metrics["selected_candidate_margin_sign_accuracy"]),
        -float(metrics["hold_aware_decision_accuracy"]),
        float(metrics["selected_action_mean_regret_m3"]),
        -float(metrics["within_query_pairwise_rank_accuracy"]),
        -float(metrics["within_query_candidate_top1_accuracy"]),
        abs(float(metrics["predicted_hold_fraction"]) - float(metrics["oracle_hold_optimal_fraction"])),
        float(metrics["query_margin_asinh_coordinate_mae"]),
        float(metrics["query_margin_selected_return_mae_m3"]),
    )


def _accepted(metrics: dict[str, Any]) -> bool:
    t = DEVELOPMENT_ACCEPTANCE
    return bool(
        metrics["selected_action_false_beneficial_fraction"]
        <= t["selected_action_false_beneficial_fraction_max"]
        and metrics["selected_action_false_reject_fraction"]
        <= t["selected_action_false_reject_fraction_max"]
        and metrics["hold_aware_decision_accuracy"]
        >= t["hold_aware_decision_accuracy_min"]
        and metrics["within_query_pairwise_rank_accuracy"]
        >= t["within_query_pairwise_rank_accuracy_min"]
        and metrics["within_query_candidate_top1_accuracy"]
        >= t["within_query_candidate_top1_accuracy_min"]
        and not metrics["execute_all_collapse"]
        and not metrics["hold_all_collapse"]
    )


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
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _selected_hold_weight(
    rank_model: torch.nn.Module,
    adapter: QueryConditionedPolicyReturnAdapterV2,
    train: dict[str, Any],
    *,
    graph: Any,
    normalization: Any,
    mask: np.ndarray,
    device: torch.device,
    scale: float,
) -> tuple[float, int, int]:
    hold_count = 0
    action_count = 0
    adapter.eval()
    with torch.no_grad():
        for query in sorted(set(train["queries"].tolist())):
            _indices, truth, raw, context, candidates = _query_inputs(
                rank_model,
                normalization,
                graph,
                train,
                query,
                mask=mask,
                scale=scale,
                device=device,
            )
            output = adapter(
                raw_rank_scores_m3=raw,
                context_features=context,
                candidate_features=candidates,
            )
            if float(_selected_truth(output, truth).detach().cpu()) >= 0.0:
                hold_count += 1
            else:
                action_count += 1
    if hold_count <= 0 or action_count <= 0:
        raise ValueError("V16 rank-selected training queries must contain both HOLD and ACTION classes")
    return min(HOLD_WEIGHT_CAP, action_count / hold_count), hold_count, action_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-step2", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--supervisory-control", required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--validation-dataset", required=True)
    parser.add_argument("--calibration-dataset", required=True)
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
        raise ValueError("V16 train and validation rainfall groups overlap")
    if train_meta["groups"] & calibration_meta["groups"]:
        raise ValueError("V16 train and calibration rainfall groups overlap")
    if validation_meta["groups"] & calibration_meta["groups"]:
        raise ValueError("V16 validation and calibration rainfall groups overlap")
    if not (
        train_meta["continuation_policy_sha256"]
        == validation_meta["continuation_policy_sha256"]
        == calibration_meta["continuation_policy_sha256"]
    ):
        raise ValueError("V16 development bank mixes continuation lineage")
    if not (
        train_meta["supervisory_mask_sha256"]
        == validation_meta["supervisory_mask_sha256"]
        == calibration_meta["supervisory_mask_sha256"]
    ):
        raise ValueError("V16 development bank mixes supervisory-mask lineage")

    # Only train and validation truth are loaded before the gate.
    train = _load_dataset(args.train_dataset, role="policy_return_train")
    validation = _load_dataset(args.validation_dataset, role="policy_return_validation")

    graph = _load_graph(args.graph)
    control, mask = load_native_supervisory_control(
        args.supervisory_control,
        actuator_ids=graph.actuator_ids,
    )
    if str(control["supervisory_mask_sha256"]).lower() != train_meta["supervisory_mask_sha256"]:
        raise ValueError("V16 supervisory-control artifact differs from truth lineage")

    rank_model, normalization, _base = load_direct_tfv_runtime_checkpoint(
        args.base_step2,
        graph=graph,
        device=device,
    )
    freeze_step2_for_step3(rank_model)
    step2_state_before = _model_state_sha256(rank_model)
    scale = float(rank_model.target_scale_m3.detach().cpu())

    train_queries = sorted(set(train["queries"].tolist()))
    first_query = train_queries[0]
    first_indices = np.flatnonzero(train["queries"] == first_query)
    with torch.no_grad():
        context_probe, candidate_probe = _features(
            rank_model,
            normalization,
            graph,
            train,
            first_indices,
            mask=mask,
            scale=scale,
            device=device,
        )
    adapter = QueryConditionedPolicyReturnAdapterV2(
        target_scale_m3=scale,
        context_dim=int(context_probe.numel()),
        candidate_dim=int(candidate_probe.shape[1]),
    ).to(device)

    # Stage 1: train only the Step3 ranking branch.
    adapter.set_rank_stage()
    rank_optimizer = torch.optim.AdamW(
        adapter.rank_parameters(),
        lr=RANK_ADAPTER_LR,
        weight_decay=1.0e-4,
    )
    rank_history: list[float] = []
    for epoch in range(1, RANK_ADAPTER_EPOCHS + 1):
        adapter.train()
        total = 0.0
        ordered_queries = sorted(
            train_queries,
            key=lambda value: hashlib.sha256(
                f"rank|{args.seed}|{epoch}|{value}".encode("utf-8")
            ).hexdigest(),
        )
        for query in ordered_queries:
            _indices, truth, raw, context, candidates = _query_inputs(
                rank_model,
                normalization,
                graph,
                train,
                query,
                mask=mask,
                scale=scale,
                device=device,
            )
            output = adapter(
                raw_rank_scores_m3=raw,
                context_features=context,
                candidate_features=candidates,
            )
            loss = _rank_loss(output.rank_scores_normalized, truth, scale)
            rank_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.rank_parameters(), 5.0)
            rank_optimizer.step()
            total += float(loss.detach().cpu())
        rank_history.append(total / max(1, len(train_queries)))

    # Stage 2: freeze ranking completely. The selected candidate is now a fixed learned routing rule.
    adapter.set_margin_stage()
    rank_state_before_margin = _parameters_sha256(adapter.rank_parameters())
    hold_positive_weight, selected_hold_count, selected_action_count = _selected_hold_weight(
        rank_model,
        adapter,
        train,
        graph=graph,
        normalization=normalization,
        mask=mask,
        device=device,
        scale=scale,
    )
    margin_optimizer = torch.optim.AdamW(
        adapter.margin_parameters(),
        lr=MARGIN_LR,
        weight_decay=1.0e-4,
    )

    history: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, MARGIN_EPOCHS + 1):
        adapter.train()
        total = 0.0
        ordered_queries = sorted(
            train_queries,
            key=lambda value: hashlib.sha256(
                f"margin|{args.seed}|{epoch}|{value}".encode("utf-8")
            ).hexdigest(),
        )
        for query in ordered_queries:
            _indices, truth, raw, context, candidates = _query_inputs(
                rank_model,
                normalization,
                graph,
                train,
                query,
                mask=mask,
                scale=scale,
                device=device,
            )
            output = adapter(
                raw_rank_scores_m3=raw,
                context_features=context,
                candidate_features=candidates,
            )
            loss = _margin_loss(
                output,
                truth,
                scale=scale,
                hold_positive_weight=hold_positive_weight,
            )
            margin_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.margin_parameters(), 5.0)
            margin_optimizer.step()
            total += float(loss.detach().cpu())

        validation_metrics = _evaluate(
            rank_model,
            adapter,
            validation,
            graph=graph,
            normalization=normalization,
            mask=mask,
            device=device,
            scale=scale,
        )
        key = _selection_key(validation_metrics)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / max(1, len(train_queries)),
                "validation_metrics": validation_metrics,
                "selection_key": list(key),
            }
        )
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy(adapter.state_dict())

    if best_state is None:
        raise RuntimeError("V16 failed to select a development checkpoint")
    adapter.load_state_dict(best_state)
    adapter.set_margin_stage()

    rank_state_after_margin = _parameters_sha256(adapter.rank_parameters())
    if rank_state_before_margin != rank_state_after_margin:
        raise RuntimeError("V16 margin fitting mutated the frozen Step3 rank branch")
    step2_state_after = _model_state_sha256(rank_model)
    if step2_state_before != step2_state_after:
        raise RuntimeError("V16 Step3 training mutated the frozen Step2 representation")

    validation_metrics = _evaluate(
        rank_model,
        adapter,
        validation,
        graph=graph,
        normalization=normalization,
        mask=mask,
        device=device,
        scale=scale,
    )
    validation_pass = _accepted(validation_metrics)

    calibration_truth_loaded = False
    calibration_metrics: dict[str, Any] | None = None
    if validation_pass:
        calibration = _load_dataset(args.calibration_dataset, role="policy_return_calibration")
        calibration_truth_loaded = True
        calibration_metrics = _evaluate(
            rank_model,
            adapter,
            calibration,
            graph=graph,
            normalization=normalization,
            mask=mask,
            device=device,
            scale=scale,
        )

    payload = {
        "contract": DIRECT_TFV_QUERY_MARGIN_V2_CHECKPOINT_CONTRACT,
        "training_contract": STEP3_V15_TRAINING_CONTRACT,
        "query_margin_contract": DIRECT_TFV_QUERY_MARGIN_V2_CONTRACT,
        "feature_contract": DIRECT_TFV_QUERY_MARGIN_V2_FEATURE_CONTRACT,
        "development_only": True,
        "estimand": DIRECT_TFV_POLICY_RETURN_ESTIMAND,
        "action_encoding_contract": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
        "base_step2_sha256": sha256_file(args.base_step2),
        "graph_sha256": sha256_file(args.graph),
        "supervisory_control_sha256": sha256_file(args.supervisory_control),
        "supervisory_mask_sha256": train_meta["supervisory_mask_sha256"],
        "supervisory_control_dimension": 82,
        "model_action_channel_count": 109,
        "train_dataset_sha256": train_meta["sha256"],
        "validation_dataset_sha256": validation_meta["sha256"],
        "calibration_dataset_sha256": calibration_meta["sha256"],
        "train_group_count": len(train_meta["groups"]),
        "validation_group_count": len(validation_meta["groups"]),
        "calibration_group_count": len(calibration_meta["groups"]),
        "split_contract": STEP3_DEVELOPMENT_BANK_CONTRACT,
        "base_step2_frozen": True,
        "step2_parameters_updated": False,
        "step2_retrained": False,
        "base_step2_state_sha256_before_training": step2_state_before,
        "base_step2_state_sha256_after_training": step2_state_after,
        "rank_and_margin_two_stage_training": True,
        "rank_adapter_epochs": RANK_ADAPTER_EPOCHS,
        "rank_adapter_learning_rate": RANK_ADAPTER_LR,
        "rank_loss_history": rank_history,
        "rank_branch_state_sha256_before_margin": rank_state_before_margin,
        "rank_branch_state_sha256_after_margin": rank_state_after_margin,
        "rank_branch_frozen_during_margin_training": True,
        "margin_epochs_max": MARGIN_EPOCHS,
        "margin_learning_rate": MARGIN_LR,
        "selected_margin_epoch": best_epoch,
        "margin_training_history": history,
        "hold_positive_weight_from_rank_selected_training_only": hold_positive_weight,
        "rank_selected_training_hold_count": selected_hold_count,
        "rank_selected_training_action_count": selected_action_count,
        "numeric_margin_target_transform": "ASINH_RETURN_OVER_STEP2_SCALE_ZERO_PRESERVING",
        "numeric_margin_inverse_transform": "STEP2_SCALE_TIMES_SINH_COORDINATE",
        "hold_logit_is_numeric_margin_coordinate": True,
        "independent_auxiliary_hold_head": False,
        "validation_metrics": validation_metrics,
        "calibration_truth_loaded": calibration_truth_loaded,
        "calibration_raw_metrics": calibration_metrics,
        "development_acceptance": DEVELOPMENT_ACCEPTANCE,
        "development_validation_pass": validation_pass,
        "context_dim": adapter.context_dim,
        "candidate_dim": adapter.candidate_dim,
        "query_margin_hidden_dim": QUERY_MARGIN_V2_HIDDEN_DIM,
        "query_margin_state_dict": copy.deepcopy(adapter.state_dict()),
        "online_swmm_called": False,
        "ready_for_pi1_development": False,
        "ready_for_policy_lock": False,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    report = {key: value for key, value in payload.items() if key != "query_margin_state_dict"}
    report["checkpoint_sha256"] = sha256_file(out)
    out.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))

    if not validation_pass:
        raise RuntimeError(
            "V16 Step3 failed the 8:1:1 development validation gate; calibration truth stayed sealed"
        )


if __name__ == "__main__":
    main()
