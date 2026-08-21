"""Train the accuracy-first latent Step3 critic on an 8:1:1 development bank.

This is intentionally a zero-SWMM DEVELOPMENT workflow. The train/validation/calibration NPZ files
must come from ``build_step3_development_bank_current.py``. The pretrained Step2 network is frozen:
Step3 learns only the query adapter, including its candidate-rank adjustment and selection-consistent
HOLD margin. Validation selects the adapter epoch using decision metrics rather than MAE. The final
10% calibration split is evaluated only after the selected checkpoint passes the validation gate.
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
    "PROJECT7_STEP3_ACCURACY_FIRST_LATENT_TRAINING_V2_SELECTION_CONSISTENT_FROZEN_STEP2"
)
MARGIN_EPOCHS = 60
MARGIN_LR = 3.0e-4
MARGIN_REGRESSION_WEIGHT = 1.0
MARGIN_SIGN_WEIGHT = 0.70
HOLD_CLASSIFICATION_WEIGHT = 0.80
FINAL_RETURN_WEIGHT = 0.15
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


def _assert_development_dataset(path: str | Path, expected_split: str) -> None:
    data = np.load(path, allow_pickle=False)
    if str(np.asarray(data["development_bank_contract"]).reshape(-1)[0]) != STEP3_DEVELOPMENT_BANK_CONTRACT:
        raise ValueError("V15 requires the current 8:1:1 development bank")
    if str(np.asarray(data["development_split"]).reshape(-1)[0]) != expected_split:
        raise ValueError(f"V15 dataset is not the expected {expected_split} split")


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
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
    if int(index.detach().cpu()) < 0 or int(index.detach().cpu()) >= int(truth.numel()):
        raise ValueError("V15 selected candidate index is out of range")
    return truth[index]


def _adapter_loss(
    output: Any,
    truth: torch.Tensor,
    *,
    scale: float,
    hold_positive_weight: float,
) -> torch.Tensor:
    # Critical alignment: the numeric margin authorizes the candidate actually selected by the rank
    # branch, so it must be supervised against that candidate's exact return, not oracle min(truth).
    selected_truth = _selected_truth(output, truth)
    hold_target = (selected_truth >= 0.0).to(dtype=truth.dtype)
    class_weight = 1.0 + hold_target * (float(hold_positive_weight) - 1.0)
    margin = output.query_best_margin_m3
    margin_regression = class_weight * F.smooth_l1_loss(
        margin / float(scale), selected_truth / float(scale)
    )
    if abs(float(selected_truth.detach().cpu())) > 1.0:
        margin_sign = class_weight * F.softplus(
            -(margin / float(scale)) * torch.sign(selected_truth)
        )
    else:
        margin_sign = margin.new_zeros(())
    hold_bce = F.binary_cross_entropy_with_logits(
        output.hold_logit,
        hold_target,
        pos_weight=torch.as_tensor(float(hold_positive_weight), dtype=truth.dtype, device=truth.device),
    )
    rank = _rank_loss(output.rank_scores_normalized, truth, scale)
    final_regression = F.smooth_l1_loss(
        output.predicted_returns_m3 / float(scale), truth / float(scale)
    )
    return (
        MARGIN_REGRESSION_WEIGHT * margin_regression
        + MARGIN_SIGN_WEIGHT * margin_sign
        + HOLD_CLASSIFICATION_WEIGHT * hold_bce
        + rank
        + FINAL_RETURN_WEIGHT * final_regression
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
    hold_classifier_correct = 0
    selected_margin_boundary_correct = 0
    oracle_action_boundary_correct = 0
    executed_count = 0
    ranking_regrets: list[float] = []
    regrets: list[float] = []
    predictions: list[float] = []
    truths: list[float] = []
    margins: list[float] = []
    selected_truths: list[float] = []
    oracle_best_truths: list[float] = []
    selected_sources: Counter[str] = Counter()
    oracle_sources: Counter[str] = Counter()
    queries = sorted(set(dataset["queries"].tolist()))
    with torch.no_grad():
        for query in queries:
            indices = np.flatnonzero(dataset["queries"] == query)
            truth = dataset["true_policy_return_delta_tfv_m3"][indices].astype(float).reshape(-1)
            raw = _rank_scores(
                rank_model, dataset, indices, graph=graph, normalization=normalization, device=device
            )
            context, candidates = _features(
                rank_model, normalization, graph, dataset, indices,
                mask=mask, scale=scale, device=device,
            )
            output = adapter(
                raw_rank_scores_m3=raw,
                context_features=context,
                candidate_features=candidates,
            )
            predicted = output.predicted_returns_m3.detach().cpu().numpy().astype(float)
            selected = int(output.selected_candidate_index.detach().cpu())
            if selected != int(np.argmin(predicted)):
                raise RuntimeError("V15 selected-candidate index differs from minimum predicted return")
            oracle = int(np.argmin(truth))
            selected_truth = float(truth[selected])
            oracle_best = float(truth[oracle])
            margin = float(output.query_best_margin_m3.detach().cpu())
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
            hold_classifier_correct += int(
                (float(output.hold_logit.detach().cpu()) >= 0.0) == (not selected_is_beneficial)
            )
            ranking_regrets.append(max(0.0, selected_truth - oracle_best))
            regrets.append(max(0.0, realized - oracle_value))
            predictions.extend(predicted.tolist())
            truths.extend(truth.tolist())
            margins.append(margin)
            selected_truths.append(selected_truth)
            oracle_best_truths.append(oracle_best)
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
    selected_truth_array = np.asarray(selected_truths, dtype=float)
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
        "within_query_pairwise_rank_accuracy": pairwise_correct / pairwise_count if pairwise_count else 1.0,
        "within_query_candidate_top1_accuracy": top1 / n,
        "mean_ranking_regret_before_hold_m3": float(np.mean(ranking_regrets)) if ranking_regrets else 0.0,
        "selected_action_mean_regret_m3": float(np.mean(regrets)) if regrets else 0.0,
        "predicted_hold_fraction": predicted_hold_fraction,
        "oracle_hold_optimal_fraction": oracle_hold_fraction,
        "execute_all_collapse": execute_all_collapse,
        "hold_all_collapse": hold_all_collapse,
        "aux_hold_classifier_accuracy": hold_classifier_correct / n,
        "query_margin_selected_return_mae_m3": (
            float(np.mean(np.abs(margin_array - selected_truth_array))) if margin_array.size else 0.0
        ),
        "query_margin_oracle_best_return_mae_m3": (
            float(np.mean(np.abs(margin_array - oracle_best_array))) if margin_array.size else 0.0
        ),
        "event_balanced_mae_m3": float(np.mean(np.abs(prediction_array - truth_array))) if prediction_array.size else 0.0,
        "event_balanced_sign_accuracy": float(
            np.mean(np.sign(prediction_array[informative]) == np.sign(truth_array[informative]))
        ) if np.any(informative) else 1.0,
        "selected_candidate_source_counts": dict(sorted(selected_sources.items())),
        "oracle_best_candidate_source_counts": dict(sorted(oracle_sources.items())),
        "query_margin_m3": {
            "min": float(np.min(margin_array)) if margin_array.size else 0.0,
            "q25": float(np.quantile(margin_array, 0.25)) if margin_array.size else 0.0,
            "median": float(np.median(margin_array)) if margin_array.size else 0.0,
            "q75": float(np.quantile(margin_array, 0.75)) if margin_array.size else 0.0,
            "max": float(np.max(margin_array)) if margin_array.size else 0.0,
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
    return (
        float(metrics["selected_action_worst_side_error_fraction"]),
        float(metrics["selected_action_balanced_error_fraction"]),
        -float(metrics["selected_candidate_margin_sign_accuracy"]),
        -float(metrics["hold_aware_decision_accuracy"]),
        float(metrics["selected_action_mean_regret_m3"]),
        -float(metrics["within_query_pairwise_rank_accuracy"]),
        -float(metrics["within_query_candidate_top1_accuracy"]),
        abs(float(metrics["predicted_hold_fraction"]) - float(metrics["oracle_hold_optimal_fraction"])),
        float(metrics["query_margin_selected_return_mae_m3"]),
        float(metrics["event_balanced_mae_m3"]),
    )


def _accepted(metrics: dict[str, Any]) -> bool:
    t = DEVELOPMENT_ACCEPTANCE
    return bool(
        metrics["selected_action_false_beneficial_fraction"] <= t["selected_action_false_beneficial_fraction_max"]
        and metrics["selected_action_false_reject_fraction"] <= t["selected_action_false_reject_fraction_max"]
        and metrics["hold_aware_decision_accuracy"] >= t["hold_aware_decision_accuracy_min"]
        and metrics["within_query_pairwise_rank_accuracy"] >= t["within_query_pairwise_rank_accuracy_min"]
        and metrics["within_query_candidate_top1_accuracy"] >= t["within_query_candidate_top1_accuracy_min"]
        and not bool(metrics["execute_all_collapse"])
        and not bool(metrics["hold_all_collapse"])
    )


def hashlib_sha(seed: int, epoch: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{epoch}|{value}".encode("utf-8")).hexdigest()


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
    for path, split in (
        (args.train_dataset, "train"),
        (args.validation_dataset, "validation"),
        (args.calibration_dataset, "calibration"),
    ):
        _assert_development_dataset(path, split)
    train = _load_dataset(args.train_dataset, role="policy_return_train")
    validation = _load_dataset(args.validation_dataset, role="policy_return_validation")
    calibration = _load_dataset(args.calibration_dataset, role="policy_return_calibration")
    train_groups = set(train["groups"].tolist())
    validation_groups = set(validation["groups"].tolist())
    calibration_groups = set(calibration["groups"].tolist())
    if train_groups & validation_groups or train_groups & calibration_groups or validation_groups & calibration_groups:
        raise ValueError("V15 8:1:1 rainfall-group splits overlap")
    if not (train["continuation_policy_sha256"] == validation["continuation_policy_sha256"] == calibration["continuation_policy_sha256"]):
        raise ValueError("V15 development bank mixes continuation lineage")
    if not (train["supervisory_mask_sha256"] == validation["supervisory_mask_sha256"] == calibration["supervisory_mask_sha256"]):
        raise ValueError("V15 development bank mixes supervisory-mask lineage")

    graph = _load_graph(args.graph)
    control, mask = load_native_supervisory_control(args.supervisory_control, actuator_ids=graph.actuator_ids)
    if str(control["supervisory_mask_sha256"]).lower() != train["supervisory_mask_sha256"]:
        raise ValueError("V15 supervisory-control artifact differs from truth lineage")
    rank_model, normalization, _base = load_direct_tfv_runtime_checkpoint(args.base_step2, graph=graph, device=device)
    freeze_step2_for_step3(rank_model)
    frozen_step2_sha_before = _model_state_sha256(rank_model)
    scale = float(rank_model.target_scale_m3.detach().cpu())

    train_queries = sorted(set(train["queries"].tolist()))
    first_query = train_queries[0]
    first_indices = np.flatnonzero(train["queries"] == first_query)
    with torch.no_grad():
        context_probe, candidate_probe = _features(
            rank_model, normalization, graph, train, first_indices,
            mask=mask, scale=scale, device=device,
        )
    adapter = QueryConditionedPolicyReturnAdapterV2(
        target_scale_m3=scale,
        context_dim=int(context_probe.numel()),
        candidate_dim=int(candidate_probe.shape[1]),
    ).to(device)

    hold_count = 0
    for query in train_queries:
        indices = np.flatnonzero(train["queries"] == query)
        hold_count += int(float(np.min(train["true_policy_return_delta_tfv_m3"][indices])) >= 0.0)
    action_count = len(train_queries) - hold_count
    if hold_count <= 0 or action_count <= 0:
        raise ValueError("V15 training split must contain both oracle-HOLD and oracle-ACTION queries")
    hold_positive_weight = min(HOLD_WEIGHT_CAP, action_count / hold_count)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=MARGIN_LR, weight_decay=1.0e-4)
    history: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, MARGIN_EPOCHS + 1):
        adapter.train()
        total = 0.0
        ordered_queries = sorted(train_queries, key=lambda value: hashlib_sha(args.seed, epoch, value))
        for query in ordered_queries:
            indices = np.flatnonzero(train["queries"] == query)
            truth = torch.as_tensor(
                train["true_policy_return_delta_tfv_m3"][indices], dtype=torch.float32, device=device
            ).reshape(-1)
            with torch.no_grad():
                raw = _rank_scores(
                    rank_model, train, indices, graph=graph, normalization=normalization, device=device
                )
                context, candidates = _features(
                    rank_model, normalization, graph, train, indices,
                    mask=mask, scale=scale, device=device,
                )
            output = adapter(
                raw_rank_scores_m3=raw,
                context_features=context,
                candidate_features=candidates,
            )
            loss = _adapter_loss(
                output, truth, scale=scale, hold_positive_weight=hold_positive_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu())
        validation_metrics = _evaluate(
            rank_model, adapter, validation,
            graph=graph, normalization=normalization, mask=mask, device=device, scale=scale,
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
        raise RuntimeError("V15 failed to select a development checkpoint")
    adapter.load_state_dict(best_state)
    frozen_step2_sha_after = _model_state_sha256(rank_model)
    if frozen_step2_sha_after != frozen_step2_sha_before:
        raise RuntimeError("V15 modified the frozen Step2 state")

    validation_metrics = _evaluate(
        rank_model, adapter, validation,
        graph=graph, normalization=normalization, mask=mask, device=device, scale=scale,
    )
    validation_pass = _accepted(validation_metrics)
    calibration_metrics: dict[str, Any] | None = None
    if validation_pass:
        calibration_metrics = _evaluate(
            rank_model, adapter, calibration,
            graph=graph, normalization=normalization, mask=mask, device=device, scale=scale,
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
        "base_step2_state_sha256_before_training": frozen_step2_sha_before,
        "base_step2_state_sha256_after_training": frozen_step2_sha_after,
        "base_step2_frozen": True,
        "step2_parameters_updated": False,
        "step2_retrained": False,
        "graph_sha256": sha256_file(args.graph),
        "supervisory_control_sha256": sha256_file(args.supervisory_control),
        "supervisory_mask_sha256": train["supervisory_mask_sha256"],
        "supervisory_control_dimension": 82,
        "model_action_channel_count": 109,
        "train_dataset_sha256": train["sha256"],
        "validation_dataset_sha256": validation["sha256"],
        "calibration_dataset_sha256": calibration["sha256"],
        "train_group_count": len(train_groups),
        "validation_group_count": len(validation_groups),
        "calibration_group_count": len(calibration_groups),
        "split_contract": STEP3_DEVELOPMENT_BANK_CONTRACT,
        "margin_target": "TRUE_RETURN_OF_RANK_SELECTED_CANDIDATE_VS_HOLD",
        "oracle_best_truth_used_for_margin_target": False,
        "hold_positive_weight_from_training_only": hold_positive_weight,
        "separate_step2_rank_finetuning_epochs": 0,
        "selected_margin_epoch": best_epoch,
        "margin_epochs_max": MARGIN_EPOCHS,
        "margin_training_history": history,
        "validation_metrics": validation_metrics,
        "calibration_evaluated_only_after_validation_pass": True,
        "calibration_raw_metrics": calibration_metrics,
        "development_acceptance": DEVELOPMENT_ACCEPTANCE,
        "development_validation_pass": validation_pass,
        "context_dim": adapter.context_dim,
        "candidate_dim": adapter.candidate_dim,
        "query_margin_hidden_dim": QUERY_MARGIN_V2_HIDDEN_DIM,
        "query_margin_state_dict": copy.deepcopy(adapter.state_dict()),
        "aux_hold_classifier_used_for_training_only": True,
        "aux_hold_classifier_used_online": False,
        "online_swmm_called": False,
        "step1_retrained": False,
        "base_step2_checkpoint_overwritten": False,
        "ready_for_pi1_development": False,
        "ready_for_policy_lock": False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    report = {key: value for key, value in payload.items() if key != "query_margin_state_dict"}
    report["checkpoint_sha256"] = sha256_file(out)
    out.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not validation_pass:
        raise RuntimeError(
            "V15 Step3 failed the 8:1:1 development validation gate; calibration remained unread "
            "and no new truth may be generated"
        )


if __name__ == "__main__":
    main()
