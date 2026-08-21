"""Train the fixed query-conditioned exact-policy-return decision critic.

The retired scalar critic used one output for two different tasks: within-query candidate ranking and
whether the best candidate beats HOLD=0. This trainer instead learns a rank critic and a separate
query-level best-candidate margin. The architecture, loss weights, optimization schedule and
acceptance thresholds are frozen before fresh Validation is inspected.

The supplied fresh Validation is evaluated once after training. It is not used for epoch selection,
learning-rate selection or loss-weight tuning. The consumed scalar-critic Validation is supplied via
--deprecated-validation-dataset and is explicitly forbidden from the fresh set.
"""
from __future__ import annotations

import argparse
import copy
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
    DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_ESTIMAND,
    encode_policy_return_action_token,
    sha256_file,
)
from rtc.direct_tfv_policy_return_query_margin import (
    DIRECT_TFV_QUERY_MARGIN_CHECKPOINT_CONTRACT,
    DIRECT_TFV_QUERY_MARGIN_CONTRACT,
    DIRECT_TFV_QUERY_MARGIN_FEATURE_CONTRACT,
    QUERY_MARGIN_HIDDEN_DIM,
    QueryConditionedPolicyReturnAdapter,
    build_query_margin_features,
)
from rtc.native_supervisory_control import load_native_supervisory_control
from rtc.production_cli import _load_graph


RANK_EPOCHS = 8
MARGIN_EPOCHS = 40
RANK_LR = 2.0e-5
MARGIN_LR = 5.0e-4
PAIRWISE_WEIGHT = 0.70
RELATIVE_REGRESSION_WEIGHT = 0.45
TOP1_WEIGHT = 0.35
MARGIN_REGRESSION_WEIGHT = 1.0
MARGIN_SIGN_WEIGHT = 0.50
FINAL_RETURN_WEIGHT = 0.20

FRESH_VALIDATION_THRESHOLDS = {
    "selected_action_false_beneficial_fraction_max": 0.25,
    "selected_action_false_reject_fraction_max": 0.25,
    "hold_aware_decision_accuracy_min": 0.50,
    "within_query_pairwise_rank_accuracy_min": 0.60,
    "within_query_candidate_top1_accuracy_min": 0.50,
}


def _scalar(data: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in data or np.asarray(data[key]).size != 1:
        raise ValueError(f"policy-return dataset lacks scalar {key}")
    return str(np.asarray(data[key]).reshape(-1)[0])


def _load_dataset(path: str | Path, *, role: str) -> dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    if _scalar(data, "contract") != DIRECT_TFV_POLICY_RETURN_DATASET_CONTRACT:
        raise ValueError("query-margin dataset has wrong contract")
    if _scalar(data, "estimand") != DIRECT_TFV_POLICY_RETURN_ESTIMAND:
        raise ValueError("query-margin dataset has wrong estimand")
    if _scalar(data, "action_encoding_contract") != DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING:
        raise ValueError("query-margin dataset has wrong action encoding")
    if _scalar(data, "data_role") != role:
        raise ValueError(f"query-margin dataset role must be {role}")
    keys = (
        "current_state",
        "rainfall_scenarios",
        "active_target",
        "candidate_target",
        "previous_actuator_flow",
        "true_policy_return_delta_tfv_m3",
        "base_step2_h10_score_m3",
        "rainfall_group",
        "query_set_id",
        "candidate_source",
        "first_move_changed_facility_count",
    )
    result = {key: np.asarray(data[key]) for key in keys}
    n = int(result["true_policy_return_delta_tfv_m3"].shape[0])
    if n <= 0 or any(int(result[key].shape[0]) != n for key in keys):
        raise ValueError("query-margin dataset arrays are not row aligned")
    if tuple(result["active_target"].shape[1:]) != (109,):
        raise ValueError("query-margin active target lost 109 channels")
    if tuple(result["candidate_target"].shape[1:]) != (109,):
        raise ValueError("query-margin candidate target lost 109 channels")
    result["groups"] = result.pop("rainfall_group").astype(str)
    result["queries"] = result.pop("query_set_id").astype(str)
    result["sources"] = result.pop("candidate_source").astype(str)
    result.update(
        {
            "path": str(Path(path).resolve()),
            "sha256": sha256_file(path),
            "continuation_policy_sha256": _scalar(
                data, "continuation_policy_sha256"
            ).lower(),
            "candidate_portfolio_contract": _scalar(data, "candidate_portfolio_contract"),
            "supervisory_mask_sha256": _scalar(data, "supervisory_mask_sha256").lower(),
        }
    )
    return result


def _group_set(path: str | Path) -> set[str]:
    data = np.load(path, allow_pickle=False)
    if "rainfall_group" not in data:
        raise ValueError("deprecated validation dataset lacks rainfall_group")
    return {
        str(value)
        for value in np.asarray(data["rainfall_group"]).astype(str).tolist()
    }


def _normalization_tensors(
    normalization: Any, *, dtype: torch.dtype, device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        "state_mean": torch.as_tensor(
            normalization.state_mean, dtype=dtype, device=device
        ),
        "state_std": torch.as_tensor(
            normalization.state_std, dtype=dtype, device=device
        ).clamp_min(1.0e-6),
        "rain_mean": torch.as_tensor(
            normalization.rainfall_mean, dtype=dtype, device=device
        ),
        "rain_std": torch.as_tensor(
            normalization.rainfall_std, dtype=dtype, device=device
        ).clamp_min(1.0e-6),
        "flow_mean": torch.as_tensor(
            normalization.flow_mean, dtype=dtype, device=device
        ),
        "flow_std": torch.as_tensor(
            normalization.flow_std, dtype=dtype, device=device
        ).clamp_min(1.0e-6),
    }


def _rank_scores(
    model: torch.nn.Module,
    dataset: dict[str, Any],
    indices: np.ndarray,
    *,
    graph: Any,
    normalization: Any,
    device: torch.device,
) -> torch.Tensor:
    state_raw = torch.as_tensor(
        dataset["current_state"][indices], dtype=torch.float32, device=device
    )
    rain_raw = torch.as_tensor(
        dataset["rainfall_scenarios"][indices], dtype=torch.float32, device=device
    )
    active = torch.as_tensor(
        dataset["active_target"][indices], dtype=torch.float32, device=device
    )
    target = torch.as_tensor(
        dataset["candidate_target"][indices], dtype=torch.float32, device=device
    )
    flow_raw = torch.as_tensor(
        dataset["previous_actuator_flow"][indices], dtype=torch.float32, device=device
    )
    k, scenarios, horizon, nodes, rain_features = rain_raw.shape
    norm = _normalization_tensors(
        normalization, dtype=state_raw.dtype, device=device
    )
    state = (state_raw - norm["state_mean"]) / norm["state_std"]
    rain = (rain_raw - norm["rain_mean"]) / norm["rain_std"]
    flow = (flow_raw - norm["flow_mean"]) / norm["flow_std"]
    state = state[:, None].expand(-1, scenarios, -1, -1).reshape(
        k * scenarios, *state.shape[1:]
    )
    rain = rain.reshape(k * scenarios, horizon, nodes, rain_features)
    flow = flow[:, None].expand(-1, scenarios, -1).reshape(k * scenarios, 109)
    active = active[:, None].expand(-1, scenarios, -1).reshape(k * scenarios, 109)
    target = target[:, None].expand(-1, scenarios, -1).reshape(k * scenarios, 109)
    reference, candidate = encode_policy_return_action_token(
        active,
        target,
        horizon_steps=int(horizon),
        first_action_steps=2,
    )
    output = model(
        current_state=state,
        rainfall=rain,
        reference_settings=reference,
        candidate_settings=candidate,
        previous_actuator_flow=flow,
        actuator_upstream=torch.as_tensor(
            graph.actuator_upstream, dtype=torch.long, device=device
        ),
        actuator_downstream=torch.as_tensor(
            graph.actuator_downstream, dtype=torch.long, device=device
        ),
        actuator_physics=torch.as_tensor(
            graph.actuator_physics, dtype=state.dtype, device=device
        ),
    )
    scores = output.total_delta_tfv_m3.reshape(k, scenarios).mean(dim=1)
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError("query-margin rank critic produced non-finite scores")
    return scores


def _features(
    dataset: dict[str, Any],
    indices: np.ndarray,
    *,
    mask: np.ndarray,
    scale: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    first = int(indices[0])
    return build_query_margin_features(
        current_state=torch.as_tensor(
            dataset["current_state"][first], dtype=torch.float32, device=device
        ),
        rainfall_scenarios=torch.as_tensor(
            dataset["rainfall_scenarios"][first], dtype=torch.float32, device=device
        ),
        previous_actuator_flow=torch.as_tensor(
            dataset["previous_actuator_flow"][first], dtype=torch.float32, device=device
        ),
        active_target=torch.as_tensor(
            dataset["active_target"][first], dtype=torch.float32, device=device
        ),
        candidate_targets=torch.as_tensor(
            dataset["candidate_target"][indices], dtype=torch.float32, device=device
        ),
        base_step2_scores_m3=torch.as_tensor(
            dataset["base_step2_h10_score_m3"][indices],
            dtype=torch.float32,
            device=device,
        ),
        candidate_sources=[
            str(value) for value in dataset["sources"][indices].tolist()
        ],
        supervisory_mask=mask,
        target_scale_m3=scale,
    )


def _pairwise_loss(score: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    if int(score.numel()) < 2:
        return score.new_zeros(())
    score_difference = score[:, None] - score[None, :]
    truth_difference = truth[:, None] - truth[None, :]
    use = torch.triu(torch.abs(truth_difference) > 1.0, diagonal=1)
    if not bool(use.any()):
        return score.new_zeros(())
    return F.softplus(
        -score_difference[use] * torch.sign(truth_difference[use])
    ).mean()


def _rank_loss(
    score_normalized: torch.Tensor, truth: torch.Tensor, scale: float
) -> torch.Tensor:
    truth_relative = (truth - torch.min(truth)) / float(scale)
    score_relative = score_normalized - torch.min(score_normalized)
    pairwise = _pairwise_loss(score_normalized, truth)
    relative = F.smooth_l1_loss(score_relative, truth_relative)
    top1 = F.cross_entropy(
        (-score_normalized).reshape(1, -1), torch.argmin(truth).reshape(1)
    )
    return (
        PAIRWISE_WEIGHT * pairwise
        + RELATIVE_REGRESSION_WEIGHT * relative
        + TOP1_WEIGHT * top1
    )


def _adapter_loss(output: Any, truth: torch.Tensor, scale: float) -> torch.Tensor:
    best = torch.min(truth)
    margin = output.query_best_margin_m3
    margin_regression = F.smooth_l1_loss(
        margin / float(scale), best / float(scale)
    )
    if abs(float(best.detach().cpu())) > 1.0:
        # Correct direction: truth<0 must push margin negative; truth>0 must push it positive.
        margin_sign = F.softplus(
            -(margin / float(scale)) * torch.sign(best)
        )
    else:
        margin_sign = margin.new_zeros(())
    rank = _rank_loss(output.rank_scores_normalized, truth, scale)
    final_regression = F.smooth_l1_loss(
        output.predicted_returns_m3 / float(scale), truth / float(scale)
    )
    return (
        MARGIN_REGRESSION_WEIGHT * margin_regression
        + MARGIN_SIGN_WEIGHT * margin_sign
        + rank
        + FINAL_RETURN_WEIGHT * final_regression
    )


def _evaluate(
    rank_model: torch.nn.Module,
    adapter: QueryConditionedPolicyReturnAdapter,
    dataset: dict[str, Any],
    *,
    graph: Any,
    normalization: Any,
    mask: np.ndarray,
    device: torch.device,
    scale: float,
) -> dict[str, float]:
    rank_model.eval()
    adapter.eval()
    false_beneficial = 0
    false_reject = 0
    correct_decision = 0
    predicted_hold = 0
    oracle_hold = 0
    top1 = 0
    pairwise_correct = 0
    pairwise_count = 0
    regrets: list[float] = []
    predictions: list[float] = []
    truths: list[float] = []
    queries = sorted(set(dataset["queries"].tolist()))
    with torch.no_grad():
        for query in queries:
            indices = np.flatnonzero(dataset["queries"] == query)
            raw = _rank_scores(
                rank_model,
                dataset,
                indices,
                graph=graph,
                normalization=normalization,
                device=device,
            )
            context, candidates = _features(
                dataset, indices, mask=mask, scale=scale, device=device
            )
            output = adapter(
                raw_rank_scores_m3=raw,
                context_features=context,
                candidate_features=candidates,
            )
            predicted = (
                output.predicted_returns_m3.detach().cpu().numpy().astype(float)
            )
            truth = (
                dataset["true_policy_return_delta_tfv_m3"][indices]
                .astype(float)
                .reshape(-1)
            )
            selected = int(np.argmin(predicted))
            oracle = int(np.argmin(truth))
            execute = bool(float(output.query_best_margin_m3.detach().cpu()) < 0.0)
            oracle_execute = bool(float(truth[oracle]) < 0.0)
            realized = float(truth[selected]) if execute else 0.0
            oracle_value = min(0.0, float(truth[oracle]))
            false_beneficial += int(execute and truth[selected] >= 0.0)
            false_reject += int((not execute) and oracle_execute)
            predicted_hold += int(not execute)
            oracle_hold += int(not oracle_execute)
            correct_decision += int(
                ((not execute) and (not oracle_execute))
                or (execute and oracle_execute and selected == oracle)
            )
            top1 += int(selected == oracle)
            regrets.append(max(0.0, realized - oracle_value))
            predictions.extend(predicted.tolist())
            truths.extend(truth.tolist())
            for left in range(len(predicted)):
                for right in range(left + 1, len(predicted)):
                    if abs(truth[left] - truth[right]) <= 1.0:
                        continue
                    pairwise_count += 1
                    pairwise_correct += int(
                        np.sign(predicted[left] - predicted[right])
                        == np.sign(truth[left] - truth[right])
                    )
    query_count = max(1, len(queries))
    prediction_array = np.asarray(predictions)
    truth_array = np.asarray(truths)
    informative = np.abs(truth_array) > 1.0
    return {
        "query_count": float(len(queries)),
        "selected_action_false_beneficial_fraction": float(
            false_beneficial / query_count
        ),
        "selected_action_false_reject_fraction": float(
            false_reject / query_count
        ),
        "selected_action_worst_side_error_fraction": float(
            max(false_beneficial / query_count, false_reject / query_count)
        ),
        "selected_action_balanced_error_fraction": float(
            0.5 * (false_beneficial / query_count + false_reject / query_count)
        ),
        "hold_aware_decision_accuracy": float(correct_decision / query_count),
        "within_query_pairwise_rank_accuracy": (
            float(pairwise_correct / pairwise_count) if pairwise_count else 1.0
        ),
        "within_query_candidate_top1_accuracy": float(top1 / query_count),
        "selected_action_mean_regret_m3": (
            float(np.mean(regrets)) if regrets else 0.0
        ),
        "predicted_hold_fraction": float(predicted_hold / query_count),
        "oracle_hold_optimal_fraction": float(oracle_hold / query_count),
        "event_balanced_mae_m3": (
            float(np.mean(np.abs(prediction_array - truth_array)))
            if prediction_array.size
            else 0.0
        ),
        "event_balanced_sign_accuracy": (
            float(
                np.mean(
                    np.sign(prediction_array[informative])
                    == np.sign(truth_array[informative])
                )
            )
            if np.any(informative)
            else 1.0
        ),
    }


def _accept(metrics: dict[str, float]) -> bool:
    thresholds = FRESH_VALIDATION_THRESHOLDS
    return bool(
        metrics["selected_action_false_beneficial_fraction"]
        <= thresholds["selected_action_false_beneficial_fraction_max"]
        and metrics["selected_action_false_reject_fraction"]
        <= thresholds["selected_action_false_reject_fraction_max"]
        and metrics["hold_aware_decision_accuracy"]
        >= thresholds["hold_aware_decision_accuracy_min"]
        and metrics["within_query_pairwise_rank_accuracy"]
        >= thresholds["within_query_pairwise_rank_accuracy_min"]
        and metrics["within_query_candidate_top1_accuracy"]
        >= thresholds["within_query_candidate_top1_accuracy_min"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-step2", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--supervisory-control", required=True)
    parser.add_argument("--train-dataset", required=True)
    parser.add_argument("--fresh-validation-dataset", required=True)
    parser.add_argument("--deprecated-validation-dataset", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        args.device
        if args.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    graph = _load_graph(args.graph)
    train = _load_dataset(args.train_dataset, role="policy_return_train")
    validation = _load_dataset(
        args.fresh_validation_dataset, role="policy_return_validation"
    )
    train_groups = set(train["groups"].tolist())
    validation_groups = set(validation["groups"].tolist())
    deprecated_groups = _group_set(args.deprecated_validation_dataset)
    if train_groups & validation_groups:
        raise ValueError("query-margin train/fresh-validation rainfall groups overlap")
    if validation_groups & deprecated_groups:
        raise ValueError(
            "fresh validation reuses consumed architecture-development validation groups"
        )
    if len(train_groups) < 48 or len(validation_groups) < 12:
        raise ValueError("query-margin requires >=48 train and >=12 fresh validation groups")
    if train["continuation_policy_sha256"] != validation["continuation_policy_sha256"]:
        raise ValueError("query-margin train/validation continuation lineage differs")
    if train["supervisory_mask_sha256"] != validation["supervisory_mask_sha256"]:
        raise ValueError("query-margin train/validation supervisory masks differ")
    control, mask = load_native_supervisory_control(
        args.supervisory_control, actuator_ids=graph.actuator_ids
    )
    if (
        str(control["supervisory_mask_sha256"]).lower()
        != train["supervisory_mask_sha256"]
    ):
        raise ValueError(
            "query-margin supervisory-control artifact differs from truth lineage"
        )

    rank_model, normalization, _base = load_direct_tfv_runtime_checkpoint(
        args.base_step2, graph=graph, device=device
    )
    scale = float(rank_model.target_scale_m3.detach().cpu())
    adapter = QueryConditionedPolicyReturnAdapter(target_scale_m3=scale).to(device)
    rank_optimizer = torch.optim.AdamW(
        rank_model.parameters(), lr=RANK_LR, weight_decay=1.0e-5
    )
    train_queries = sorted(set(train["queries"].tolist()))
    rank_history: list[float] = []
    for _epoch in range(RANK_EPOCHS):
        rank_model.train()
        total = 0.0
        for query in train_queries:
            indices = np.flatnonzero(train["queries"] == query)
            truth = torch.as_tensor(
                train["true_policy_return_delta_tfv_m3"][indices],
                dtype=torch.float32,
                device=device,
            ).reshape(-1)
            raw = _rank_scores(
                rank_model,
                train,
                indices,
                graph=graph,
                normalization=normalization,
                device=device,
            )
            loss = _rank_loss(raw / scale, truth, scale)
            rank_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(rank_model.parameters(), 5.0)
            rank_optimizer.step()
            total += float(loss.detach().cpu())
        rank_history.append(total / max(1, len(train_queries)))

    for parameter in rank_model.parameters():
        parameter.requires_grad_(False)
    rank_model.eval()
    margin_optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=MARGIN_LR, weight_decay=1.0e-4
    )
    margin_history: list[float] = []
    for _epoch in range(MARGIN_EPOCHS):
        adapter.train()
        total = 0.0
        for query in train_queries:
            indices = np.flatnonzero(train["queries"] == query)
            truth = torch.as_tensor(
                train["true_policy_return_delta_tfv_m3"][indices],
                dtype=torch.float32,
                device=device,
            ).reshape(-1)
            with torch.no_grad():
                raw = _rank_scores(
                    rank_model,
                    train,
                    indices,
                    graph=graph,
                    normalization=normalization,
                    device=device,
                )
            context, candidates = _features(
                train, indices, mask=mask, scale=scale, device=device
            )
            output = adapter(
                raw_rank_scores_m3=raw,
                context_features=context,
                candidate_features=candidates,
            )
            loss = _adapter_loss(output, truth, scale)
            margin_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 5.0)
            margin_optimizer.step()
            total += float(loss.detach().cpu())
        margin_history.append(total / max(1, len(train_queries)))

    metrics = _evaluate(
        rank_model,
        adapter,
        validation,
        graph=graph,
        normalization=normalization,
        mask=mask,
        device=device,
        scale=scale,
    )
    accepted = _accept(metrics)
    payload = {
        "contract": DIRECT_TFV_QUERY_MARGIN_CHECKPOINT_CONTRACT,
        "query_margin_contract": DIRECT_TFV_QUERY_MARGIN_CONTRACT,
        "query_margin_feature_contract": DIRECT_TFV_QUERY_MARGIN_FEATURE_CONTRACT,
        "development_only": True,
        "estimand": DIRECT_TFV_POLICY_RETURN_ESTIMAND,
        "action_encoding_contract": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
        "base_step2_sha256": sha256_file(args.base_step2),
        "graph_sha256": sha256_file(args.graph),
        "supervisory_control_sha256": sha256_file(args.supervisory_control),
        "supervisory_mask_sha256": train["supervisory_mask_sha256"],
        "supervisory_control_dimension": 82,
        "model_action_channel_count": 109,
        "candidate_portfolio_contract": train["candidate_portfolio_contract"],
        "continuation_policy_sha256": train["continuation_policy_sha256"],
        "train_dataset_sha256": train["sha256"],
        "fresh_validation_dataset_sha256": validation["sha256"],
        "deprecated_validation_dataset_sha256": sha256_file(
            args.deprecated_validation_dataset
        ),
        "train_rainfall_group_count": len(train_groups),
        "validation_rainfall_group_count": len(validation_groups),
        "fresh_validation_groups_disjoint_from_consumed_validation": True,
        "fresh_validation_used_for_training": False,
        "fresh_validation_used_for_hyperparameter_selection": False,
        "fresh_validation_verified": accepted,
        "fresh_validation_thresholds": FRESH_VALIDATION_THRESHOLDS,
        "fresh_validation_metrics": metrics,
        "rank_training_epochs_fixed": RANK_EPOCHS,
        "margin_training_epochs_fixed": MARGIN_EPOCHS,
        "rank_learning_rate_fixed": RANK_LR,
        "margin_learning_rate_fixed": MARGIN_LR,
        "query_margin_hidden_dim": QUERY_MARGIN_HIDDEN_DIM,
        "rank_training_loss_history": rank_history,
        "margin_training_loss_history": margin_history,
        "rank_model_state_dict": copy.deepcopy(rank_model.state_dict()),
        "query_margin_state_dict": copy.deepcopy(adapter.state_dict()),
        "base_step2_retrained": False,
        "step1_retrained": False,
        "online_swmm_called": False,
        "ready_for_calibration": accepted,
        "ready_for_pi1_development": False,
        "ready_for_policy_lock": False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    report = {
        key: value
        for key, value in payload.items()
        if key not in {"rank_model_state_dict", "query_margin_state_dict"}
    }
    report["checkpoint_sha256"] = sha256_file(out)
    out.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not accepted:
        raise RuntimeError(
            "query-conditioned critic failed the pre-registered fresh-validation gate; "
            "do not calibrate or run pi1"
        )


if __name__ == "__main__":
    main()
