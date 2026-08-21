"""Train Step3 V17 on the existing 8:1:1 development bank with zero new SWMM truth.

V17 deliberately reuses the already development-validated V15 rank subnetwork and freezes it. Only
the selected-candidate vs HOLD boundary and absolute return magnitude are learned. Validation selects
the margin checkpoint; calibration truth remains unopened until Validation passes.
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
import torch.nn.functional as F

from rtc.checkpoint_direct_tfv import load_direct_tfv_runtime_checkpoint
from rtc.direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
    DIRECT_TFV_POLICY_RETURN_ESTIMAND,
    sha256_file,
)
from rtc.direct_tfv_policy_return_query_margin_v17 import (
    DIRECT_TFV_QUERY_MARGIN_V17_CHECKPOINT_CONTRACT,
    DIRECT_TFV_QUERY_MARGIN_V17_CONTRACT,
    LEGACY_V15_CHECKPOINT_CONTRACT,
    QueryConditionedPolicyReturnAdapterV17,
    import_v15_rank_state,
    rank_state_sha256,
)
from rtc.native_supervisory_control import load_native_supervisory_control
from rtc.production_cli import _load_graph
from build_step3_development_bank_current import STEP3_DEVELOPMENT_BANK_CONTRACT
from train_direct_tfv_policy_return_query_margin_current import _load_dataset, _rank_scores
from train_step3_query_margin_v2_current import _development_metadata, _features, _model_state_sha256


STEP3_V17_TRAINING_CONTRACT = (
    "PROJECT7_STEP3_V17_REUSE_VALIDATED_RANK_SELECTED_ACTION_BOUNDARY_MAGNITUDE"
)
MARGIN_EPOCHS = 60
MARGIN_LR = 3.0e-4
BOUNDARY_WEIGHT = 1.0
MAGNITUDE_WEIGHT = 0.45
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
RANK_SOURCE_MINIMUM = {
    "within_query_pairwise_rank_accuracy_min": 0.90,
    "within_query_candidate_top1_accuracy_min": 0.90,
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
        dataset["true_policy_return_delta_tfv_m3"][indices], dtype=torch.float32, device=device
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


def _evaluate(
    rank_model: torch.nn.Module,
    adapter: QueryConditionedPolicyReturnAdapterV17,
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
    queries = sorted(set(dataset["queries"].tolist()))
    false_beneficial = false_reject = correct_decision = 0
    predicted_hold = oracle_hold = top1 = 0
    pairwise_correct = pairwise_count = 0
    boundary_correct = 0
    executed = 0
    ranking_regrets: list[float] = []
    decision_regrets: list[float] = []
    selected_truths: list[float] = []
    margins: list[float] = []
    boundary_logits: list[float] = []
    magnitudes: list[float] = []
    selected_sources: dict[str, int] = {}
    oracle_sources: dict[str, int] = {}

    with torch.no_grad():
        for query in queries:
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
            output = adapter(
                raw_rank_scores_m3=raw,
                context_features=context,
                candidate_features=candidates,
            )
            truth = truth_tensor.detach().cpu().numpy().astype(float)
            selected = int(output.selected_candidate_index.detach().cpu())
            oracle = int(np.argmin(truth))
            selected_truth = float(truth[selected])
            oracle_best = float(truth[oracle])
            margin = float(output.query_best_margin_m3.detach().cpu())
            boundary_logit = float(output.hold_logit.detach().cpu())
            magnitude = float(output.magnitude_coordinate.detach().cpu())
            execute = margin < 0.0
            selected_beneficial = selected_truth < 0.0
            oracle_execute = oracle_best < 0.0
            realized = selected_truth if execute else 0.0
            oracle_value = min(0.0, oracle_best)

            false_beneficial += int(execute and not selected_beneficial)
            false_reject += int((not execute) and oracle_execute)
            correct_decision += int(
                ((not execute) and (not oracle_execute))
                or (execute and oracle_execute and selected == oracle)
            )
            predicted_hold += int(not execute)
            oracle_hold += int(not oracle_execute)
            boundary_correct += int((boundary_logit >= 0.0) == (not selected_beneficial))
            executed += int(execute)
            top1 += int(selected == oracle)
            ranking_regrets.append(max(0.0, selected_truth - oracle_best))
            decision_regrets.append(max(0.0, realized - oracle_value))
            selected_truths.append(selected_truth)
            margins.append(margin)
            boundary_logits.append(boundary_logit)
            magnitudes.append(magnitude)
            source = str(dataset["sources"][indices[selected]])
            oracle_source = str(dataset["sources"][indices[oracle]])
            selected_sources[source] = selected_sources.get(source, 0) + 1
            oracle_sources[oracle_source] = oracle_sources.get(oracle_source, 0) + 1

            rank_scores = output.rank_scores_normalized.detach().cpu().numpy().astype(float)
            for left in range(len(rank_scores)):
                for right in range(left + 1, len(rank_scores)):
                    if abs(truth[left] - truth[right]) <= 1.0:
                        continue
                    pairwise_count += 1
                    pairwise_correct += int(
                        np.sign(rank_scores[left] - rank_scores[right])
                        == np.sign(truth[left] - truth[right])
                    )

    n = max(1, len(queries))
    selected_truth_array = np.asarray(selected_truths, dtype=float)
    margin_array = np.asarray(margins, dtype=float)
    logit_array = np.asarray(boundary_logits, dtype=float)
    magnitude_array = np.asarray(magnitudes, dtype=float)
    predicted_hold_fraction = predicted_hold / n
    oracle_hold_fraction = oracle_hold / n
    execute_all = bool(oracle_hold > 0 and predicted_hold == 0)
    hold_all = bool(oracle_hold < len(queries) and predicted_hold == len(queries))
    return {
        "query_count": len(queries),
        "selected_action_false_beneficial_fraction": false_beneficial / n,
        "selected_action_false_reject_fraction": false_reject / n,
        "selected_action_worst_side_error_fraction": max(false_beneficial / n, false_reject / n),
        "selected_action_balanced_error_fraction": 0.5 * (false_beneficial / n + false_reject / n),
        "hold_aware_decision_accuracy": correct_decision / n,
        "selected_candidate_margin_sign_accuracy": boundary_correct / n,
        "within_query_pairwise_rank_accuracy": pairwise_correct / pairwise_count if pairwise_count else 1.0,
        "within_query_candidate_top1_accuracy": top1 / n,
        "mean_ranking_regret_before_hold_m3": float(np.mean(ranking_regrets)) if ranking_regrets else 0.0,
        "selected_action_mean_regret_m3": float(np.mean(decision_regrets)) if decision_regrets else 0.0,
        "executed_action_count": executed,
        "executed_action_false_beneficial_conditional_fraction": (
            false_beneficial / executed if executed else 0.0
        ),
        "predicted_hold_fraction": predicted_hold_fraction,
        "oracle_hold_optimal_fraction": oracle_hold_fraction,
        "execute_all_collapse": execute_all,
        "hold_all_collapse": hold_all,
        "query_margin_selected_return_mae_m3": (
            float(np.mean(np.abs(margin_array - selected_truth_array))) if margin_array.size else 0.0
        ),
        "boundary_logit": {
            "min": float(np.min(logit_array)) if logit_array.size else 0.0,
            "q25": float(np.quantile(logit_array, 0.25)) if logit_array.size else 0.0,
            "median": float(np.median(logit_array)) if logit_array.size else 0.0,
            "q75": float(np.quantile(logit_array, 0.75)) if logit_array.size else 0.0,
            "max": float(np.max(logit_array)) if logit_array.size else 0.0,
        },
        "magnitude_coordinate": {
            "min": float(np.min(magnitude_array)) if magnitude_array.size else 0.0,
            "median": float(np.median(magnitude_array)) if magnitude_array.size else 0.0,
            "max": float(np.max(magnitude_array)) if magnitude_array.size else 0.0,
        },
        "query_margin_m3": {
            "min": float(np.min(margin_array)) if margin_array.size else 0.0,
            "q25": float(np.quantile(margin_array, 0.25)) if margin_array.size else 0.0,
            "median": float(np.median(margin_array)) if margin_array.size else 0.0,
            "q75": float(np.quantile(margin_array, 0.75)) if margin_array.size else 0.0,
            "max": float(np.max(margin_array)) if margin_array.size else 0.0,
        },
        "selected_candidate_source_counts": dict(sorted(selected_sources.items())),
        "oracle_best_candidate_source_counts": dict(sorted(oracle_sources.items())),
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


def _selection_key(metrics: dict[str, Any]) -> tuple[float, ...]:
    collapse = bool(metrics["execute_all_collapse"] or metrics["hold_all_collapse"])
    return (
        float(collapse),
        float(metrics["selected_action_worst_side_error_fraction"]),
        float(metrics["selected_action_balanced_error_fraction"]),
        -float(metrics["selected_candidate_margin_sign_accuracy"]),
        -float(metrics["hold_aware_decision_accuracy"]),
        float(metrics["selected_action_mean_regret_m3"]),
        float(metrics["query_margin_selected_return_mae_m3"]),
    )


def _rank_source_payload(
    path: str | Path,
    *,
    base_step2_sha256: str,
    train_sha256: str,
    validation_sha256: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(payload, dict) or payload.get("contract") != LEGACY_V15_CHECKPOINT_CONTRACT:
        raise ValueError("V17 requires the V15 selection-consistent checkpoint as rank source")
    if str(payload.get("base_step2_sha256", "")).lower() != base_step2_sha256.lower():
        raise ValueError("V17 rank source uses another base Step2 checkpoint")
    if str(payload.get("train_dataset_sha256", "")).lower() != train_sha256.lower():
        raise ValueError("V17 rank source uses another development train split")
    if str(payload.get("validation_dataset_sha256", "")).lower() != validation_sha256.lower():
        raise ValueError("V17 rank source uses another development validation split")
    metrics = payload.get("validation_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("V17 rank source lacks validation metrics")
    if float(metrics.get("within_query_pairwise_rank_accuracy", 0.0)) < RANK_SOURCE_MINIMUM[
        "within_query_pairwise_rank_accuracy_min"
    ]:
        raise ValueError("V17 rank source pairwise accuracy is below the reuse threshold")
    if float(metrics.get("within_query_candidate_top1_accuracy", 0.0)) < RANK_SOURCE_MINIMUM[
        "within_query_candidate_top1_accuracy_min"
    ]:
        raise ValueError("V17 rank source top1 accuracy is below the reuse threshold")
    return payload


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
        raise ValueError("V17 train/validation rainfall groups overlap")
    if train_meta["groups"] & calibration_meta["groups"]:
        raise ValueError("V17 train/calibration rainfall groups overlap")
    if validation_meta["groups"] & calibration_meta["groups"]:
        raise ValueError("V17 validation/calibration rainfall groups overlap")

    train = _load_dataset(args.train_dataset, role="policy_return_train")
    validation = _load_dataset(args.validation_dataset, role="policy_return_validation")
    graph = _load_graph(args.graph)
    control, mask = load_native_supervisory_control(args.supervisory_control, actuator_ids=graph.actuator_ids)
    if str(control["supervisory_mask_sha256"]).lower() != train_meta["supervisory_mask_sha256"]:
        raise ValueError("V17 supervisory-control artifact differs from truth lineage")

    rank_model, normalization, _base = load_direct_tfv_runtime_checkpoint(
        args.base_step2, graph=graph, device=device
    )
    for parameter in rank_model.parameters():
        parameter.requires_grad_(False)
    rank_model.eval()
    step2_before = _model_state_sha256(rank_model)
    scale = float(rank_model.target_scale_m3.detach().cpu())

    train_queries = sorted(set(train["queries"].tolist()))
    first_indices = np.flatnonzero(train["queries"] == train_queries[0])
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
    adapter = QueryConditionedPolicyReturnAdapterV17(
        target_scale_m3=scale,
        context_dim=int(context_probe.numel()),
        candidate_dim=int(candidate_probe.shape[1]),
    ).to(device)

    rank_source = _rank_source_payload(
        args.rank_source_checkpoint,
        base_step2_sha256=sha256_file(args.base_step2),
        train_sha256=train_meta["sha256"],
        validation_sha256=validation_meta["sha256"],
    )
    import_v15_rank_state(adapter, rank_source)
    rank_sha_before_margin = rank_state_sha256(adapter)

    # Verify that migration reproduces the previously validated ranking before touching margin.
    pre_margin_validation = _evaluate(
        rank_model,
        adapter,
        validation,
        graph=graph,
        normalization=normalization,
        mask=mask,
        device=device,
        scale=scale,
    )
    if pre_margin_validation["within_query_pairwise_rank_accuracy"] < 0.90:
        raise RuntimeError("V17 migrated rank failed pairwise reproduction")
    if pre_margin_validation["within_query_candidate_top1_accuracy"] < 0.90:
        raise RuntimeError("V17 migrated rank failed top1 reproduction")

    # Compute HOLD class weight from rank-selected training targets only.
    hold_count = action_count = 0
    selected_truth_by_query: dict[str, float] = {}
    for query in train_queries:
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
        with torch.no_grad():
            output = adapter(
                raw_rank_scores_m3=raw,
                context_features=context,
                candidate_features=candidates,
            )
            selected = int(output.selected_candidate_index.detach().cpu())
            selected_truth = float(truth[selected].detach().cpu())
        selected_truth_by_query[query] = selected_truth
        if selected_truth >= 0.0:
            hold_count += 1
        else:
            action_count += 1
    if hold_count <= 0 or action_count <= 0:
        raise ValueError("V17 training requires both selected-action HOLD and ACTION examples")
    hold_positive_weight = min(HOLD_WEIGHT_CAP, action_count / hold_count)

    adapter.train_margin_only()
    optimizer = torch.optim.AdamW(adapter.margin_parameters(), lr=MARGIN_LR, weight_decay=1.0e-4)
    history: list[dict[str, Any]] = []
    best_key: tuple[float, ...] | None = None
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, MARGIN_EPOCHS + 1):
        adapter.train()
        total = 0.0
        ordered = sorted(
            train_queries,
            key=lambda value: hashlib.sha256(f"v17|{args.seed}|{epoch}|{value}".encode()).hexdigest(),
        )
        for query in ordered:
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
            selected = int(output.selected_candidate_index.detach().cpu())
            selected_truth = truth[selected]
            hold_target = (selected_truth >= 0.0).to(dtype=truth.dtype)
            boundary_loss = F.binary_cross_entropy_with_logits(
                output.hold_logit,
                hold_target,
                pos_weight=torch.as_tensor(
                    hold_positive_weight, dtype=truth.dtype, device=truth.device
                ),
            )
            magnitude_target = torch.abs(torch.asinh(selected_truth / float(scale)))
            magnitude_loss = F.smooth_l1_loss(output.magnitude_coordinate, magnitude_target)
            loss = BOUNDARY_WEIGHT * boundary_loss + MAGNITUDE_WEIGHT * magnitude_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.margin_parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach().cpu())

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
        key = _selection_key(metrics)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total / max(1, len(train_queries)),
                "validation_metrics": metrics,
                "selection_key": list(key),
            }
        )
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = epoch
            best_state = copy.deepcopy(adapter.state_dict())

    if best_state is None:
        raise RuntimeError("V17 failed to select a margin checkpoint")
    adapter.load_state_dict(best_state)
    adapter.freeze_rank()
    rank_sha_after_margin = rank_state_sha256(adapter)
    if rank_sha_before_margin != rank_sha_after_margin:
        raise RuntimeError("V17 margin fitting mutated the imported frozen rank")
    step2_after = _model_state_sha256(rank_model)
    if step2_before != step2_after:
        raise RuntimeError("V17 mutated frozen Step2")

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
        "contract": DIRECT_TFV_QUERY_MARGIN_V17_CHECKPOINT_CONTRACT,
        "training_contract": STEP3_V17_TRAINING_CONTRACT,
        "query_margin_contract": DIRECT_TFV_QUERY_MARGIN_V17_CONTRACT,
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
        "train_dataset_sha256": train_meta["sha256"],
        "validation_dataset_sha256": validation_meta["sha256"],
        "calibration_dataset_sha256": calibration_meta["sha256"],
        "split_contract": STEP3_DEVELOPMENT_BANK_CONTRACT,
        "train_group_count": len(train_meta["groups"]),
        "validation_group_count": len(validation_meta["groups"]),
        "calibration_group_count": len(calibration_meta["groups"]),
        "rank_source_checkpoint_sha256": sha256_file(args.rank_source_checkpoint),
        "rank_source_checkpoint_contract": LEGACY_V15_CHECKPOINT_CONTRACT,
        "rank_reused_from_v15": True,
        "rank_retrained_in_v17": False,
        "rank_source_validation_metrics": rank_source["validation_metrics"],
        "rank_reproduction_metrics_before_margin": pre_margin_validation,
        "rank_branch_state_sha256_before_margin": rank_sha_before_margin,
        "rank_branch_state_sha256_after_margin": rank_sha_after_margin,
        "rank_branch_frozen_during_margin_training": True,
        "hold_positive_weight_from_training_only": hold_positive_weight,
        "rank_selected_training_hold_count": hold_count,
        "rank_selected_training_action_count": action_count,
        "boundary_controls_numeric_margin_sign": True,
        "magnitude_regression_can_flip_boundary": False,
        "selected_margin_epoch": best_epoch,
        "margin_epochs_max": MARGIN_EPOCHS,
        "margin_training_history": history,
        "validation_metrics": validation_metrics,
        "development_acceptance": DEVELOPMENT_ACCEPTANCE,
        "development_validation_pass": validation_pass,
        "calibration_truth_loaded": calibration_truth_loaded,
        "calibration_raw_metrics": calibration_metrics,
        "context_dim": adapter.context_dim,
        "candidate_dim": adapter.candidate_dim,
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
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))

    if not validation_pass:
        raise RuntimeError("V17 Step3 failed development Validation; calibration truth remained sealed")


if __name__ == "__main__":
    main()
