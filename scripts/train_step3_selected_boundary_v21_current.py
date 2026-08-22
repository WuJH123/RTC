"""Train Project7 Step3 V21 on the existing bank with deployment-aligned selected-query supervision.

V21 fixes the task mismatch exposed by V20. The frozen V15 rank selects one candidate per query first.
Only that selected candidate's exact return is used as the ACTION/HOLD boundary label. Sibling
candidate truth is not used to fit the boundary; sibling candidate features remain available as
causal portfolio context because the same portfolio exists online. No SWMM is called.
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
from rtc.direct_tfv_policy_return_query_margin_v17 import rank_state_sha256
from rtc.direct_tfv_policy_return_selected_boundary_v21 import (
    BOUNDARY_ZERO,
    DIRECT_TFV_SELECTED_BOUNDARY_V21_CHECKPOINT_CONTRACT,
    DIRECT_TFV_SELECTED_BOUNDARY_V21_CONTRACT,
    DIRECT_TFV_SELECTED_BOUNDARY_V21_FEATURE_CONTRACT,
    SVD_COMPONENTS,
    SelectedBoundaryCalibratorV21,
    SelectedBoundaryPartsV21,
    SelectedBoundaryPreprocessorV21,
    build_selected_portfolio_feature_v21,
)
from rtc.native_supervisory_control import load_native_supervisory_control
from rtc.production_cli import _load_graph
from build_step3_development_bank_current import STEP3_DEVELOPMENT_BANK_CONTRACT
from train_direct_tfv_policy_return_query_margin_current import _load_dataset
from train_step3_facility_boundary_v20_current import (
    DEVELOPMENT_ACCEPTANCE,
    RIDGE_GRID,
    TRAIN_OOF_AUC_MIN,
    _accepted,
    _auc_hold,
    _collect_rows,
    _development_metadata,
    _fit_logistic_no_intercept,
    _fit_magnitude_no_intercept,
    _model_state_sha256,
)
from train_step3_query_margin_v18_current import _make_rank_adapter, _rank_source_payload


STEP3_V21_TRAINING_CONTRACT = (
    "PROJECT7_STEP3_V21_FROZEN_V15_RANK_SELECTED_QUERY_DEPLOYMENT_BOUNDARY"
)
CV_FOLDS = 6


def _action_masses(dataset: dict[str, Any], mask: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for query in sorted(set(dataset["queries"].tolist())):
        indices = np.flatnonzero(dataset["queries"] == query)
        for row_index in indices.tolist():
            active = np.asarray(dataset["active_target"][row_index], dtype=np.float64)
            candidate = np.asarray(dataset["candidate_target"][row_index], dtype=np.float64)
            delta = candidate[mask] - active[mask]
            values.append(float(np.sqrt(np.mean(np.square(delta)))))
    return np.asarray(values, dtype=np.float64)


def _selected_query_rows(all_rows: dict[str, Any], action_masses: np.ndarray) -> dict[str, Any]:
    if int(action_masses.size) != int(all_rows["returns"].size):
        raise RuntimeError("V21 action-mass rows do not align with V20 candidate features")
    features: list[np.ndarray] = []
    returns: list[float] = []
    queries: list[str] = []
    selected_sources: list[str] = []
    oracle_sources: list[str] = []
    selected_indices: list[int] = []
    oracle_indices: list[int] = []

    for query in sorted(set(all_rows["queries"].tolist())):
        idx = np.flatnonzero(all_rows["queries"] == query)
        local_selected = np.flatnonzero(all_rows["selected_mask"][idx])
        if local_selected.size != 1:
            raise RuntimeError("V21 requires exactly one frozen-rank selected candidate per query")
        selected_local = int(local_selected[0])
        selected_global = int(idx[selected_local])
        truth = all_rows["returns"][idx]
        oracle_local = int(np.argmin(truth))
        oracle_global = int(idx[oracle_local])
        parts = build_selected_portfolio_feature_v21(
            candidate_features=torch.as_tensor(
                all_rows["features"][idx], dtype=torch.float32
            ),
            rank_scores=torch.as_tensor(all_rows["rank_scores"][idx], dtype=torch.float32),
            selected_index=selected_local,
            selected_action_mass=float(action_masses[selected_global]),
        )
        features.append(parts.feature.detach().cpu().numpy().astype(np.float64))
        returns.append(float(all_rows["returns"][selected_global]))
        queries.append(str(query))
        selected_sources.append(str(all_rows["sources"][selected_global]))
        oracle_sources.append(str(all_rows["sources"][oracle_global]))
        selected_indices.append(selected_global)
        oracle_indices.append(oracle_global)

    width = {int(value.size) for value in features}
    if len(width) != 1:
        raise RuntimeError(f"V21 selected-query feature width drifted: {sorted(width)}")
    return {
        "features": np.stack(features),
        "returns": np.asarray(returns, dtype=np.float64),
        "queries": np.asarray(queries),
        "selected_sources": np.asarray(selected_sources),
        "oracle_sources": np.asarray(oracle_sources),
        "selected_indices": np.asarray(selected_indices, dtype=np.int64),
        "oracle_indices": np.asarray(oracle_indices, dtype=np.int64),
    }


def _fit_preprocessor(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.sqrt(np.mean(np.square(features), axis=0))
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    z = features / scale
    _u, _s, vh = np.linalg.svd(z, full_matrices=False)
    k = min(SVD_COMPONENTS, max(1, int(features.shape[0]) - 1), int(features.shape[1]), int(vh.shape[0]))
    components = vh[:k]
    return scale, components


def _transform(scale: np.ndarray, components: np.ndarray, features: np.ndarray) -> np.ndarray:
    out = (features / scale) @ components.T
    if not bool(np.isfinite(out).all()):
        raise RuntimeError("V21 transformed selected-query feature contains non-finite values")
    return out


def _decision_metrics(scores: np.ndarray, returns: np.ndarray) -> dict[str, Any]:
    execute = scores < BOUNDARY_ZERO
    hold_truth = returns >= 0.0
    action_truth = ~hold_truth
    n = max(1, int(returns.size))
    fb = float(np.sum(execute & hold_truth) / n)
    fr = float(np.sum((~execute) & action_truth) / n)
    regret = np.where(execute & hold_truth, returns, 0.0)
    regret += np.where((~execute) & action_truth, -returns, 0.0)
    predicted_hold_fraction = float(np.mean(~execute))
    oracle_hold_fraction = float(np.mean(hold_truth))
    return {
        "fb": fb,
        "fr": fr,
        "worst": max(fb, fr),
        "balanced": 0.5 * (fb + fr),
        "mean_selected_vs_hold_regret_m3": float(np.mean(regret)),
        "predicted_hold_fraction": predicted_hold_fraction,
        "oracle_hold_fraction": oracle_hold_fraction,
        "collapse": bool(
            (hold_truth.any() and predicted_hold_fraction == 0.0)
            or (action_truth.any() and predicted_hold_fraction == 1.0)
        ),
    }


def _query_folds(rows: dict[str, Any], seed: int) -> list[np.ndarray]:
    hold = [str(q) for q, value in zip(rows["queries"], rows["returns"]) if float(value) >= 0.0]
    action = [str(q) for q, value in zip(rows["queries"], rows["returns"]) if float(value) < 0.0]
    if min(len(hold), len(action)) < CV_FOLDS:
        raise ValueError("V21 Train needs at least one selected HOLD/ACTION query per OOF fold")
    folds: list[list[str]] = [[] for _ in range(CV_FOLDS)]
    for label, values in (("hold", hold), ("action", action)):
        ordered = sorted(
            values,
            key=lambda query: hashlib.sha256(
                f"v21|{seed}|{label}|{query}".encode("utf-8")
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


def _crossfit(rows: dict[str, Any], seed: int) -> dict[str, Any]:
    folds = _query_folds(rows, seed)
    candidates: list[dict[str, Any]] = []
    for ridge in RIDGE_GRID:
        scores = np.full(rows["returns"].shape, np.nan, dtype=np.float64)
        for held_queries in folds:
            is_test = np.isin(rows["queries"], held_queries)
            is_train = ~is_test
            train_rows = _subset(rows, is_train)
            scale, components = _fit_preprocessor(train_rows["features"])
            z_train = _transform(scale, components, train_rows["features"])
            z_test = _transform(scale, components, rows["features"][is_test])
            weight = _fit_logistic_no_intercept(
                z_train,
                train_rows["returns"],
                train_rows["queries"],
                ridge=float(ridge),
            )
            scores[is_test] = z_test @ weight
        if not bool(np.isfinite(scores).all()):
            raise RuntimeError("V21 OOF selected-query scores are non-finite")
        metrics = _decision_metrics(scores, rows["returns"])
        auc = _auc_hold(scores, rows["returns"])
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
                "score_std": float(np.std(scores)),
                "score_min": float(np.min(scores)),
                "score_max": float(np.max(scores)),
                "selected_metrics": metrics,
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


def _torch_preprocessor(scale: np.ndarray, components: np.ndarray, device: torch.device) -> SelectedBoundaryPreprocessorV21:
    return SelectedBoundaryPreprocessorV21(
        feature_scale=torch.as_tensor(scale, dtype=torch.float32, device=device),
        components=torch.as_tensor(components, dtype=torch.float32, device=device),
    )


def _predict(
    calibrator: SelectedBoundaryCalibratorV21,
    rows: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    advantages: list[float] = []
    device = calibrator.boundary_weight.device
    with torch.no_grad():
        for feature in rows["features"]:
            out = calibrator.predict(
                SelectedBoundaryPartsV21(
                    feature=torch.as_tensor(feature, dtype=torch.float32, device=device)
                )
            )
            scores.append(float(out.hold_score.detach().cpu()))
            advantages.append(float(out.advantage_m3.detach().cpu()))
    return np.asarray(scores, dtype=np.float64), np.asarray(advantages, dtype=np.float64)


def _rank_metrics(all_rows: dict[str, Any]) -> dict[str, Any]:
    pairwise_correct = pairwise_count = top1 = 0
    regrets: list[float] = []
    selected_sources: Counter[str] = Counter()
    oracle_sources: Counter[str] = Counter()
    queries = sorted(set(all_rows["queries"].tolist()))
    for query in queries:
        idx = np.flatnonzero(all_rows["queries"] == query)
        selected_local = np.flatnonzero(all_rows["selected_mask"][idx])
        if selected_local.size != 1:
            raise RuntimeError("V21 rank audit lost one selected candidate per query")
        selected = int(selected_local[0])
        truth = all_rows["returns"][idx]
        rank = all_rows["rank_scores"][idx]
        oracle = int(np.argmin(truth))
        top1 += int(selected == oracle)
        regrets.append(max(0.0, float(truth[selected] - truth[oracle])))
        selected_sources[str(all_rows["sources"][idx[selected]])] += 1
        oracle_sources[str(all_rows["sources"][idx[oracle]])] += 1
        for left in range(len(rank)):
            for right in range(left + 1, len(rank)):
                if abs(float(truth[left]) - float(truth[right])) <= 1.0:
                    continue
                pairwise_count += 1
                pairwise_correct += int(
                    np.sign(rank[left] - rank[right]) == np.sign(truth[left] - truth[right])
                )
    n = max(1, len(queries))
    return {
        "within_query_pairwise_rank_accuracy": pairwise_correct / pairwise_count if pairwise_count else 1.0,
        "within_query_candidate_top1_accuracy": top1 / n,
        "mean_ranking_regret_before_hold_m3": float(np.mean(regrets)),
        "selected_candidate_source_counts": dict(sorted(selected_sources.items())),
        "oracle_best_candidate_source_counts": dict(sorted(oracle_sources.items())),
    }


def _evaluate(
    all_rows: dict[str, Any],
    selected_rows: dict[str, Any],
    calibrator: SelectedBoundaryCalibratorV21,
) -> dict[str, Any]:
    scores, advantages = _predict(calibrator, selected_rows)
    truth = selected_rows["returns"]
    execute = scores < BOUNDARY_ZERO
    oracle_hold = truth >= 0.0
    fb = float(np.mean(execute & oracle_hold))
    fr = float(np.mean((~execute) & (~oracle_hold)))
    regret = np.where(execute & oracle_hold, truth, 0.0)
    regret += np.where((~execute) & (~oracle_hold), -truth, 0.0)
    rank = _rank_metrics(all_rows)
    predicted_hold_fraction = float(np.mean(~execute))
    oracle_hold_fraction = float(np.mean(oracle_hold))
    metrics = {
        "query_count": int(truth.size),
        "candidate_record_count": int(all_rows["returns"].size),
        "selected_action_false_beneficial_fraction": fb,
        "selected_action_false_reject_fraction": fr,
        "selected_action_worst_side_error_fraction": max(fb, fr),
        "selected_action_balanced_error_fraction": 0.5 * (fb + fr),
        "hold_aware_decision_accuracy": float(np.mean(execute == (~oracle_hold))),
        "selected_action_mean_regret_m3": float(np.mean(regret)),
        "predicted_hold_fraction": predicted_hold_fraction,
        "oracle_hold_optimal_fraction": oracle_hold_fraction,
        "execute_all_collapse": bool(oracle_hold.any() and predicted_hold_fraction == 0.0),
        "hold_all_collapse": bool((~oracle_hold).any() and predicted_hold_fraction == 1.0),
        "boundary_auc_hold_vs_action": _auc_hold(scores, truth),
        "boundary_score_std": float(np.std(scores)),
        "boundary_score_unique_count_1e8": int(np.unique(np.round(scores, 8)).size),
        "boundary_score": {
            "min": float(np.min(scores)),
            "q25": float(np.quantile(scores, 0.25)),
            "median": float(np.median(scores)),
            "q75": float(np.quantile(scores, 0.75)),
            "max": float(np.max(scores)),
        },
        "decision_threshold": BOUNDARY_ZERO,
        "selected_return_mae_m3": float(np.mean(np.abs(advantages - truth))),
    }
    metrics.update(rank)
    return metrics


def _save_report(payload: dict[str, Any], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    report = copy.deepcopy(payload)
    for key in (
        "rank_source_state_dict",
        "feature_scale",
        "svd_components",
        "boundary_weight",
        "magnitude_weight",
    ):
        report.pop(key, None)
    report["checkpoint_sha256"] = sha256_file(out)
    out.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


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
        raise ValueError("V21 Train/Validation rainfall groups overlap")
    if train_meta["groups"] & calibration_meta["groups"]:
        raise ValueError("V21 Train/Calibration rainfall groups overlap")
    if validation_meta["groups"] & calibration_meta["groups"]:
        raise ValueError("V21 Validation/Calibration rainfall groups overlap")
    if len({
        train_meta["continuation_policy_sha256"],
        validation_meta["continuation_policy_sha256"],
        calibration_meta["continuation_policy_sha256"],
    }) != 1:
        raise ValueError("V21 splits do not share one frozen continuation policy")

    train = _load_dataset(args.train_dataset, role="policy_return_train")
    graph = _load_graph(args.graph)
    control, mask = load_native_supervisory_control(
        args.supervisory_control,
        actuator_ids=graph.actuator_ids,
    )
    if str(control["supervisory_mask_sha256"]).lower() != train_meta["supervisory_mask_sha256"]:
        raise ValueError("V21 supervisory-control artifact differs from truth lineage")

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

    train_all = _collect_rows(
        rank_model,
        rank_adapter,
        train,
        graph=graph,
        normalization=normalization,
        mask=mask,
        device=device,
        scale=scale,
    )
    train_selected = _selected_query_rows(train_all, _action_masses(train, mask))
    if int(train_selected["returns"].size) != int(train_all["selected_mask"].sum()):
        raise RuntimeError("V21 training unit must be exactly one selected row per query")

    oof = _crossfit(train_selected, args.seed)
    selected_oof = oof["selected"]
    train_oof_supported = bool(
        float(selected_oof["auc"]) >= TRAIN_OOF_AUC_MIN
        and not bool(selected_oof["selected_metrics"]["collapse"])
        and float(selected_oof["score_std"]) > 1.0e-6
    )

    feature_scale, components = _fit_preprocessor(train_selected["features"])
    z_train = _transform(feature_scale, components, train_selected["features"])
    ridge = float(selected_oof["ridge"])
    boundary_weight = _fit_logistic_no_intercept(
        z_train,
        train_selected["returns"],
        train_selected["queries"],
        ridge=ridge,
    )
    magnitude_weight = _fit_magnitude_no_intercept(
        z_train,
        train_selected["returns"],
        train_selected["queries"],
        scale=scale,
        ridge=ridge,
    )
    calibrator = SelectedBoundaryCalibratorV21(
        preprocessor=_torch_preprocessor(feature_scale, components, device),
        boundary_weight=torch.as_tensor(boundary_weight, dtype=torch.float32, device=device),
        magnitude_weight=torch.as_tensor(magnitude_weight, dtype=torch.float32, device=device),
        target_scale_m3=scale,
    ).to(device)
    calibrator.eval()

    rank_source_state = rank_source.get("query_margin_state_dict")
    if not isinstance(rank_source_state, dict):
        raise ValueError("V21 V15 rank source lacks query_margin_state_dict")
    frozen_rank_state = {
        key: value.detach().cpu()
        for key, value in rank_source_state.items()
        if key.startswith("context_encoder.")
        or key.startswith("candidate_encoder.")
        or key.startswith("rank_adjustment.")
    }

    base_payload: dict[str, Any] = {
        "contract": DIRECT_TFV_SELECTED_BOUNDARY_V21_CHECKPOINT_CONTRACT,
        "training_contract": STEP3_V21_TRAINING_CONTRACT,
        "boundary_contract": DIRECT_TFV_SELECTED_BOUNDARY_V21_CONTRACT,
        "feature_contract": DIRECT_TFV_SELECTED_BOUNDARY_V21_FEATURE_CONTRACT,
        "development_only": True,
        "estimand": DIRECT_TFV_POLICY_RETURN_ESTIMAND,
        "action_encoding_contract": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
        "base_step2_sha256": sha256_file(args.base_step2),
        "base_step2_frozen": True,
        "base_step2_state_sha256_before_training": step2_before,
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
        "rank_source_state_dict": frozen_rank_state,
        "rank_reused_from_v15": True,
        "rank_retrained_in_v21": False,
        "rank_branch_state_sha256_before_boundary": rank_sha_before,
        "target_scale_m3": scale,
        "train_candidate_record_count_available": int(train_all["returns"].size),
        "train_selected_query_count": int(train_selected["returns"].size),
        "boundary_supervision_unit": "FROZEN_RANK_SELECTED_CANDIDATE_PER_QUERY",
        "sibling_truth_used_for_boundary_fit": False,
        "sibling_candidate_features_used_as_online_portfolio_context": True,
        "selected_query_feature_width": int(train_selected["features"].shape[1]),
        "svd_components": components.astype(np.float32),
        "svd_component_count": int(components.shape[0]),
        "svd_centered": False,
        "feature_scale": feature_scale.astype(np.float32),
        "boundary_weight": boundary_weight.astype(np.float32),
        "magnitude_weight": magnitude_weight.astype(np.float32),
        "decision_threshold": BOUNDARY_ZERO,
        "decision_threshold_tuned": False,
        "boundary_has_intercept": False,
        "ridge_grid": list(RIDGE_GRID),
        "boundary_cv_folds": CV_FOLDS,
        "boundary_oof_selection": oof,
        "train_oof_boundary_supported": train_oof_supported,
        "train_oof_auc_min": TRAIN_OOF_AUC_MIN,
        "validation_truth_loaded": False,
        "validation_metrics": None,
        "development_acceptance": DEVELOPMENT_ACCEPTANCE,
        "development_validation_pass": False,
        "calibration_truth_loaded": False,
        "calibration_raw_metrics": None,
        "online_swmm_called": False,
        "new_swmm_runs": 0,
        "ready_for_pi1_development": False,
        "ready_for_policy_lock": False,
    }

    out = Path(args.out)
    if not train_oof_supported:
        rank_sha_after = rank_state_sha256(rank_adapter)
        step2_after = _model_state_sha256(rank_model)
        if rank_sha_before != rank_sha_after or step2_before != step2_after:
            raise RuntimeError("V21 fail-closed path mutated frozen Step2/rank")
        base_payload["rank_branch_state_sha256_after_boundary"] = rank_sha_after
        base_payload["base_step2_state_sha256_after_training"] = step2_after
        _save_report(base_payload, out)
        raise RuntimeError("V21 selected-query boundary lacks Train-OOF sign support")

    validation = _load_dataset(args.validation_dataset, role="policy_return_validation")
    validation_all = _collect_rows(
        rank_model,
        rank_adapter,
        validation,
        graph=graph,
        normalization=normalization,
        mask=mask,
        device=device,
        scale=scale,
    )
    validation_selected = _selected_query_rows(validation_all, _action_masses(validation, mask))
    validation_metrics = _evaluate(validation_all, validation_selected, calibrator)
    validation_pass = bool(_accepted(validation_metrics))
    base_payload["validation_truth_loaded"] = True
    base_payload["validation_metrics"] = validation_metrics
    base_payload["development_validation_pass"] = validation_pass

    if validation_pass:
        calibration = _load_dataset(args.calibration_dataset, role="policy_return_calibration")
        calibration_all = _collect_rows(
            rank_model,
            rank_adapter,
            calibration,
            graph=graph,
            normalization=normalization,
            mask=mask,
            device=device,
            scale=scale,
        )
        calibration_selected = _selected_query_rows(calibration_all, _action_masses(calibration, mask))
        base_payload["calibration_truth_loaded"] = True
        base_payload["calibration_raw_metrics"] = _evaluate(
            calibration_all, calibration_selected, calibrator
        )

    rank_sha_after = rank_state_sha256(rank_adapter)
    step2_after = _model_state_sha256(rank_model)
    if rank_sha_before != rank_sha_after:
        raise RuntimeError("V21 fitting mutated frozen V15 rank")
    if step2_before != step2_after:
        raise RuntimeError("V21 fitting mutated frozen Step2")
    base_payload["rank_branch_state_sha256_after_boundary"] = rank_sha_after
    base_payload["base_step2_state_sha256_after_training"] = step2_after
    _save_report(base_payload, out)

    if not validation_pass:
        raise RuntimeError("V21 selected-query boundary failed one-shot Validation; Calibration stayed sealed")


if __name__ == "__main__":
    main()
