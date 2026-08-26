from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rtc.direct_tfv_operational_v26_runtime import (
    OPERATIONAL_V26_RUNTIME_CONTRACT,
    V26_SELECTION_CONTRACT,
)
from rtc.direct_tfv_v26_value_model import (
    FittedV26ValueModel,
    candidate_metrics,
    checkpoint_payload,
    decision_metrics,
    fit_v26_value_model,
    load_v26_value_model,
)


def test_v26_decision_metrics_select_min_candidate_against_hold_zero() -> None:
    prediction = np.asarray([-10.0, 5.0, -1.0, 2.0], dtype=np.float64)
    truth = np.asarray([-8.0, -20.0, 4.0, -3.0], dtype=np.float64)
    queries = ["q1", "q1", "q2", "q2"]
    metrics = decision_metrics(prediction, truth, queries)
    assert metrics["query_count"] == 2
    assert metrics["action_count"] == 2
    assert metrics["beneficial_action_count"] == 1
    assert metrics["harmful_action_count"] == 1
    assert metrics["sum_selected_true_delta_tfv_m3"] == -4.0


def test_v26_fit_uses_validation_for_model_selection_without_support_gate() -> None:
    x_train = np.asarray(
        [[-2.0, 0.0], [-1.0, 0.2], [1.0, 0.8], [2.0, 1.0], [-1.5, 0.1], [1.5, 0.9]],
        dtype=np.float64,
    )
    y_train = np.asarray([-20.0, -10.0, 10.0, 20.0, -15.0, 15.0], dtype=np.float64)
    x_val = np.asarray([[-1.2, 0.1], [1.2, 0.9], [-0.5, 0.3], [0.5, 0.7]], dtype=np.float64)
    y_val = np.asarray([-12.0, 12.0, -5.0, 5.0], dtype=np.float64)
    model, report = fit_v26_value_model(
        x_train,
        y_train,
        x_val,
        y_val,
        ["a", "b", "c", "d"],
    )
    pred = model.predict_numpy(x_val)
    assert pred.shape == (4,)
    assert np.isfinite(pred).all()
    assert report["scientific_metrics_block_runtime"] is False
    assert report["selection_rule"] == "MIN_VALIDATION_REALIZED_EXACT_TFV_THEN_RMSE_THEN_RIDGE"


def test_v26_checkpoint_load_does_not_require_perfect_validation_metrics(tmp_path: Path) -> None:
    fitted = FittedV26ValueModel(
        feature_mean=np.zeros(2, dtype=np.float64),
        feature_scale=np.ones(2, dtype=np.float64),
        weight=np.asarray([1.0, -1.0], dtype=np.float64),
        intercept=0.0,
        target_scale_m3=100.0,
        ridge=1.0,
    )
    lineage = {"base_step2_sha256": "a" * 64, "feature_contract": "feature-v26"}
    payload = checkpoint_payload(
        fitted,
        lineage=lineage,
        training_report={
            "validation_decision_metrics": {"harmful_action_count": 7},
            "scientific_metrics_block_runtime": False,
        },
        test_report={"harmful_action_count": 9},
    )
    path = tmp_path / "v26.pt"
    torch.save(payload, path)
    model, loaded = load_v26_value_model(
        str(path),
        device=torch.device("cpu"),
        expected_lineage=lineage,
    )
    value = model.predict(torch.as_tensor([0.25, 0.0], dtype=torch.float32))
    assert torch.isfinite(value)
    assert loaded["training_report"]["validation_decision_metrics"]["harmful_action_count"] == 7


def test_v26_candidate_metrics_are_reporting_only() -> None:
    metrics = candidate_metrics(
        np.asarray([-1.0, -2.0, 3.0]),
        np.asarray([2.0, -3.0, 4.0]),
    )
    assert metrics["count"] == 3
    assert metrics["predicted_action_count"] == 2
    assert "auc_beneficial_vs_nonbeneficial" in metrics


def test_v26_contract_names_make_direct_value_path_explicit() -> None:
    assert "DIRECT_EXACT_RETURN" in OPERATIONAL_V26_RUNTIME_CONTRACT
    assert "MIN_PREDICTED_EXACT_RETURN" in V26_SELECTION_CONTRACT
