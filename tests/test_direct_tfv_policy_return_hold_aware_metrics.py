from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_direct_tfv_policy_return_current.py"


def _module():
    spec = importlib.util.spec_from_file_location("train_policy_return_hold_aware", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_all_harmful_candidates_are_correctly_solved_by_hold() -> None:
    metrics = _module()._hold_aware_query_decision_metrics(
        prediction=[100.0, 200.0, 300.0],
        truth=[10.0, 20.0, 30.0],
    )
    assert metrics["predicted_hold"] is True
    assert metrics["oracle_hold"] is True
    assert metrics["false_beneficial"] is False
    assert metrics["false_reject"] is False
    assert metrics["decision_correct"] is True
    assert metrics["regret_m3"] == pytest.approx(0.0)


def test_holding_when_a_beneficial_candidate_exists_is_false_reject() -> None:
    metrics = _module()._hold_aware_query_decision_metrics(
        prediction=[1.0, 2.0, 3.0],
        truth=[-10.0, 4.0, 8.0],
    )
    assert metrics["predicted_hold"] is True
    assert metrics["oracle_hold"] is False
    assert metrics["false_reject"] is True
    assert metrics["false_beneficial"] is False
    assert metrics["regret_m3"] == pytest.approx(10.0)


def test_executing_predicted_beneficial_but_truly_harmful_candidate_is_false_beneficial() -> None:
    metrics = _module()._hold_aware_query_decision_metrics(
        prediction=[-5.0, 1.0, 2.0],
        truth=[12.0, 20.0, 30.0],
    )
    assert metrics["predicted_hold"] is False
    assert metrics["oracle_hold"] is True
    assert metrics["false_beneficial"] is True
    assert metrics["regret_m3"] == pytest.approx(12.0)


def test_correct_beneficial_candidate_has_zero_hold_aware_regret() -> None:
    metrics = _module()._hold_aware_query_decision_metrics(
        prediction=[-4.0, -2.0, 1.0],
        truth=[-9.0, -3.0, 6.0],
    )
    assert metrics["predicted_execute"] is True
    assert metrics["oracle_execute"] is True
    assert metrics["selected_candidate_index"] == 0
    assert metrics["oracle_candidate_index"] == 0
    assert metrics["decision_correct"] is True
    assert metrics["regret_m3"] == pytest.approx(0.0)
