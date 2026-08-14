from __future__ import annotations

import numpy as np
import pytest

from rtc.closed_loop import ControllerAction
from rtc.controller_v122 import V122TorchMPCController
from rtc.controller_v123 import V123TorchMPCController
from rtc.step2_acceptance_v123 import evaluate_continuous_value_gate_v123
from rtc.step2_causal_rainfall_v123 import CausalForecastStoreV123


_SHA_A = "a" * 64
_SHA_B = "b" * 64


def _store(**overrides) -> CausalForecastStoreV123:
    values = {
        "group_names": ("D3::rain::event::cp",),
        "event_ids": ("event",),
        "checkpoint_ids": ("cp",),
        "forecast_mmhr": np.zeros((1, 72, 3, 1), dtype=np.float32),
        "history_sha256": (_SHA_A,),
        "forecast_sha256": (_SHA_B,),
        "forecast_contract": "PERSISTENCE_DECAY_RUNTIME_V1",
        "future_realized_rainfall_not_used": True,
    }
    values.update(overrides)
    return CausalForecastStoreV123(**values)


def test_causal_store_requires_future_rainfall_exclusion_proof() -> None:
    with pytest.raises(ValueError, match="future realised rainfall exclusion"):
        _store(future_realized_rainfall_not_used=False).validate()


def test_causal_store_requires_canonical_history_and_forecast_sha() -> None:
    with pytest.raises(ValueError, match="history hashes"):
        _store(history_sha256=("not-a-sha",)).validate()
    with pytest.raises(ValueError, match="forecast hashes"):
        _store(forecast_sha256=("not-a-sha",)).validate()


def test_continuous_gradient_gate_blocks_current_causal_value_level() -> None:
    report = evaluate_continuous_value_gate_v123(
        causal_input_verified=True,
        holdout_rank=0.4142,
        holdout_top1=0.5625,
        gradient_sign_accuracy=0.60,
        gradient_cosine=0.40,
    )
    assert report["continuous_gradient_search"] is False
    assert report["verdict"] == "V123_GRADIENT_BLOCKED_FINITE_ONLY"


def test_continuous_gradient_gate_passes_only_all_thresholds() -> None:
    report = evaluate_continuous_value_gate_v123(
        causal_input_verified=True,
        holdout_rank=0.71,
        holdout_top1=0.55,
        gradient_sign_accuracy=0.72,
        gradient_cosine=0.62,
    )
    assert report["continuous_gradient_search"] is True


def test_v123_controller_relabels_successful_v122_shell_action(monkeypatch) -> None:
    def fake_decide(self, obs, *, observation_already_recorded=False):
        return ControllerAction(
            settings={"a": 0.5},
            source="MPC_V122",
            diagnostics={"candidate_count": 169, "selected_group_score_m3": -10.0},
        )

    monkeypatch.setattr(V122TorchMPCController, "decide", fake_decide)
    controller = object.__new__(V123TorchMPCController)
    action = controller.decide(object())
    assert action.source == "MPC_V123"
    assert action.diagnostics["candidate_count"] == 169
    assert action.diagnostics["selected_group_score_m3"] == -10.0
    assert "v123_controller_contract" in action.diagnostics
