from __future__ import annotations

from types import SimpleNamespace

import torch

from rtc.controller_direct_tfv import DirectTFVRuntimeMPCAdapter


class _Inner:
    def __init__(self) -> None:
        self.design = SimpleNamespace(max_setting_delta_per_update=0.5)
        self.model = torch.nn.Linear(1, 1)
        self.called = None

    def optimize(self, **kwargs):
        self.called = kwargs
        return SimpleNamespace(
            settings=torch.full((72, 109), 0.5),
            predicted_delta_tfv_m3=-1234.0,
            selected_source="DIRECT_TFV_RECEDING_LBFGSB",
            optimizer_success=True,
            optimizer_steps=4,
            optimizer_starts=2,
            gradient_norm=3.0,
            elapsed_seconds=0.2,
            screened_facility_count=109,
            predicted_beneficial_facility_count=17,
            active_facility_count=17,
            first_move_changed_facility_count=5,
            maximum_support_ratio=0.8,
            scipy_message="ok",
        )


def test_runtime_adapter_exposes_model_required_by_controller_base() -> None:
    inner = _Inner()
    adapter = DirectTFVRuntimeMPCAdapter(inner)  # type: ignore[arg-type]
    assert adapter.model is inner.model
    adapter.model.to(torch.device("cpu")).eval()
    assert adapter.model.training is False


def test_runtime_adapter_maps_target_latch_call_to_direct_step3() -> None:
    inner = _Inner()
    adapter = DirectTFVRuntimeMPCAdapter(inner)  # type: ignore[arg-type]
    initial = torch.zeros((1, 8, 3))
    rain = torch.zeros((1, 72, 8, 1))
    current = torch.full((109,), 0.4)
    target = torch.full((109,), 0.5)
    flow = torch.zeros((1, 109))
    result = adapter.optimize(
        initial_state=initial,
        rainfall_scenarios=rain,
        current_settings=current,
        previous_requested_settings=target,
        fallback_settings=torch.full((1, 72, 109), 0.5),
        previous_actuator_flow=flow,
        max_delta_per_update=0.5,
    )
    assert result.candidate_valid is True
    assert result.screened_facility_count == 109
    assert result.predicted_beneficial_facility_count == 17
    assert result.first_move_changed_facility_count == 5
    assert adapter.last_result is result
    assert inner.called is not None
    assert inner.called["active_target"] is target
    assert inner.called["rainfall"] is rain


def test_runtime_adapter_rejects_rate_contract_drift() -> None:
    adapter = DirectTFVRuntimeMPCAdapter(_Inner())  # type: ignore[arg-type]
    try:
        adapter.optimize(
            initial_state=torch.zeros((1, 8, 3)),
            rainfall_scenarios=torch.zeros((1, 72, 8, 1)),
            current_settings=torch.zeros(109),
            previous_requested_settings=torch.zeros(109),
            fallback_settings=torch.zeros((1, 72, 109)),
            previous_actuator_flow=torch.zeros((1, 109)),
            max_delta_per_update=0.25,
        )
    except ValueError as exc:
        assert "max-delta" in str(exc)
    else:
        raise AssertionError("runtime adapter accepted inconsistent max-delta contract")
