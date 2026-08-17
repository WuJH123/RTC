from __future__ import annotations

from types import SimpleNamespace

import torch

from rtc.controller_direct_tfv import DirectTFVRuntimeMPCAdapter
from rtc.direct_tfv_admission import DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT


class _Inner:
    def __init__(self) -> None:
        self.design = SimpleNamespace(
            max_setting_delta_per_update=0.5,
            control_block_steps=2,
            free_control_blocks=12,
            active_facility_count=0,
            active_support_quantile="q95",
        )
        self.graph = SimpleNamespace(actuator_ids=tuple(f"A{i:03d}" for i in range(109)))
        self.model = torch.nn.Linear(1, 1)
        self.called = None
        self.fail = False

    def active_support_quantile_effective(self) -> str:
        return "q95"

    def active_support_ceiling(self) -> int:
        return 23

    def optimize(self, **kwargs):
        self.called = kwargs
        if self.fail:
            raise RuntimeError("synthetic solve failure")
        settings = torch.full((72, 109), 0.5)
        settings[:24, :5] = 0.6
        return SimpleNamespace(
            settings=settings,
            optimized_candidate_settings=settings.clone(),
            predicted_delta_tfv_m3=-1234.0,
            raw_optimized_predicted_delta_tfv_m3=-6234.0,
            admission_margin_m3=5000.0,
            admission_upper_bound_m3=-1234.0,
            admission_margin_kind="dense",
            admission_passed=True,
            calibrated_admission_contract=DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT,
            selected_source="DIRECT_TFV_RECEDING_LBFGSB",
            optimizer_success=True,
            optimizer_steps=4,
            optimizer_starts=2,
            gradient_norm=3.0,
            elapsed_seconds=0.2,
            screened_facility_count=109,
            predicted_beneficial_facility_count=30,
            active_facility_count=23,
            active_facility_ids=tuple(f"A{i:03d}" for i in range(23)),
            active_facility_screening_scores_m3=tuple(-1000.0 + i for i in range(23)),
            first_move_changed_facility_count=5,
            maximum_support_ratio=0.8,
            training_joint_changed_facility_q90=20.0,
            scipy_message="ok",
        )


def _optimize(adapter: DirectTFVRuntimeMPCAdapter):
    return adapter.optimize(
        initial_state=torch.zeros((1, 8, 3)),
        rainfall_scenarios=torch.zeros((1, 72, 8, 1)),
        current_settings=torch.full((109,), 0.4),
        previous_requested_settings=torch.full((109,), 0.5),
        fallback_settings=torch.full((1, 72, 109), 0.5),
        previous_actuator_flow=torch.zeros((1, 109)),
        max_delta_per_update=0.5,
    )


def test_runtime_adapter_exposes_model_required_by_controller_base() -> None:
    inner = _Inner()
    adapter = DirectTFVRuntimeMPCAdapter(inner)  # type: ignore[arg-type]
    assert adapter.model is inner.model
    adapter.model.to(torch.device("cpu")).eval()
    assert adapter.model.training is False


def test_runtime_adapter_maps_calibrated_step3_and_preserves_raw_plan() -> None:
    inner = _Inner()
    adapter = DirectTFVRuntimeMPCAdapter(inner)  # type: ignore[arg-type]
    result = _optimize(adapter)
    assert result.candidate_valid is True
    assert result.screened_facility_count == 109
    assert result.predicted_beneficial_facility_count == 30
    assert result.first_move_changed_facility_count == 5
    assert result.best_screening_predicted_delta_tfv_m3 == -1000.0
    assert result.raw_optimized_predicted_delta_tfv_m3 == -6234.0
    assert result.admission_margin_m3 == 5000.0
    assert result.admission_upper_bound_m3 == -1234.0
    assert result.admission_margin_kind == "dense"
    assert result.admission_passed is True
    assert result.calibrated_admission_contract == DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT
    assert result.active_support_quantile_effective == "q95"
    assert result.active_support_ceiling == 23
    assert result.active_set_ceiling_binding is True
    assert result.counterfactual_actuator_ids == tuple(f"A{i:03d}" for i in range(109))
    assert len(result.optimized_free_control_blocks) == 12
    assert all(len(block) == 109 for block in result.optimized_free_control_blocks)
    assert result.optimized_free_control_blocks[0][0] == 0.6000000238418579
    assert result.hold_reference_settings == (0.5,) * 109
    assert adapter.last_result is result
    assert inner.called is not None
    assert torch.allclose(inner.called["active_target"], torch.full((109,), 0.5))
    assert inner.called["rainfall"].shape == (1, 72, 8, 1)


def test_runtime_adapter_clears_stale_result_before_failed_solve() -> None:
    inner = _Inner()
    adapter = DirectTFVRuntimeMPCAdapter(inner)  # type: ignore[arg-type]
    first = _optimize(adapter)
    assert adapter.last_result is first
    inner.fail = True
    try:
        _optimize(adapter)
    except RuntimeError as exc:
        assert "synthetic solve failure" in str(exc)
    else:
        raise AssertionError("synthetic solve failure was not propagated")
    assert adapter.last_result is None


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
