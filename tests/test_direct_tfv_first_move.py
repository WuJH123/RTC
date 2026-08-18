from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from rtc.direct_tfv_first_move import DIRECT_TFV_FIRST_MOVE_SEMANTICS, refine_supported_first_move
from rtc.direct_tfv_first_move_admission import (
    DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT,
    DIRECT_TFV_FIRST_MOVE_MIN_CALIBRATION_GROUPS,
    DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT,
    DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT,
    DIRECT_TFV_FIRST_MOVE_SCALE,
    derive_first_move_admission,
    first_move_margin_m3,
)


@dataclass(frozen=True)
class _Design:
    prediction_horizon_steps: int = 72
    control_block_steps: int = 2


class _Graph:
    actuator_ids = tuple(f"A{i:03d}" for i in range(109))


class _DummyMPC:
    design = _Design()
    graph = _Graph()

    @staticmethod
    def _hold_sequence(active_target: torch.Tensor) -> torch.Tensor:
        return active_target[None].expand(72, -1).clone()

    @staticmethod
    def _contract_to_joint_sequence_support(
        sequence: torch.Tensor, active_target: torch.Tensor
    ) -> torch.Tensor:
        # The unit test only verifies the refinement cannot expand the upstream direction.
        return sequence

    @staticmethod
    def score_sequence(
        *,
        current_state: torch.Tensor,
        rainfall: torch.Tensor,
        sequence: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        active_target: torch.Tensor,
    ) -> torch.Tensor:
        del current_state, rainfall, previous_actuator_flow
        first = sequence[:2].mean(dim=0)
        # Minimum is at half of the upstream first move for A000/A001/A002.
        desired = active_target.clone()
        desired[:3] = 0.25
        return torch.sum((first - desired) ** 2) - 1.0


def test_refiner_changes_only_h10_and_never_expands_upstream_first_move() -> None:
    mpc = _DummyMPC()
    active = torch.zeros(109, dtype=torch.float32)
    base = active[None].expand(72, -1).clone()
    base[:2, :3] = 0.5
    refined = refine_supported_first_move(
        mpc=mpc,
        base_candidate=base,
        current_state=torch.zeros((1, 1, 1)),
        rainfall=torch.zeros((1, 1, 1, 1)),
        previous_actuator_flow=torch.zeros((1, 109)),
        active_target=active,
        maxiter=20,
        deadline_seconds=10.0,
    )
    first = refined.sequence[:2].mean(dim=0)
    assert refined.changed_facility_count == 3
    assert torch.all(first[:3] >= -1.0e-7)
    assert torch.all(first[:3] <= 0.5 + 1.0e-7)
    assert torch.allclose(refined.sequence[2:], torch.zeros_like(refined.sequence[2:]))
    assert refined.predicted_delta_tfv_m3 <= refined.base_prefix_predicted_delta_tfv_m3 + 1.0e-6
    assert refined.gain_vs_base_prefix_m3 >= -1.0e-6


def _records(count: int) -> list[dict[str, object]]:
    return [
        {
            "rainfall_group": f"R{i:02d}",
            "plan_sha256": f"sha-{i:02d}",
            "predicted_refined_delta_tfv_m3": -100.0,
            "true_refined_delta_tfv_m3": -80.0 + float(i % 3),
            "first_move_changed_facility_count": 4 + (i % 5),
        }
        for i in range(count)
    ]


def test_normalized_admission_requires_24_independent_rainfall_groups() -> None:
    records = _records(DIRECT_TFV_FIRST_MOVE_MIN_CALIBRATION_GROUPS - 1)
    with pytest.raises(ValueError, match="at least 24 fresh rainfall groups"):
        derive_first_move_admission(
            panel_contract=DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT,
            panel_step3_contract=DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT,
            panel_records=records,
            expected_rainfall_groups=[str(row["rainfall_group"]) for row in records],
            coverage=0.90,
        )


def test_normalized_margin_scales_with_sqrt_actual_first_move_density() -> None:
    records = _records(DIRECT_TFV_FIRST_MOVE_MIN_CALIBRATION_GROUPS)
    calibration = derive_first_move_admission(
        panel_contract=DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT,
        panel_step3_contract=DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT,
        panel_records=records,
        expected_rainfall_groups=[str(row["rainfall_group"]) for row in records],
        coverage=0.90,
    )
    assert calibration["contract"] == DIRECT_TFV_FIRST_MOVE_ADMISSION_CONTRACT
    assert calibration["execution_estimand"] == DIRECT_TFV_FIRST_MOVE_SEMANTICS
    assert calibration["normalization"] == DIRECT_TFV_FIRST_MOVE_SCALE
    assert calibration["generic_d3_floor_controls_execution"] is False
    assert calibration["v9_full_plan_margin_controls_execution"] is False
    assert calibration["v10_prefix_margin_controls_execution"] is False
    margin4 = first_move_margin_m3(calibration, 4)
    margin16 = first_move_margin_m3(calibration, 16)
    assert margin16 == pytest.approx(2.0 * margin4)


def test_first_move_margin_rejects_zero_changed_facilities() -> None:
    records = _records(DIRECT_TFV_FIRST_MOVE_MIN_CALIBRATION_GROUPS)
    calibration = derive_first_move_admission(
        panel_contract=DIRECT_TFV_FIRST_MOVE_PANEL_CONTRACT,
        panel_step3_contract=DIRECT_TFV_FIRST_MOVE_QUERY_STEP3_CONTRACT,
        panel_records=records,
        expected_rainfall_groups=[str(row["rainfall_group"]) for row in records],
    )
    with pytest.raises(ValueError, match="changed facility count"):
        first_move_margin_m3(calibration, 0)
