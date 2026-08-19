from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.actuator_release_semantics import (
    graph_release_setting_signs,
    release_fraction_to_setting,
    release_setting_sign,
)
from rtc.direct_tfv_policy_return_portfolio import (
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
    build_policy_return_candidate_portfolio,
    hydraulic_pressure_setting_delta,
)
from rtc.direct_tfv_policy_return_portfolio_admission import (
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_ADMISSION_CONTRACT,
    derive_policy_return_portfolio_admission,
)


def _graph() -> SimpleNamespace:
    actuator_ids = tuple(f"A{i:03d}" for i in range(109))
    names = (
        "min_setting",
        "max_setting",
        "is_pump",
        "is_orifice",
        "is_weir",
        "is_outlet",
    )
    physics = np.zeros((109, len(names)), dtype=np.float32)
    physics[:, 1] = 1.0
    for i in range(109):
        physics[i, 2 + (i % 4)] = 1.0
    return SimpleNamespace(
        actuator_ids=actuator_ids,
        actuator_physics_feature_names=names,
        actuator_physics=physics,
        actuator_upstream=np.zeros(109, dtype=np.int64),
        actuator_downstream=np.ones(109, dtype=np.int64),
        node_ids=("UP", "DOWN"),
        static_node_feature_names=("max_depth_m", "storage_capacity_m3"),
        static_node_features=np.asarray([[2.0, 1000.0], [2.0, 1000.0]], dtype=np.float32),
    )


def test_swmm_release_setting_direction_is_type_aware() -> None:
    assert release_setting_sign("pump") == 1.0
    assert release_setting_sign("orifice") == 1.0
    assert release_setting_sign("outlet") == 1.0
    assert release_setting_sign("weir") == -1.0
    assert release_fraction_to_setting("pump", 0.8) == 0.8
    assert abs(release_fraction_to_setting("weir", 0.8) - 0.2) < 1.0e-12

    signs = graph_release_setting_signs(_graph())
    assert tuple(signs[:4]) == (1.0, 1.0, -1.0, 1.0)


def test_hydraulic_pressure_candidate_reverses_weir_setting_direction() -> None:
    graph = _graph()
    # Upstream is almost full and downstream has headroom, so the hydraulic release intent is positive.
    state = np.asarray(
        [
            [1.8, 0.0, 0.0, 900.0],
            [0.2, 0.0, 0.0, 100.0],
        ],
        dtype=np.float32,
    )
    rainfall = np.ones((3, 72, 2, 1), dtype=np.float32)
    delta = hydraulic_pressure_setting_delta(
        current_state=state,
        rainfall_scenarios=rainfall,
        graph=graph,
        max_delta_per_update=0.5,
    )
    assert delta[0] > 0.0  # pump: release increase -> higher setting
    assert delta[1] > 0.0  # orifice
    assert delta[2] < 0.0  # weir: release increase -> lower freeboard setting
    assert delta[3] > 0.0  # outlet


def test_candidate_portfolio_is_supported_bounded_and_not_baseline_imitation() -> None:
    graph = _graph()
    state = torch.tensor(
        [[[1.8, 0.0, 0.0, 900.0], [0.2, 0.0, 0.0, 100.0]]],
        dtype=torch.float32,
    )
    rainfall = torch.ones((3, 72, 2, 1), dtype=torch.float32)
    active = torch.full((109,), 0.5, dtype=torch.float32)
    learned = active.clone()
    learned[:6] += 0.2
    candidates = build_policy_return_candidate_portfolio(
        current_state=state,
        rainfall_scenarios=rainfall,
        active_target=active,
        v12_target=learned,
        graph=graph,
        first_radius=np.full(109, 0.15, dtype=np.float32),
        max_changed_facilities=4,
        max_delta_per_update=0.5,
    )
    sources = {row.source for row in candidates}
    assert "TYPE_AWARE_HYDRAULIC_PRESSURE" in sources
    assert "V12_DIRECTION_SCALE_1.00" in sources
    assert not any("AUTO_RBC" in value or "ALL_OPEN" in value for value in sources)
    for row in candidates:
        delta = torch.abs(row.target - active)
        assert row.changed_facility_count <= 4
        assert float(delta.max()) <= 0.150001
        assert bool(torch.all((row.target >= 0.0) & (row.target <= 1.0)))


def _calibration_record(group: str, source: str, truth: float, prediction: float) -> dict:
    query = (group.encode("utf-8").hex() + "0" * 64)[:64]
    return {
        "estimand": "EXECUTE_CANDIDATE_H10_THEN_FROZEN_POLICY_VS_HOLD_H10_THEN_SAME_FROZEN_POLICY",
        "data_role": "policy_return_calibration",
        "rainfall_group": group,
        "event_id": f"event-{group}",
        "decision_index": 1,
        "decision_elapsed_seconds": 4200,
        "query_set_id": query,
        "candidate_source": source,
        "candidate_portfolio_contract": DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
        "first_move_changed_facility_count": 4,
        "predicted_policy_return_delta_tfv_m3": prediction,
        "true_policy_return_delta_tfv_m3": truth,
        "candidate_branch_tfv_m3": 1000.0 + truth,
        "hold_branch_tfv_m3": 1000.0,
        "same_prefix_verified": True,
        "same_continuation_policy_verified": True,
        "future_realized_rainfall_used_online": False,
        "continuation_policy_sha256": "a" * 64,
        "prefix_sha256": "b" * 64,
        "candidate_first_target_sha256": "c" * 64,
        "hold_first_target_sha256": "d" * 64,
    }


def test_portfolio_admission_requires_and_records_multi_candidate_query_sets() -> None:
    records = []
    groups = [f"G{i:02d}" for i in range(24)]
    for i, group in enumerate(groups):
        records.append(
            _calibration_record(
                group,
                "V12_DIRECTION_SCALE_1.00",
                truth=-100.0 - i,
                prediction=-110.0 - i,
            )
        )
        records.append(
            _calibration_record(
                group,
                "TYPE_AWARE_HYDRAULIC_PRESSURE",
                truth=-80.0 - i,
                prediction=-90.0 - i,
            )
        )
    payload = derive_policy_return_portfolio_admission(
        records=records,
        expected_rainfall_groups=groups,
        policy_return_checkpoint_sha256="e" * 64,
        continuation_policy_sha256="a" * 64,
        coverage=0.90,
    )
    assert payload["portfolio_admission_contract"] == DIRECT_TFV_POLICY_RETURN_PORTFOLIO_ADMISSION_CONTRACT
    assert payload["candidate_portfolio_contract"] == DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT
    assert payload["query_set_count"] == 24
    assert payload["multi_candidate_query_set_count"] == 24
    assert payload["candidate_source_counts"]["TYPE_AWARE_HYDRAULIC_PRESSURE"] == 24
