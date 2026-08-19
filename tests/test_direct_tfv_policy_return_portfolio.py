from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.actuator_release_semantics import (
    graph_release_setting_signs,
    release_fraction_to_setting,
    release_setting_sign,
)
from rtc.direct_tfv_policy_return import DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING
from rtc.direct_tfv_policy_return_portfolio import (
    DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT,
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
    build_policy_return_candidate_portfolio,
    hydraulic_pressure_setting_delta,
)
from rtc.direct_tfv_policy_return_portfolio_admission import (
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_ADMISSION_CONTRACT,
    derive_policy_return_portfolio_admission,
)
from rtc.policy_return_replay import (
    POLICY_RETURN_BRANCH_RELEASE_CONTRACT,
    POLICY_RETURN_PREFIX_AUDIT_CONTRACT,
    audit_policy_return_prefix_contexts,
    snapshot_and_release_policy_return_branch,
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
    state = np.asarray([[1.8, 0.0, 0.0, 900.0], [0.2, 0.0, 0.0, 100.0]], dtype=np.float32)
    rainfall = np.ones((3, 72, 2, 1), dtype=np.float32)
    delta = hydraulic_pressure_setting_delta(
        current_state=state,
        rainfall_scenarios=rainfall,
        graph=graph,
        max_delta_per_update=0.5,
    )
    assert delta[0] > 0.0
    assert delta[1] > 0.0
    assert delta[2] < 0.0
    assert delta[3] > 0.0


def test_candidate_portfolio_is_three_family_supported_and_not_baseline_imitation() -> None:
    graph = _graph()
    state = torch.tensor([[[1.8, 0.0, 0.0, 900.0], [0.2, 0.0, 0.0, 100.0]]], dtype=torch.float32)
    rainfall = torch.ones((3, 72, 2, 1), dtype=torch.float32)
    active = torch.full((109,), 0.5, dtype=torch.float32)
    learned = active.clone()
    learned[:6] += 0.2
    candidates = build_policy_return_candidate_portfolio(
        current_state=state,
        rainfall_scenarios=rainfall,
        active_target=active,
        learned_target=learned,
        graph=graph,
        first_radius=np.full(109, 0.15, dtype=np.float32),
        max_changed_facilities=4,
        max_delta_per_update=0.5,
    )
    sources = {row.source for row in candidates}
    assert "TYPE_AWARE_HYDRAULIC_PRESSURE" in sources
    assert any(value.startswith("STEP2_H10_PROBE_SCALE_") for value in sources)
    assert all("V12_DIRECTION" not in value and "BLEND" not in value for value in sources)
    assert not any("AUTO_RBC" in value or "ALL_OPEN" in value for value in sources)
    assert len(candidates) <= 3
    for row in candidates:
        delta = torch.abs(row.target - active)
        assert row.changed_facility_count <= 4
        assert float(delta.max()) <= 0.150001
        assert bool(torch.all((row.target >= 0.0) & (row.target <= 1.0)))


def test_policy_return_branch_release_copies_context_and_severs_delegate() -> None:
    original = np.arange(12, dtype=np.float32).reshape(3, 4)
    wrapper = SimpleNamespace(
        branch_context={
            "current_state": original,
            "rainfall_scenarios": np.ones((3, 2, 2, 1), dtype=np.float32),
            "active_target": np.full(109, 0.5, dtype=np.float32),
            "previous_actuator_flow": np.zeros(109, dtype=np.float32),
        },
        delegate=SimpleNamespace(model=torch.nn.Linear(4, 2)),
    )
    context, telemetry = snapshot_and_release_policy_return_branch(wrapper, device=torch.device("cpu"))
    assert wrapper.delegate is None
    assert np.array_equal(context["current_state"], original)
    assert not np.shares_memory(context["current_state"], original)
    assert telemetry["contract"] == POLICY_RETURN_BRANCH_RELEASE_CONTRACT
    assert telemetry["cuda_device_type"] == "cpu"
    assert telemetry["allocated_before_bytes"] == 0
    assert telemetry["reserved_after_bytes"] == 0


def _prefix_context() -> dict[str, np.ndarray]:
    return {
        "raw_observed_history": np.zeros((13, 2, 6), dtype=np.float32),
        "raw_mask_history": np.ones((13, 2, 6), dtype=np.float32),
        "raw_context_history": np.zeros((13, 2, 2), dtype=np.float32),
        "raw_rainfall_history": np.zeros((13, 2, 1), dtype=np.float32),
        "raw_sensor_depth_m": np.asarray([0.4, 0.8], dtype=np.float64),
        "raw_sensor_head_m": np.asarray([10.4, 20.8], dtype=np.float64),
        "raw_observed_rainfall_mmhr": np.asarray([5.0, 5.0], dtype=np.float64),
        "raw_actuator_target_setting": np.full(109, 0.5, dtype=np.float64),
        "raw_actuator_current_setting": np.full(109, 0.5, dtype=np.float64),
        "raw_actuator_flow_m3s": np.full(109, 20.0, dtype=np.float64),
        "current_state": np.zeros((2, 6), dtype=np.float32),
        "rainfall_scenarios": np.ones((3, 72, 2, 1), dtype=np.float32),
        "active_target": np.full(109, 0.5, dtype=np.float32),
        "previous_actuator_flow": np.full(109, 20.0, dtype=np.float32),
    }


def test_prefix_audit_does_not_confuse_step1_float_noise_with_physical_prefix() -> None:
    candidate = _prefix_context()
    hold = {key: value.copy() for key, value in candidate.items()}
    hold["current_state"][0, 0] += 0.001129150390625
    report = audit_policy_return_prefix_contexts(candidate, hold)
    assert report["contract"] == POLICY_RETURN_PREFIX_AUDIT_CONTRACT
    assert report["same_authoritative_prefix_verified"] is True
    assert report["derived_step1_reconstruction_max_abs"] > 0.001
    assert report["derived_context_controls_prefix_gate"] is False


def test_prefix_audit_rejects_material_raw_sensor_difference() -> None:
    candidate = _prefix_context()
    hold = {key: value.copy() for key, value in candidate.items()}
    hold["raw_sensor_depth_m"][0] += 0.01
    report = audit_policy_return_prefix_contexts(candidate, hold)
    assert report["same_authoritative_prefix_verified"] is False
    assert report["exact_raw_max_abs_by_field"]["raw_sensor_depth_m"] > 0.009


def _calibration_record(group: str, source: str, truth: float, prediction: float) -> dict:
    query = (group.encode("utf-8").hex() + "0" * 64)[:64]
    return {
        "estimand": "EXECUTE_CANDIDATE_H10_THEN_FROZEN_POLICY_VS_HOLD_H10_THEN_SAME_FROZEN_POLICY",
        "action_encoding_contract": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
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
                "STEP2_H10_PROBE_SCALE_0.50",
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
    assert payload["action_encoding_contract"] == DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING
    assert payload["query_set_count"] == 24
    assert payload["multi_candidate_query_set_count"] == 24
    assert payload["candidate_source_counts"]["TYPE_AWARE_HYDRAULIC_PRESSURE"] == 24
    assert payload["required_candidate_families_present"]["step2_h10_probe_direction"] is True
    assert DIRECT_TFV_H10_PROBE_GENERATOR_CONTRACT
