from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rtc.controller_v123 import V123_CONTROLLER_CONTRACT, V123TorchMPCController
from rtc.step2_policy_v123 import (
    FirstMoveTFVPFVResultV123,
    anchor_base_settings_v123,
    V123_POLICY_MODE_CONTRACTS,
    V123_POLICY_MODES,
    safe_runtime_delta_v123,
)
from scripts.run_policy_v123 import _policy_mode_runtime_metadata


def test_v123_result_exposes_controller_diagnostic_aliases() -> None:
    result = FirstMoveTFVPFVResultV123(
        settings=np.zeros((2, 1)),
        candidate_valid=True,
        selected_candidate_index=3,
        raw_candidate_count=169,
        first_move_group_count=75,
        tail_only_noop_candidate_count=58,
        scenario_count=1,
        predicted_delta_tfv_m3=-12000.0,
        predicted_delta_pfv_m3=2.0,
        tfv_risk_m3=-12000.0,
        pfv_risk_m3=2.0,
        pfv_soft_excess_m3=0.0,
        pfv_penalty_m3_equivalent=0.0,
        objective_score_m3_equivalent=-11998.0,
        false_benefit_margin_m3=1000.0,
        scoring_projection_max=0.0,
    )
    assert result.candidate_count == 169
    assert result.selected_group_score_m3 == -11998.0


def test_v123_policy_modes_have_independent_contracts() -> None:
    assert V123_POLICY_MODES == ("anchor_only", "learned_only", "hybrid")
    assert len(set(V123_POLICY_MODE_CONTRACTS.values())) == 3
    result = FirstMoveTFVPFVResultV123(
        settings=np.zeros((2, 1)),
        candidate_valid=False,
        selected_candidate_index=0,
        raw_candidate_count=1,
        first_move_group_count=1,
        tail_only_noop_candidate_count=0,
        scenario_count=1,
        predicted_delta_tfv_m3=0.0,
        predicted_delta_pfv_m3=0.0,
        tfv_risk_m3=0.0,
        pfv_risk_m3=0.0,
        pfv_soft_excess_m3=0.0,
        pfv_penalty_m3_equivalent=0.0,
        objective_score_m3_equivalent=0.0,
        false_benefit_margin_m3=1.0,
        scoring_projection_max=0.0,
        policy_mode="anchor_only",
        policy_mode_contract=V123_POLICY_MODE_CONTRACTS["anchor_only"],
    )
    assert result.policy_mode == "anchor_only"
    assert result.policy_mode_contract == V123_POLICY_MODE_CONTRACTS["anchor_only"]


def test_v123_anchor_uses_active_target_for_command_continuity() -> None:
    current = np.asarray([0.1, 0.2], dtype=np.float32)
    target = np.asarray([0.4, 0.5], dtype=np.float32)
    assert np.array_equal(anchor_base_settings_v123(current, target), target)
    assert np.array_equal(anchor_base_settings_v123(current, None), current)


def test_v123_runtime_delta_leaves_float32_guard_margin() -> None:
    assert safe_runtime_delta_v123(0.5) < 0.5
    assert safe_runtime_delta_v123(0.5) > 0.499


def test_v123_runtime_mode_metadata_does_not_claim_anchor_for_learned_only() -> None:
    learned = _policy_mode_runtime_metadata("learned_only")
    hybrid = _policy_mode_runtime_metadata("hybrid")
    assert learned["knowledge_data_fusion"] is False
    assert learned["knowledge_anchor_default_when_value_uncertain"] is False
    assert hybrid["knowledge_data_fusion"] is True
    assert hybrid["knowledge_anchor_default_when_value_uncertain"] is True


def test_v123_controller_contract_is_distinct_from_v122() -> None:
    assert V123_CONTROLLER_CONTRACT.startswith("PROJECT7_V123_")
    assert issubclass(V123TorchMPCController, object)
