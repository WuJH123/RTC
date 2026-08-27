from __future__ import annotations

from types import SimpleNamespace

import torch

from rtc.controller_direct_tfv_portfolio import (
    _runtime_policy_return_passed,
    _v27_structured_diagnostics,
)
from rtc.direct_tfv_operational_v27r1_runtime import DirectTFVOperationalV27R1PhysicalOnlyMPC


def test_runtime_portfolio_telemetry_falls_back_to_adapter_admission_passed() -> None:
    wrapped = SimpleNamespace(admission_passed=True, candidate_valid=True)
    assert _runtime_policy_return_passed(wrapped) is True
    wrapped = SimpleNamespace(admission_passed=False, candidate_valid=False)
    assert _runtime_policy_return_passed(wrapped) is False


def test_runtime_portfolio_telemetry_prefers_explicit_policy_return_field() -> None:
    wrapped = SimpleNamespace(
        policy_return_admission_passed=False,
        admission_passed=True,
        candidate_valid=True,
    )
    assert _runtime_policy_return_passed(wrapped) is False


def test_v27_string_diagnostics_are_exposed_as_structured_fields() -> None:
    parsed = _v27_structured_diagnostics(
        "V27_DECISION_AWARE|latent_min=-1.25|latent_max=2.5|clip_hits=0|raw_clip_hits=1|"
        "q95_binding_candidates=3|raw_best=STEP2_H10_PROBE_SCALE_0.50|"
        "supported_best=TYPE_AWARE_HYDRAULIC_PRESSURE|q95_selection_changed=true|"
        "raw_best_latent=-0.75|supported_best_latent=0.25|shadow_present=true|"
        "shadow_selected=false|shadow_duplicate=false"
    )
    assert parsed["v27_q95_binding_candidate_count"] == 3
    assert parsed["v27_q95_selection_changed"] is True
    assert parsed["v27_raw_best_source"] == "STEP2_H10_PROBE_SCALE_0.50"
    assert parsed["v27_supported_best_source"] == "TYPE_AWARE_HYDRAULIC_PRESSURE"
    assert parsed["v27_raw_best_latent"] == -0.75
    assert parsed["v27_supported_best_latent"] == 0.25
    assert parsed["v27_auto_rbc_shadow_present"] is True
    assert parsed["v27_auto_rbc_shadow_selected"] is False


def test_v27r1_physical_only_h10_keeps_raw_action_while_reporting_q95_counterfactual() -> None:
    mpc = object.__new__(DirectTFVOperationalV27R1PhysicalOnlyMPC)
    mpc.design = SimpleNamespace(prediction_horizon_steps=72, control_block_steps=2)
    mpc.joint_sequence_support_diagnostics = lambda sequence, active: {
        "quantile": "q95",
        "first_block_l1": 4.0,
        "first_block_l1_limit": 2.0,
        "first_block_l1_ratio": 2.0,
        "h120_l1": 4.0,
        "h120_l1_limit": 2.0,
        "h120_l1_ratio": 2.0,
        "h120_total_variation_l1": 4.0,
        "h120_total_variation_l1_limit": 2.0,
        "h120_total_variation_l1_ratio": 2.0,
        "max_ratio": 2.0,
        "binding": True,
    }
    active = torch.zeros(109, dtype=torch.float32)
    target = active.clone()
    target[:4] = 0.5
    supported_target, sequence, changed, diagnostics = mpc._h10_supported_target(target, active)
    assert torch.allclose(supported_target, target)
    assert changed == 4
    assert diagnostics["quantile"] == "PHYSICAL_ONLY_Q95_ABLATION"
    assert diagnostics["q95_counterfactual_binding"] is True
    assert diagnostics["q95_counterfactual_max_ratio"] == 2.0
    assert diagnostics["binding"] is False
    assert diagnostics["max_ratio"] == 0.0
    assert tuple(sequence.shape) == (72, 109)
