from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

import rtc.step2_policy_v125 as policy_module
from rtc.step2_policy_v125 import AnchorOverridePolicyV125
from rtc.step3_objective_v123 import TFVPFVObjectiveV123


class _Basis:
    def __init__(self) -> None:
        self.horizon = SimpleNamespace(horizon_steps=6, control_block_steps=2)
        self.grouping = SimpleNamespace(
            actuator_count=3,
            group_id_by_actuator=np.asarray([0, 1, 2], dtype=np.int64),
        )
        self.contract = SimpleNamespace(max_setting_delta_per_update=0.5)
        self.min_setting = np.zeros(3, dtype=np.float32)
        self.max_setting = np.ones(3, dtype=np.float32)

    def validate(self) -> None:
        return None


class _Normalization:
    def validate(self) -> None:
        return None

    def state(self, value):
        return value

    def rainfall(self, value):
        return value

    def flow(self, value):
        return value


class _DirectAdvantageModel:
    def __init__(self, *, tfv_direction: float = 1.0, pfv_direction: float = 0.0) -> None:
        self.tfv_direction = float(tfv_direction)
        self.pfv_direction = float(pfv_direction)

    def __call__(self, state, rainfall, reference, candidate, flow, prepared):
        reference_expanded = reference[:, None].expand_as(candidate)
        # Only the first executable block carries candidate identity in V125.
        delta = (candidate[:, :, :2] - reference_expanded[:, :, :2]).mean(dim=(2, 3))
        tfv = 1000.0 * self.tfv_direction * delta
        pfv = 1000.0 * self.pfv_direction * delta
        return SimpleNamespace(delta_tfv_m3=tfv, delta_pfv_m3=pfv)


def _objective() -> TFVPFVObjectiveV123:
    return TFVPFVObjectiveV123(
        pfv_soft_margin_m3=0.0,
        pfv_scale_m3=100.0,
        tfv_scale_m3=100.0,
        pfv_penalty_weight=1.0,
        pfv_model_error_margin_m3=0.0,
        movement_penalty_m3=0.0,
    )


def _policy(*, margin: float, tfv_direction: float = 1.0, pfv_direction: float = 0.0):
    policy = AnchorOverridePolicyV125(
        model=_DirectAdvantageModel(
            tfv_direction=tfv_direction, pfv_direction=pfv_direction
        ),
        basis=_Basis(),
        prepared=object(),
        normalization=_Normalization(),
        objective=_objective(),
        anchor_override_margin_m3=margin,
        graph=object(),
        max_active_groups=3,
        local_fraction=0.25,
    )
    # Isolate Step3 semantics from the already separately-tested Sparse-RBC builder.
    policy._anchor = lambda **_: torch.full((6, 3), 0.6, dtype=torch.float32)
    return policy


def _inputs():
    return dict(
        initial_state=torch.zeros(3, 6),
        rainfall_scenarios=torch.zeros(1, 6, 3, 1),
        fallback_settings=torch.full((1, 6, 3), 0.2),
        current_settings=torch.full((3,), 0.2),
        previous_requested_settings=torch.full((3,), 0.2),
        previous_actuator_flow=torch.zeros(3),
        max_delta_per_update=0.5,
    )


def test_anchor_is_exact_zero_reference_and_default_when_margin_not_cleared() -> None:
    result = _policy(margin=1000.0).optimize(**_inputs())
    assert result.selected_source == "anchor_default"
    assert not result.learned_override_admitted
    assert result.knowledge_anchor_selected
    assert result.anchor_tfv_risk_m3 == 0.0
    assert result.anchor_pfv_risk_m3 == 0.0
    torch.testing.assert_close(result.settings, torch.full((6, 3), 0.6))


def test_learned_override_is_scored_directly_against_anchor_and_tail_is_common() -> None:
    result = _policy(margin=50.0).optimize(**_inputs())
    assert result.selected_source == "learned_override"
    assert result.learned_override_admitted
    assert result.predicted_override_advantage_tfv_m3 < -50.0
    # Only first 600 s may differ; all later settings are the exact anchor continuation.
    torch.testing.assert_close(result.settings[2:], torch.full((4, 3), 0.6))


def test_pfv_improvement_cannot_buy_tfv_worse_override() -> None:
    # For lower-than-anchor candidates TFV is positive (worse) while PFV is negative
    # (better). The TFV-primary admission gate must still retain the anchor.
    result = _policy(margin=0.0, tfv_direction=-1.0, pfv_direction=1.0).optimize(**_inputs())
    assert result.selected_source == "anchor_default"
    assert not result.learned_override_admitted


def test_runtime_candidate_family_matches_d4_local_support() -> None:
    result = _policy(margin=50.0).optimize(**_inputs())
    assert result.raw_candidate_count >= 4
    assert result.raw_candidate_count <= 10
    assert result.selected_candidate_family in {
        "hold",
        "anchor_scale_0.50",
        "anchor_scale_0.75",
        "anchor_scale_1.00",
        "anchor_group_0_minus25",
        "anchor_group_0_plus25",
        "anchor_group_1_minus25",
        "anchor_group_1_plus25",
        "anchor_group_2_minus25",
        "anchor_group_2_plus25",
    }


def test_float_projection_drift_cannot_destroy_exact_anchor(monkeypatch) -> None:
    original = policy_module._project_executable_sequences_v120

    def tiny_drift(values, **kwargs):
        projected, maximum = original(values, **kwargs)
        projected = projected.clone()
        # Mimic the observed boundary-case float32 ulp perturbation of the exact anchor.
        distances = torch.amax(torch.abs(projected - 0.6), dim=(1, 2))
        slot = int(torch.argmin(distances).item())
        projected[slot, 0, 0] += torch.finfo(projected.dtype).eps
        return projected, max(float(maximum), float(torch.finfo(projected.dtype).eps))

    monkeypatch.setattr(policy_module, "_project_executable_sequences_v120", tiny_drift)
    result = _policy(margin=1000.0).optimize(**_inputs())
    assert result.selected_source == "anchor_default"
    assert result.knowledge_anchor_selected
    torch.testing.assert_close(result.settings, torch.full((6, 3), 0.6), rtol=0.0, atol=0.0)
