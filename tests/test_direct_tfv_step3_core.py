from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.step2_tfv_support import DIRECT_TFV_ACTION_SUPPORT_CONTRACT
from rtc.step2_tfv_value import DirectFacilityTFVValueModel
from rtc.step2_train_response_v60 import InputNormalizationV60
from rtc.step3_tfv_value_mpc_v3 import DirectTFVMPCDesignV3, DirectTFVRecedingMPC


def _controller() -> DirectTFVRecedingMPC:
    actuator_ids = tuple(f"A{i:03d}" for i in range(109))
    graph = SimpleNamespace(
        actuator_ids=actuator_ids,
        actuator_physics_feature_names=("min_setting", "max_setting"),
        actuator_physics=np.column_stack((np.zeros(109), np.ones(109))).astype(np.float32),
        actuator_upstream=np.arange(109) % 8,
        actuator_downstream=(np.arange(109) + 1) % 8,
    )
    model = DirectFacilityTFVValueModel(
        state_dim=3,
        rainfall_dim=1,
        actuator_physics_dim=2,
        target_scale_m3=5000.0,
    )
    normalization = InputNormalizationV60(
        state_mean=np.zeros(3, dtype=np.float32),
        state_std=np.ones(3, dtype=np.float32),
        rainfall_mean=np.zeros(1, dtype=np.float32),
        rainfall_std=np.ones(1, dtype=np.float32),
        flow_mean=np.zeros(109, dtype=np.float32),
        flow_std=np.ones(109, dtype=np.float32),
    )
    support = {
        "contract": DIRECT_TFV_ACTION_SUPPORT_CONTRACT,
        "actuator_ids": list(actuator_ids),
        "single_facility_coverage_count": 109,
        "first_move_abs_q95_per_facility": [0.10] * 109,
        "sequence_abs_q95_per_facility": [0.20] * 109,
        "joint_changed_facility_count_q50": 8.0,
        "joint_changed_facility_count_q75": 16.0,
        "joint_changed_facility_count_q90": 22.0,
    }
    return DirectTFVRecedingMPC(
        model=model,
        graph=graph,
        normalization=normalization,
        action_support=support,
        design=DirectTFVMPCDesignV3(),
    )


def test_default_active_set_uses_joint_trainfit_q90_to_preserve_control_freedom() -> None:
    controller = _controller()
    assert controller._active_count(109) == 22
    assert controller._active_count(5) == 5


def test_core_decoder_keeps_nonactive_facilities_at_hold_and_respects_support() -> None:
    controller = _controller()
    active_target = torch.full((109,), 0.5)
    active = torch.tensor([2, 7, 11], dtype=torch.long)
    fractions = torch.ones((12, 3))
    sequence = controller._decode_active_fractions(
        fractions,
        active_indices=active,
        active_target=active_target,
    )
    assert sequence.shape == (72, 109)
    inactive = torch.ones(109, dtype=torch.bool)
    inactive[active] = False
    torch.testing.assert_close(
        sequence[:, inactive],
        active_target[inactive][None].expand(72, -1),
    )
    first = sequence[:2].mean(dim=0)
    assert float(torch.max(torch.abs(first[active] - 0.5))) <= 0.100001
    assert controller._support_ratio(sequence, active_target) <= 1.00001


def test_screening_uses_two_scales_and_immediate_delayed_modes() -> None:
    design = DirectTFVMPCDesignV3()
    assert design.screening_probe_scales == (0.5, 1.0)
    assert design.screening_probe_modes == ("pulse", "persistent")


def test_pulse_and_persistent_probe_have_distinct_temporal_support() -> None:
    controller = _controller()
    hold = torch.full((72, 109), 0.5)
    pulse = controller._probe_sequence(
        hold=hold,
        actuator_index=3,
        target=0.6,
        mode="pulse",
    )
    persistent = controller._probe_sequence(
        hold=hold,
        actuator_index=3,
        target=0.6,
        mode="persistent",
    )
    assert torch.allclose(pulse[:2, 3], torch.full((2,), 0.6))
    assert torch.allclose(pulse[2:, 3], torch.full((70,), 0.5))
    assert torch.allclose(persistent[:, 3], torch.full((72,), 0.6))
