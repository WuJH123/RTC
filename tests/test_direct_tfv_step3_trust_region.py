from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.step2_tfv_support import DIRECT_TFV_ACTION_SUPPORT_CONTRACT
from rtc.step2_tfv_value import DirectFacilityTFVValueModel
from rtc.step2_train_response_v60 import InputNormalizationV60
from rtc.step3_tfv_value_mpc_v2 import DirectTFVMPCDesignV2, DirectTFVTrustRegionMPC


def _controller() -> DirectTFVTrustRegionMPC:
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
    }
    return DirectTFVTrustRegionMPC(
        model=model,
        graph=graph,
        normalization=normalization,
        action_support=support,
        design=DirectTFVMPCDesignV2(),
    )


def test_default_active_set_uses_joint_trainfit_median_not_all_109() -> None:
    controller = _controller()
    assert controller._active_count(109) == 8


def test_trust_decoder_keeps_nonactive_facilities_at_hold_and_respects_support() -> None:
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
