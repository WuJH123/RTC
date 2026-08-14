from __future__ import annotations

import numpy as np
import torch

from rtc.step2_control_basis_v60 import build_control_basis_v60
from rtc.step2_train_response_v60 import V60GroupBatch
from rtc.step2_v110 import (
    V110ActionEffectModel,
    V110TrainingDesign,
    action_effect_loss_v110,
    derive_effect_scales_v110,
    prepare_action_effect_batch_v110,
)


class _Prepared:
    def __init__(self):
        self.reference_settings = torch.zeros(1, 72, 2)
        self.candidate_settings = torch.zeros(1, 2, 72, 2)
        self.true_reference_states = torch.zeros(1, 1, 72, 3, 6)
        self.true_candidate_states = torch.zeros(1, 2, 72, 3, 6)
        self.true_reference_flows = torch.zeros(1, 1, 72, 2)
        self.true_candidate_flows = torch.zeros(1, 2, 72, 2)


def _prepared():
    return _Prepared()


def test_v110_import_and_forward_shapes():
    model = V110ActionEffectModel(
        state_dim=6,
        actuator_count=2,
        coefficient_dim=4,
        hidden_dim=16,
    )
    initial = torch.zeros(3, 6)
    coefficients = torch.zeros(3, 4)
    state, flow = model(initial, coefficients)
    assert state.shape == (3, 6)
    assert flow.shape == (3, 2)


def test_v110_loss_is_finite():
    prediction = torch.zeros(2, 4, requires_grad=True)
    target = torch.ones(2, 4)
    active = torch.ones(2, 4, dtype=torch.bool)
    loss = action_effect_loss_v110(
        prediction,
        target,
        active,
        scale=torch.ones(4),
        design=V110TrainingDesign(),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None


def test_v110_prepare_batch_uses_candidate_minus_reference():
    batch = V60GroupBatch(
        source_kind="D2",
        group_name="D2::x",
        initial_state=torch.zeros(1, 3, 6),
        rainfall=torch.zeros(1, 72, 3, 1),
        reference_settings=torch.zeros(1, 72, 2),
        candidate_settings=torch.ones(1, 2, 72, 2),
        previous_actuator_flow=torch.zeros(1, 2),
        elapsed_seconds=torch.zeros(1),
        true_reference_states=torch.zeros(1, 1, 72, 3, 6),
        true_candidate_states=torch.ones(1, 2, 72, 3, 6),
        true_reference_flows=torch.zeros(1, 1, 72, 2),
        true_candidate_flows=torch.ones(1, 2, 72, 2),
        true_delta_tfv_m3=torch.zeros(1, 2),
    )
    prepared = prepare_action_effect_batch_v110(batch)
    assert torch.all(prepared.state_effect == 1.0)
    assert torch.all(prepared.flow_effect == 1.0)


class _BasisGraph:
    actuator_ids = ("a0", "a1")
    actuator_upstream = np.asarray([0, 1], dtype=np.int64)
    actuator_downstream = np.asarray([1, 2], dtype=np.int64)
    node_ids = ("n0", "n1", "n2")


def test_v110_basis_integration_is_finite():
    basis = build_control_basis_v60(_BasisGraph())
    assert basis.coefficient_dimension > 0


class _FakeEntry:
    def __init__(self, arrays):
        self.arrays = arrays
        self.indices = tuple(range(len(arrays["target_states"])))


class _FakeCache:
    def __init__(self, arrays):
        self._entry = _FakeEntry(arrays)

    def entry(self, name):
        return self._entry


def test_v110_active_thresholds_are_local_and_physically_floored():
    target_states = np.zeros((2, 72, 3, 6), dtype=np.float32)
    target_states[1, :, :, 0] = 0.005
    target_states[1, :, :, 1] = 0.005
    target_flows = np.zeros((2, 72, 2), dtype=np.float32)
    arrays = {"target_states": target_states, "target_actuator_flows": target_flows}
    scales = derive_effect_scales_v110(_FakeCache(arrays), ["D2::x"], _prepared())
    depth_threshold = scales.state_active_threshold[:, 0]
    assert depth_threshold[0] >= 0.01
    assert depth_threshold[2] >= 0.04
    assert depth_threshold[2] > depth_threshold[0]


def test_historical_v110_remains_importable_but_current_surface_is_v127():
    import rtc.step2_current as current

    assert current.CURRENT_PROJECT7_CONTRACT == "PROJECT7_V127_CONTINUOUS_DIFFERENTIABLE_MPC_CORRECTNESS_V2"
    assert current.CURRENT_STEP2_CONTRACT.startswith(
        "PROJECT7_V127_CONTROL_ORIENTED_DIFFERENTIABLE_HYDRAULIC_SURROGATE"
    )
    assert current.CURRENT_STEP3_CONTRACT.startswith(
        "PROJECT7_V127_109ACT_H120_LBFGSB_RECEDING_HORIZON_MPC"
    )
    assert current.CONTINUOUS_MPC_ENABLED is True
    assert current.CONTINUOUS_MPC_RUNTIME_REQUIRES_GATE is True
    assert current.HYDRAULIC_MODEL_REQUIRED_ONLINE is True
    assert current.RBC_IS_VALUE_REFERENCE is False
    assert current.RBC_IS_ACTION_SPACE_CEILING is False
    assert current.CURRENT_POLICY_CLASS.__name__ == "DifferentiableRollingMPCV127"
    assert "step2_current" in current.__file__
