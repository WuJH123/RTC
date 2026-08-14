from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from rtc.step2_differentiable_v127 import ControlOrientedDifferentiableSurrogateV127
from rtc.step2_state_store_v127 import CausalStateStoreV127
from rtc.step3_mpc_v127 import (
    ContinuousMPCDesignV127,
    Step2GradientEvidenceV127,
    decode_fractional_targets_v127,
)


def _sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def test_v127_smooth_tfv_has_finite_action_gradient() -> None:
    torch.manual_seed(3)
    model = ControlOrientedDifferentiableSurrogateV127(
        state_dim=3,
        rainfall_dim=1,
        node_static_dim=2,
        actuator_physics_dim=2,
        actuator_count=1,
        hidden_dim=16,
        actuator_embedding_dim=4,
        delta_state_scale=torch.tensor([0.1, 0.1, 0.05]),
        delta_flow_scale=torch.tensor([0.1]),
    )
    initial = torch.zeros(1, 2, 3)
    rainfall = torch.zeros(1, 4, 2, 1)
    settings = torch.full((1, 4, 1), 0.5, requires_grad=True)
    result = model.objective_rollout(
        initial_state=initial,
        rainfall=rainfall,
        settings=settings,
        previous_actuator_flow=torch.zeros(1, 1),
        actuator_upstream=torch.tensor([0]),
        actuator_downstream=torch.tensor([1]),
        actuator_physics=torch.zeros(1, 2),
        static_node_features=torch.zeros(2, 2),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        flood_rate_index=2,
        priority_indices=torch.tensor([0]),
        dt_seconds=300.0,
    )
    assert torch.isfinite(result.tfv_m3).all()
    assert torch.isfinite(result.optimization_tfv_m3).all()
    gradient = torch.autograd.grad(result.optimization_tfv_m3.sum(), settings)[0]
    assert gradient.shape == settings.shape
    assert torch.isfinite(gradient).all()


def test_v127_fraction_decoder_is_1308_variable_bounded_rate_feasible() -> None:
    design = ContinuousMPCDesignV127()
    fractions = torch.linspace(0.0, 1.0, 12 * 109).reshape(12, 109)
    active = torch.full((109,), 0.4)
    lo = torch.zeros(109)
    hi = torch.ones(109)
    sequence = decode_fractional_targets_v127(
        fractions,
        active_target=active,
        min_setting=lo,
        max_setting=hi,
        design=design,
    )
    assert sequence.shape == (72, 109)
    torch.testing.assert_close(sequence[0], sequence[1])
    blocks = sequence[::2]
    assert bool(torch.all((blocks >= 0.0) & (blocks <= 1.0)))
    first_delta = torch.max(torch.abs(blocks[0] - active))
    later_delta = torch.max(torch.abs(blocks[1:] - blocks[:-1]))
    assert float(first_delta) <= 0.500001
    assert float(later_delta) <= 0.500001
    torch.testing.assert_close(blocks[12:], blocks[11:12].expand_as(blocks[12:]))


def test_v127_continuous_gate_is_fail_closed() -> None:
    good = Step2GradientEvidenceV127(
        holdout_rank=0.71,
        holdout_top1=0.51,
        d2_gradient_sign_accuracy=0.71,
        d2_gradient_cosine_similarity=0.61,
        d5_gradient_sign_accuracy=0.72,
        d5_gradient_cosine_similarity=0.62,
        causal_step1_state_verified=True,
        causal_rainfall_verified=True,
    )
    good.validate()
    bad = Step2GradientEvidenceV127(
        holdout_rank=0.69,
        holdout_top1=0.9,
        d2_gradient_sign_accuracy=0.9,
        d2_gradient_cosine_similarity=0.9,
        d5_gradient_sign_accuracy=0.9,
        d5_gradient_cosine_similarity=0.9,
        causal_step1_state_verified=True,
        causal_rainfall_verified=True,
    )
    with pytest.raises(ValueError, match="rank"):
        bad.validate()


def test_v127_causal_state_store_content_hash_and_unique_identity() -> None:
    state = np.zeros((2, 3, 4), dtype=np.float32)
    store = CausalStateStoreV127(
        event_ids=("e1", "e2"),
        checkpoint_ids=("c1", "c2"),
        elapsed_seconds=np.asarray([600, 1200], dtype=np.int64),
        state_si=state,
        current_setting=np.full((2, 109), 0.5, dtype=np.float32),
        state_sha256=(_sha(state[0]), _sha(state[1])),
        step1_sha256="a" * 64,
        sensor_sha256="b" * 64,
        graph_sha256="c" * 64,
    )
    store.validate()
    broken = CausalStateStoreV127(
        **{**store.__dict__, "state_sha256": ("0" * 64, _sha(state[1]))}
    )
    with pytest.raises(ValueError, match="content hash"):
        broken.validate()
