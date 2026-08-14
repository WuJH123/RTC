from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.step2_control_value_v124 import (
    ControlValueSurrogateV124,
    ValueLossContractV124,
    value_loss_v124,
)


def _prepared() -> SimpleNamespace:
    return SimpleNamespace(
        actuator_upstream=torch.tensor([0, 1, 2], dtype=torch.long),
        actuator_downstream=torch.tensor([1, 2, 3], dtype=torch.long),
        actuator_physics=torch.zeros((3, 2), dtype=torch.float32),
    )


def _model() -> ControlValueSurrogateV124:
    temporal = np.asarray(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return ControlValueSurrogateV124(
        state_dim=6,
        rainfall_dim=1,
        physics_dim=2,
        actuator_count=3,
        temporal_basis=temporal,
        control_block_steps=2,
        tfv_scale_m3=1000.0,
        hidden_dim=32,
        actuator_embedding_dim=8,
        attention_heads=4,
    )


def test_v124_exact_zero_for_reference_action() -> None:
    model = _model().eval()
    state = torch.zeros((1, 4, 6), dtype=torch.float32)
    rainfall = torch.zeros((1, 6, 4, 1), dtype=torch.float32)
    reference = torch.full((1, 6, 3), 0.5, dtype=torch.float32)
    candidates = torch.stack(
        [
            reference[0],
            reference[0].clone(),
        ],
        dim=0,
    )[None]
    candidates[0, 1, 0:2, 0] = 0.8
    flow = torch.zeros((1, 3), dtype=torch.float32)
    out = model(state, rainfall, reference, candidates, flow, _prepared())
    assert out.delta_tfv_m3.shape == (1, 2)
    assert out.delta_tfv_m3[0, 0].item() == 0.0
    assert torch.isfinite(out.delta_tfv_m3).all()
    assert not torch.allclose(out.actuator_effect_tokens[0, 1], torch.zeros_like(out.actuator_effect_tokens[0, 1]))


def test_v124_attention_handles_all_hold_without_nan() -> None:
    model = _model().eval()
    state = torch.zeros((1, 4, 6), dtype=torch.float32)
    rainfall = torch.zeros((1, 6, 4, 1), dtype=torch.float32)
    reference = torch.full((1, 6, 3), 0.5, dtype=torch.float32)
    candidates = reference[:, None].expand(1, 4, -1, -1).clone()
    flow = torch.zeros((1, 3), dtype=torch.float32)
    out = model(state, rainfall, reference, candidates, flow, _prepared())
    assert torch.all(out.delta_tfv_m3 == 0.0)
    assert torch.isfinite(out.pooled_interaction).all()


def test_v124_listwise_loss_is_finite_and_backpropagates() -> None:
    model = _model().train()
    state = torch.zeros((1, 4, 6), dtype=torch.float32)
    rainfall = torch.zeros((1, 6, 4, 1), dtype=torch.float32)
    reference = torch.full((1, 6, 3), 0.5, dtype=torch.float32)
    candidates = reference[:, None].expand(1, 3, -1, -1).clone()
    candidates[0, 1, 0:2, 0] = 0.8
    candidates[0, 2, 0:2, 1] = 0.2
    flow = torch.zeros((1, 3), dtype=torch.float32)
    out = model(state, rainfall, reference, candidates, flow, _prepared())
    truth = torch.tensor([[0.0, -800.0, 400.0]], dtype=torch.float32)
    loss, metrics = value_loss_v124(
        out,
        truth,
        scale_m3=1000.0,
        contract=ValueLossContractV124(listwise_weight=0.30),
    )
    assert torch.isfinite(loss)
    assert metrics["listwise"] > 0.0
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(g).all() for g in gradients)
