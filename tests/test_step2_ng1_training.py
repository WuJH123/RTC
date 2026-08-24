from __future__ import annotations

from types import SimpleNamespace

import torch

from rtc.step2_tfv_value_ng1 import (
    NG1ProcessAwareDirectTFVValueModel,
    build_control_interaction_graph,
)
from rtc.step2_tfv_value_training_ng1 import (
    freeze_ng1_main,
    ng1_d3_group_loss,
    ng1_main_parameter_sha256,
)
from tests.test_step2_ng1_value import _graph


def _batch() -> tuple[
    NG1ProcessAwareDirectTFVValueModel,
    SimpleNamespace,
    dict[str, torch.Tensor],
]:
    torch.manual_seed(19)
    graph = _graph()
    model = NG1ProcessAwareDirectTFVValueModel(
        state_dim=3,
        rainfall_dim=1,
        actuator_physics_dim=3,
        target_scale_m3=5000.0,
        interaction_graph=build_control_interaction_graph(graph),
    )
    reference = torch.full((1, 72, 109), 0.5)
    candidates = reference.repeat(3, 1, 1)
    candidates[1, :12, 5] += 0.2
    candidates[1, :12, 17] -= 0.1
    candidates[2, :12, 5] += 0.1
    candidates[2, :12, 17] -= 0.2
    batch = SimpleNamespace(
        initial_state=torch.randn(1, 8, 3),
        rainfall=torch.rand(1, 72, 8, 1),
        reference_settings=reference,
        candidate_settings=candidates[None],
        previous_actuator_flow=torch.randn(1, 109),
        true_delta_tfv_m3=torch.tensor([[-1200.0, 800.0, -400.0]]),
    )
    static = {
        "up": torch.as_tensor(graph.actuator_upstream),
        "down": torch.as_tensor(graph.actuator_downstream),
        "physics": torch.as_tensor(graph.actuator_physics),
    }
    return model, batch, static


def test_d3_optimizer_step_cannot_change_frozen_main_parameters() -> None:
    model, batch, static = _batch()
    freeze_ng1_main(model)
    before = ng1_main_parameter_sha256(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert parameters
    optimizer = torch.optim.AdamW(parameters, lr=1.0e-3)
    loss, metrics = ng1_d3_group_loss(
        model,
        batch,
        indices=torch.arange(3),
        graph_tensors=static,
        scale_m3=model.target_scale_m3,
        tau_m3=torch.tensor(1000.0),
    )
    assert torch.isfinite(loss)
    assert metrics["branches"] == 3.0
    assert "selection_regret" in metrics
    assert "oracle_best_margin" not in metrics
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert before == ng1_main_parameter_sha256(model)
    assert all(
        not p.requires_grad
        for name, p in model.named_parameters()
        if name not in model.interaction_parameter_names()
    )
