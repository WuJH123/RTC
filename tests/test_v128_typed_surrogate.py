from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rtc.checkpoint_v128 import (
    V128_CHECKPOINT_CONTRACT,
    load_step2_v128,
    save_step2_v128,
)
from rtc.step2_differentiable_v128 import (
    TypedActuatorMessageSurrogateV128,
    V128_STEP2_CONTRACT,
)
from rtc.step2_train_response_v60 import InputNormalizationV60


def _graph() -> SimpleNamespace:
    return SimpleNamespace(
        node_ids=("n0", "n1", "n2"),
        static_node_feature_names=("f0", "f1"),
        actuator_ids=("pump_A", "orifice_B"),
        actuator_physics_feature_names=(
            "is_pump",
            "is_orifice",
            "min_setting",
            "max_setting",
        ),
        system_units="SI",
        edge_index=np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
        static_node_features=np.zeros((3, 2), dtype=np.float32),
        actuator_upstream=np.asarray([0, 0], dtype=np.int64),
        actuator_downstream=np.asarray([1, 1], dtype=np.int64),
        actuator_physics=np.asarray(
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
    )


def _model() -> TypedActuatorMessageSurrogateV128:
    torch.manual_seed(13)
    return TypedActuatorMessageSurrogateV128(
        state_dim=3,
        rainfall_dim=1,
        node_static_dim=2,
        actuator_physics_dim=4,
        actuator_count=2,
        hidden_dim=16,
        actuator_embedding_dim=4,
        action_message_dim=6,
        delta_state_scale=torch.tensor([0.2, 0.2, 0.05]),
        delta_flow_scale=torch.tensor([0.2, 0.2]),
    )


def test_v128_equal_scalar_setting_sum_does_not_alias_typed_actions() -> None:
    model = _model()
    graph = _graph()
    state = torch.zeros(1, 3, 3)
    physics = torch.as_tensor(graph.actuator_physics)
    physics_norm, identity = model.actuator.prepare_static(physics, batch_size=1)
    up = torch.as_tensor(graph.actuator_upstream)
    down = torch.as_tensor(graph.actuator_downstream)
    previous = torch.zeros(1, 2)
    predicted = torch.tensor([[0.1, 0.2]])
    response = torch.tensor([[0.6, 0.7]])

    # Both actions have the same raw setting sum at n0/n1.  The V127 two-scalar direct
    # context would therefore be identical.  V128 must preserve which physical device
    # received which target.
    first = model._typed_action_context(
        state=state,
        setting=torch.tensor([[0.8, 0.2]]),
        previous_flow=previous,
        predicted_flow=predicted,
        responsiveness=response,
        upstream=up,
        downstream=down,
        physics_norm=physics_norm,
        identity_embedding=identity,
    )
    second = model._typed_action_context(
        state=state,
        setting=torch.tensor([[0.2, 0.8]]),
        previous_flow=previous,
        predicted_flow=predicted,
        responsiveness=response,
        upstream=up,
        downstream=down,
        physics_norm=physics_norm,
        identity_embedding=identity,
    )
    assert first.shape == second.shape == (1, 3, 12)
    assert not torch.allclose(first, second)


def test_v128_smooth_tfv_retains_finite_action_gradient() -> None:
    model = _model()
    graph = _graph()
    settings = torch.full((1, 4, 2), 0.5, requires_grad=True)
    result = model.objective_rollout(
        initial_state=torch.zeros(1, 3, 3),
        rainfall=torch.zeros(1, 4, 3, 1),
        settings=settings,
        previous_actuator_flow=torch.zeros(1, 2),
        actuator_upstream=torch.as_tensor(graph.actuator_upstream),
        actuator_downstream=torch.as_tensor(graph.actuator_downstream),
        actuator_physics=torch.as_tensor(graph.actuator_physics),
        static_node_features=torch.as_tensor(graph.static_node_features),
        edge_index=torch.as_tensor(graph.edge_index),
        flood_rate_index=2,
        priority_indices=torch.tensor([0]),
        dt_seconds=300.0,
    )
    gradient = torch.autograd.grad(result.optimization_tfv_m3.sum(), settings)[0]
    assert gradient.shape == settings.shape
    assert torch.isfinite(gradient).all()


def test_v128_checkpoint_round_trip_and_contract_isolation(tmp_path) -> None:
    model = _model()
    graph = _graph()
    normalization = InputNormalizationV60(
        state_mean=np.zeros(3, dtype=np.float32),
        state_std=np.ones(3, dtype=np.float32),
        rainfall_mean=np.zeros(1, dtype=np.float32),
        rainfall_std=np.ones(1, dtype=np.float32),
        flow_mean=np.zeros(2, dtype=np.float32),
        flow_std=np.ones(2, dtype=np.float32),
    )
    path = save_step2_v128(
        tmp_path / "v128.pt",
        model=model,
        graph=graph,
        input_normalization=normalization,
        training_report={"contract": "test"},
        lineage={"swmm_engine_version": "test"},
    )
    loaded, payload = load_step2_v128(path, graph=graph, device="cpu")
    assert isinstance(loaded, TypedActuatorMessageSurrogateV128)
    assert payload["checkpoint_contract"] == V128_CHECKPOINT_CONTRACT
    assert payload["step2_contract"] == V128_STEP2_CONTRACT

    fake = tmp_path / "not_v128.pt"
    torch.save({"checkpoint_contract": "PROJECT7_V127_STEP2_CHECKPOINT_V4_SEMANTIC_COMPATIBILITY"}, fake)
    with pytest.raises(ValueError, match="not a V128"):
        load_step2_v128(fake, graph=graph, device="cpu")
