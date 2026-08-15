from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.edge_physics_current_v128 import EdgePhysicsArtifactV128
from rtc.physics_diagnostics_v128 import conduit_flow_supervision_readiness, node_continuity_proxy
from rtc.step2_differentiable_v128_edge import build_v128_edge_aware_model_from_graph


def _graph():
    return SimpleNamespace(
        node_ids=("n0", "n1", "n2"),
        edge_index=np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
        static_node_features=np.zeros((3, 2), dtype=np.float32),
        actuator_ids=("a0",),
        actuator_upstream=np.asarray([0], dtype=np.int64),
        actuator_downstream=np.asarray([1], dtype=np.int64),
        actuator_physics=np.asarray([[1.0, 0.0, 0.0, 1.0]], dtype=np.float32),
    )


def _artifact(graph):
    return EdgePhysicsArtifactV128(
        edge_index=graph.edge_index.copy(),
        edge_static_features=np.asarray(
            [[0.1, 1.0], [0.1, -1.0], [0.2, 1.0], [0.2, -1.0]],
            dtype=np.float32,
        ),
        edge_static_feature_names=("log_length", "orientation"),
        effective_length_m=np.asarray([100.0, 100.0, 250.0, 250.0], dtype=np.float32),
        physical_link_count=np.ones(4, dtype=np.int32),
    )


def test_edge_aware_v128_rollout_has_finite_action_gradient() -> None:
    graph = _graph()
    artifact = _artifact(graph)
    model = build_v128_edge_aware_model_from_graph(
        graph,
        edge_artifact=artifact,
        state_dim=3,
        rainfall_dim=1,
        delta_state_scale=np.asarray([0.2, 0.2, 0.05], dtype=np.float32),
        delta_flow_scale=np.asarray([0.2], dtype=np.float32),
    )
    settings = torch.full((1, 4, 1), 0.5, requires_grad=True)
    output = model.objective_rollout(
        initial_state=torch.zeros(1, 3, 3),
        rainfall=torch.zeros(1, 4, 3, 1),
        settings=settings,
        previous_actuator_flow=torch.zeros(1, 1),
        actuator_upstream=torch.as_tensor(graph.actuator_upstream),
        actuator_downstream=torch.as_tensor(graph.actuator_downstream),
        actuator_physics=torch.as_tensor(graph.actuator_physics),
        static_node_features=torch.as_tensor(graph.static_node_features),
        edge_index=torch.as_tensor(graph.edge_index),
        flood_rate_index=2,
        priority_indices=None,
        dt_seconds=300.0,
    )
    gradient = torch.autograd.grad(output.optimization_tfv_m3.sum(), settings)[0]
    assert gradient.shape == settings.shape
    assert torch.isfinite(gradient).all()


def test_edge_aware_artifact_rejects_different_runtime_edge_order() -> None:
    graph = _graph()
    artifact = _artifact(graph)
    bad = SimpleNamespace(**graph.__dict__)
    bad.edge_index = graph.edge_index[:, ::-1].copy()
    try:
        artifact.validate(bad)
    except ValueError as exc:
        assert "edge_index" in str(exc)
    else:
        raise AssertionError("edge artifact accepted a different runtime edge order")


def test_continuity_proxy_is_explicitly_diagnostic_only() -> None:
    initial = np.zeros((2, 6), dtype=np.float32)
    states = np.zeros((2, 2, 6), dtype=np.float32)
    states[0, :, 3] = 300.0
    states[0, :, 4] = 1.0
    states[1, :, 3] = 600.0
    states[1, :, 4] = 1.0
    metrics = node_continuity_proxy(initial, states, dt_seconds=300.0)
    assert metrics["training_loss_enabled"] is False
    assert metrics["exact_swmm_mass_balance"] is False
    assert metrics["residual_mae_m3s"] == 0.0


def test_conduit_flow_supervision_requires_authoritative_field() -> None:
    missing = conduit_flow_supervision_readiness({"target_states": np.zeros(1)})
    assert missing["ordinary_conduit_flow_supervision_available"] is False
    present = conduit_flow_supervision_readiness(
        {"target_states": np.zeros(1), "target_conduit_flow_m3s": np.zeros(1)}
    )
    assert present["ordinary_conduit_flow_supervision_available"] is True
