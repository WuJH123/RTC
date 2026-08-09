from __future__ import annotations

import torch

from rtc.models import Rollout
from rtc.tfv_mpc import ContinuousTFVFirstMPC, _project_block_settings_


class _BoundaryWorldModel(torch.nn.Module):
    """One-node differentiable toy: larger setting linearly reduces flooding rate."""

    def rollout(
        self,
        initial_state,
        rainfall,
        settings,
        previous_actuator_flow,
        actuator_upstream,
        actuator_downstream,
        actuator_physics,
        static_node_features,
        edge_index,
    ):
        flood = 1.0 - 0.8 * settings[..., 0]
        zeros = torch.zeros_like(flood)
        states = torch.stack([zeros, zeros, flood], dim=-1).unsqueeze(2)
        flows = settings.clone()
        return Rollout(
            states=states,
            actuator_flows=flows,
            responsiveness=torch.ones_like(flows),
        )


def test_direct_setting_mpc_can_activate_actuator_from_exact_zero() -> None:
    model = _BoundaryWorldModel()
    mpc = ContinuousTFVFirstMPC(
        model,
        depth_index=0,
        flood_rate_index=2,
        priority_indices=None,
        dt_seconds=300.0,
        movement_tiebreak=0.0,
    )
    result = mpc.optimize(
        initial_state=torch.tensor([[[0.0, 0.0, 1.0]]]),
        rainfall_scenarios=torch.zeros(1, 4, 1, 1),
        current_settings=torch.tensor([0.0]),
        fallback_settings=torch.zeros(1, 4, 1),
        previous_actuator_flow=torch.zeros(1, 1),
        actuator_upstream=torch.tensor([0]),
        actuator_downstream=torch.tensor([0]),
        actuator_physics=torch.zeros(1, 1, 1),
        static_node_features=torch.zeros(1, 1),
        edge_index=torch.tensor([[0], [0]]),
        iterations=30,
        learning_rate=0.1,
        control_block_steps=1,
    )
    assert result.candidate_valid
    assert float(result.settings[0, 0]) > 0.1


def test_future_blocks_are_sequentially_rate_feasible() -> None:
    values = torch.nn.Parameter(torch.tensor([[[0.9], [0.9], [0.0], [1.0]]]))
    _project_block_settings_(
        values,
        current_settings=torch.tensor([0.0]),
        max_delta_per_update=0.2,
    )
    path = values.detach()[0, :, 0]
    assert torch.all(path >= 0.0) and torch.all(path <= 1.0)
    previous = torch.cat([torch.tensor([0.0]), path[:-1]])
    assert torch.all(torch.abs(path - previous) <= 0.200001)
    assert torch.allclose(path, torch.tensor([0.2, 0.4, 0.2, 0.4]), atol=1e-6)
