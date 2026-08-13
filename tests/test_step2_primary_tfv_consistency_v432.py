from __future__ import annotations

import hashlib
import json

import pytest
import torch

from rtc.step2_control_response_v432 import (
    DifferentiableCounterfactualResponseModelV432,
    nodewise_tfv_from_contributions_v432,
    project_auxiliary_gradient_v432,
)
from rtc.step2_train_response_v432 import (
    classify_first_degradation_v432,
    compare_d2_prediction_snapshots_v432,
    resolve_best_d2_checkpoint_v432,
)

from test_step2_state_topology_interaction_v43 import _fixture


def _nodewise_fixture(*, candidates: int = 1, horizon: int = 8):
    base = _fixture(candidates=candidates, horizon=horizon)
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = base
    # Rebuild with the same dimensions and static contract, enabling only the
    # V4.3.2 nodewise direct-TFV aggregation path.
    nodewise = DifferentiableCounterfactualResponseModelV432(
        state_dim=model.state_dim,
        rainfall_dim=1,
        node_static_dim=prepared.static_node_features.shape[1],
        actuator_physics_dim=prepared.actuator_physics.shape[1],
        hidden_dim=model.hidden_dim,
        actuator_count=model.actuator_count,
        actuator_embedding_dim=model.actuator_identity.embedding_dim,
        temporal_embedding_dim=model.temporal_identity.embedding_dim,
        state_mean=model.state_mean,
        state_std=model.state_std,
        flow_std=model.flow_std,
        d2_state_scale=model.d2_state_scale,
        d3_state_scale=model.d3_state_scale,
        d2_flow_scale=model.d2_flow_scale,
        d3_flow_scale=model.d3_flow_scale,
        d2_tfv_scale=float(model.d2_tfv_scale),
        d3_tfv_scale=float(model.d3_tfv_scale),
        max_horizon_steps=model.max_horizon_steps,
        effect_rank=model.effect_rank,
        topology_blocks=model.topology_blocks,
    )
    nodewise.load_state_dict(model.state_dict(), strict=False)
    return nodewise, prepared, initial, rainfall, reference, candidate, previous, elapsed


def test_primary_gradient_unchanged():
    primary = torch.tensor([2.0, -1.0, 0.5])
    auxiliary = torch.tensor([1.0, 0.0, 2.0])
    projected = project_auxiliary_gradient_v432(primary, auxiliary)
    torch.testing.assert_close(projected, auxiliary)


def test_conflicting_aux_gradient_is_projected():
    primary = torch.tensor([1.0, 0.0])
    auxiliary = torch.tensor([-2.0, 3.0])
    projected = project_auxiliary_gradient_v432(primary, auxiliary)
    assert torch.dot(projected, primary).item() == pytest.approx(0.0, abs=1e-7)
    assert projected[1].item() == pytest.approx(3.0)


def test_aligned_aux_gradient_is_preserved():
    primary = torch.tensor([1.0, 2.0])
    auxiliary = torch.tensor([0.5, 1.0])
    torch.testing.assert_close(
        project_auxiliary_gradient_v432(primary, auxiliary), auxiliary
    )


def test_zero_primary_gradient_fails_closed():
    with pytest.raises(RuntimeError, match="zero primary"):
        project_auxiliary_gradient_v432(torch.zeros(3), torch.ones(3))


def test_projected_gradient_is_finite_and_not_negative_to_primary():
    primary = torch.tensor([1.0, -2.0, 0.25])
    auxiliary = torch.tensor([-3.0, 0.5, 1.0])
    projected = project_auxiliary_gradient_v432(primary, auxiliary)
    assert torch.isfinite(projected).all()
    assert torch.dot(projected, primary).item() >= -1e-7


def test_nodewise_tfv_zero_action_exact_zero():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _nodewise_fixture()
    output = model.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D3"
    )
    assert torch.equal(
        output.direct_interaction_delta_tfv_m3,
        torch.zeros_like(output.direct_interaction_delta_tfv_m3),
    )


def test_nodewise_tfv_single_action_interaction_zero():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _nodewise_fixture()
    candidate[:, 0, 2:, 0] = (candidate[:, 0, 2:, 0] + 0.2).clamp(0.0, 1.0)
    output = model.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D3"
    )
    assert torch.equal(
        output.direct_interaction_delta_tfv_m3,
        torch.zeros_like(output.direct_interaction_delta_tfv_m3),
    )


def test_nodewise_tfv_multi_action_nonzero():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _nodewise_fixture()
    candidate[:, 0, 2:, :2] = (candidate[:, 0, 2:, :2] + 0.2).clamp(0.0, 1.0)
    output = model.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D3"
    )
    assert output.direct_interaction_delta_tfv_m3.abs().sum() > 0.0


def test_nodewise_tfv_future_action_causality():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _nodewise_fixture(horizon=8)
    candidate[:, 0, 5:, :2] = (candidate[:, 0, 5:, :2] + 0.2).clamp(0.0, 1.0)
    output = model.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D3"
    )
    assert torch.equal(
        output.interaction_delta_states_physical[:, :, :5],
        torch.zeros_like(output.interaction_delta_states_physical[:, :, :5]),
    )


def test_nodewise_sum_not_mean_pooling():
    contributions = torch.tensor([[[1.0, 2.0, 3.0]]])
    result = nodewise_tfv_from_contributions_v432(contributions)
    assert result.item() == pytest.approx(6.0)
    assert result.item() != pytest.approx(contributions.mean().item())


def test_all_d2_predictions_invariant_after_d3():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _nodewise_fixture()
    candidate[:, 0, 2:, :2] = (candidate[:, 0, 2:, :2] + 0.2).clamp(0.0, 1.0)
    before = model.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D2"
    )
    before_tensor = torch.cat(
        (
            before.delta_states_physical.flatten(),
            before.delta_flows_physical.flatten(),
            before.direct_delta_tfv_m3.flatten(),
            before.trajectory_delta_tfv_m3.flatten(),
        )
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("interaction_", "topology_", "direct_interaction_tfv_head.")))
    after_d3 = model.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D3"
    )
    loss = after_d3.direct_delta_tfv_m3.square().mean() + after_d3.trajectory_delta_tfv_m3.square().mean()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    after = model.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D2"
    )
    after_tensor = torch.cat(
        (
            after.delta_states_physical.flatten(),
            after.delta_flows_physical.flatten(),
            after.direct_delta_tfv_m3.flatten(),
            after.trajectory_delta_tfv_m3.flatten(),
        )
    )
    torch.testing.assert_close(before_tensor, after_tensor, atol=1e-7, rtol=1e-6)


def test_phase_snapshot_hash_is_stable_for_same_parameters():
    model, *_ = _nodewise_fixture()
    def digest():
        h = hashlib.sha256()
        for name, parameter in model.named_parameters():
            h.update(name.encode())
            h.update(parameter.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    assert digest() == digest()


def test_phase_transition_classification_is_ordered():
    stages = {"A0": {"rank": 0.7}, "A1": {"rank": 0.6}, "A2": {"rank": 0.8}, "A3": {"rank": 0.9}}
    assert classify_first_degradation_v432(stages) == "REFERENCE"


def test_phase_transition_none_when_metric_is_preserved():
    stages = {"A0": {"rank": 0.7}, "A1": {"rank": 0.7}, "A2": {"rank": 0.7}, "A3": {"rank": 0.7}}
    assert classify_first_degradation_v432(stages) == "NONE"


def test_d2_snapshot_comparison_is_strict():
    value = {"g": {"x": torch.ones(2)}}
    assert compare_d2_prediction_snapshots_v432(value, value)["prediction_invariant"]
    changed = {"g": {"x": torch.zeros(2)}}
    assert not compare_d2_prediction_snapshots_v432(value, changed)["prediction_invariant"]


def test_best_d2_checkpoint_resolver_reads_stage_artifact(tmp_path):
    checkpoint = tmp_path / "v42_12_group_micro.pt"
    checkpoint.write_bytes(b"immutable checkpoint fixture")
    path = tmp_path / "stage_result.json"
    path.write_text(
        json.dumps(
            {
                "training": {
                    "checkpoint": str(checkpoint),
                    "best_epoch": 22,
                    "selection_policy": "d3_magnitude",
                },
                "group_metrics": [{"source_kind": "D2"}],
            }
        ),
        encoding="utf-8",
    )
    resolved = resolve_best_d2_checkpoint_v432(path)
    assert resolved["best_epoch"] == 22
    assert resolved["sha256"]
    assert resolved["checkpoint"].endswith("v42_12_group_micro.pt")
