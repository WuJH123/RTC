from __future__ import annotations

from types import SimpleNamespace

import torch

from rtc.step2_control_response_v43 import set_trainable_phase
from rtc.step2_train_response_v431 import (
    current_state_diagnostics_v431,
    d2_single_phase_loss_v431,
    d3_interaction_phase_loss_v431,
    reference_phase_loss_v431,
    validate_fresh_parent_lineage_v431,
)

from test_step2_state_topology_interaction_v43 import _fixture, _forward


def _normalization(state_dim: int, actuators: int):
    return SimpleNamespace(
        state_mean=torch.zeros(state_dim).numpy(),
        state_std=torch.ones(state_dim).numpy(),
        flow_std=torch.ones(actuators).numpy(),
    )


def _scales(state_dim: int, actuators: int):
    return SimpleNamespace(
        state_scale=torch.ones(state_dim).numpy(),
        flow_scale=torch.ones(actuators).numpy(),
        tfv_scale_m3=1000.0,
        tfv_abs_quantiles_m3={"q33": 30.0, "q67": 100.0},
    )


def _batch(output, *, source_kind: str = "D3"):
    return SimpleNamespace(
        source_kind=source_kind,
        true_reference_states_physical=torch.zeros_like(output.reference_states_physical),
        true_reference_flows_physical=torch.zeros_like(output.reference_flows_physical),
        true_delta_states_physical=torch.zeros_like(output.delta_states_physical),
        true_delta_flows_physical=torch.zeros_like(output.delta_flows_physical),
        true_delta_tfv_m3=torch.tensor([[10.0, 40.0]], dtype=output.direct_delta_tfv_m3.dtype),
    )


def test_reference_phase_has_no_counterfactual_gradient():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture()
    candidate[:, 0, :, :2] = (candidate[:, 0, :, :2] + 0.2).clamp(0.0, 1.0)
    output = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    loss, _ = reference_phase_loss_v431(output, _batch(output), _normalization(6, 6))
    set_trainable_phase(model, "reference")
    params = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    grads = torch.autograd.grad(loss, tuple(parameter for _, parameter in params), allow_unused=True)
    names = [name for name, _ in params]
    assert any(g is not None and g.abs().sum() > 0 for n, g in zip(names, grads) if n.startswith("reference_"))
    assert all(g is None or g.abs().sum() == 0 for n, g in zip(names, grads) if not n.startswith(("reference_", "node_static_encoder.", "actuator_static_encoder.", "actuator_identity.")))


def test_reference_phase_loss_ignores_candidate_order():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture()
    output = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    batch = _batch(output)
    loss_a, _ = reference_phase_loss_v431(output, batch, _normalization(6, 6))
    batch_reordered = _batch(output)
    batch_reordered.true_delta_tfv_m3 = batch.true_delta_tfv_m3.flip(1)
    loss_b, _ = reference_phase_loss_v431(output, batch_reordered, _normalization(6, 6))
    torch.testing.assert_close(loss_a, loss_b)


def test_d2_phase_has_no_reference_or_interaction_gradient():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture()
    candidate[:, 0, :, :2] = (candidate[:, 0, :, :2] + 0.2).clamp(0.0, 1.0)
    output = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    set_trainable_phase(model, "d2")
    loss, _ = d2_single_phase_loss_v431(output, _batch(output, source_kind="D2"), _scales(6, 6), _normalization(6, 6))
    params = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    grads = torch.autograd.grad(loss, tuple(parameter for _, parameter in params), allow_unused=True)
    names = [name for name, _ in params]
    assert all(g is None or g.abs().sum() == 0 for n, g in zip(names, grads) if n.startswith(("reference_", "interaction_", "topology_", "direct_interaction_tfv_head.")))
    assert any(g is not None and g.abs().sum() > 0 for n, g in zip(names, grads) if n.startswith("single_"))


def test_d3_phase_changes_only_interaction_parameters():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture()
    candidate[:, 0, :, :2] = (candidate[:, 0, :, :2] + 0.2).clamp(0.0, 1.0)
    set_trainable_phase(model, "d3")
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    output = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    loss, _ = d3_interaction_phase_loss_v431(output, _batch(output), _scales(6, 6), _normalization(6, 6))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    for name, value in model.named_parameters():
        if name.startswith(("reference_", "node_static_encoder.", "actuator_static_encoder.", "actuator_identity.", "temporal_identity.", "single_", "direct_single_tfv_head.")):
            torch.testing.assert_close(value, before[name])
    assert any(not torch.equal(value, before[name]) for name, value in model.named_parameters() if name.startswith(("interaction_", "topology_", "direct_interaction_tfv_head.")))


def test_micro_does_not_load_tiny_checkpoint():
    assert validate_fresh_parent_lineage_v431("parent.pt", "parent.pt", "tiny.pt") is False


def test_tiny_and_micro_share_same_immutable_parent():
    assert validate_fresh_parent_lineage_v431("parent.pt", "parent.pt", "parent.pt") is True


def test_endpoint_local_state_ablation_is_real():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture()
    candidate[:, 0, :, :2] = (candidate[:, 0, :, :2] + 0.2).clamp(0.0, 1.0)
    model.set_topology_ablation(graph_enabled=True, endpoint_state_enabled=True, message_state_enabled=True)
    full = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    model.set_topology_ablation(graph_enabled=True, endpoint_state_enabled=False, message_state_enabled=True)
    ablated = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    assert not torch.equal(full.interaction_delta_states_physical, ablated.interaction_delta_states_physical)


def test_message_local_state_ablation_is_real():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture()
    candidate[:, 0, :, :2] = (candidate[:, 0, :, :2] + 0.2).clamp(0.0, 1.0)
    model.set_topology_ablation(graph_enabled=True, endpoint_state_enabled=True, message_state_enabled=True)
    full = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    model.set_topology_ablation(graph_enabled=True, endpoint_state_enabled=True, message_state_enabled=False)
    ablated = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    assert not torch.equal(full.interaction_delta_states_physical, ablated.interaction_delta_states_physical)


def test_full_local_state_ablation_removes_both_paths():
    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture()
    candidate[:, 0, :, :2] = (candidate[:, 0, :, :2] + 0.2).clamp(0.0, 1.0)
    model.set_topology_ablation(graph_enabled=True, endpoint_state_enabled=False, message_state_enabled=False)
    explicit = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    model.set_topology_ablation(graph_enabled=True, local_state_enabled=False)
    shorthand = _forward(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    torch.testing.assert_close(explicit.interaction_delta_states_physical, shorthand.interaction_delta_states_physical)


def test_current_state_diagnostic_uses_initial_state():
    pair = SimpleNamespace(
        reference={
            "initial_state": torch.tensor([[2.0, 4.0], [1.0, 8.0]]).numpy(),
            "target_states_physical": torch.full((3, 2, 2), 999.0).numpy(),
        }
    )
    norm = SimpleNamespace(state_mean=torch.zeros(2).numpy(), state_std=torch.ones(2).numpy())
    diagnostics = current_state_diagnostics_v431(pair, norm)
    assert diagnostics["source"] == "initial_state"
    assert diagnostics["mean_depth"] < 10.0


def test_future_reference_state_not_used_as_current_state():
    pair = SimpleNamespace(reference={"initial_state": torch.zeros(2, 2).numpy(), "target_states_physical": torch.full((3, 2, 2), 10000.0).numpy()})
    norm = SimpleNamespace(state_mean=torch.zeros(2).numpy(), state_std=torch.ones(2).numpy())
    first = current_state_diagnostics_v431(pair, norm)
    pair.reference["target_states_physical"][:] = -10000.0
    second = current_state_diagnostics_v431(pair, norm)
    for key in ("source", "mean_depth", "high_depth_fraction", "flooding_active_fraction"):
        assert first[key] == second[key]


def test_phased_control_and_topology_model_share_training_contract():
    from rtc.step2_train_response_v431 import TRAINING_CONTRACT_V431

    assert "REFERENCE" in TRAINING_CONTRACT_V431 and "D2" in TRAINING_CONTRACT_V431 and "D3" in TRAINING_CONTRACT_V431
