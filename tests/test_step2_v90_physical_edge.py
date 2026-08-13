from __future__ import annotations

import torch
import pytest

from test_step2_v80 import _graph, _model
from rtc.step2_control_response_v80 import prepare_static_v80
from rtc.step2_control_response_v90 import (
    DirectHydraulicEffectSurrogateV90,
    PhysicalConduitHydraulicEffectSurrogateV90,
)
from rtc.step2_physical_edge_v90 import ConduitPhysicalEdgeAssetsV90
from rtc.step2_v90_contract import DirectHydraulicEffectLossContractV90, LEVEL_C


def _assets(*, parallel: bool = True) -> ConduitPhysicalEdgeAssetsV90:
    # Two directed identities for one conduit, optionally doubled for a
    # physically parallel conduit.  The test deliberately keeps duplicate
    # src/dst rows rather than collapsing them through an adjacency matrix.
    rows = [(0, 1), (1, 0)]
    ids = ["c1", "c1"]
    signs = [1.0, -1.0]
    if parallel:
        rows += [(0, 1), (1, 0)]
        ids += ["c2", "c2"]
        signs += [1.0, -1.0]
    edge_index = torch.tensor(rows, dtype=torch.long).T.contiguous()
    edge_count = edge_index.shape[1]
    degree = torch.zeros(14, dtype=torch.long)
    degree.index_add_(0, edge_index[1], torch.ones(edge_count, dtype=torch.long))
    out = torch.zeros(14, dtype=torch.long)
    out.index_add_(0, edge_index[0], torch.ones(edge_count, dtype=torch.long))
    static = torch.zeros(edge_count, 33)
    if parallel:
        # The second physical conduit has a distinct static hydraulic feature.
        # A correct physical-multiedge operator must not silently discard or
        # average it before its own message is evaluated.
        static[2:, 0] = 2.0
    raw = static.clone()
    return ConduitPhysicalEdgeAssetsV90(
        contract="TEST",
        inp_path="synthetic.inp",
        inp_sha256="0" * 64,
        node_count=14,
        physical_link_count=2 if parallel else 1,
        conduit_physical_link_count=2 if parallel else 1,
        edge_index=edge_index,
        edge_to_link_id=tuple(ids),
        edge_to_link_type=("conduit",) * edge_count,
        orientation_sign=torch.tensor(signs),
        edge_length_m=torch.ones(edge_count),
        static_feature_names=tuple(f"f{i}" for i in range(33)),
        static_features_raw=raw,
        static_features_normalized=static,
        static_normalization_location=torch.zeros(33),
        static_normalization_scale=torch.ones(33),
        static_normalization_sha256="1" * 64,
        in_degree=degree,
        out_degree=out,
        excluded_regulator_link_ids=(),
        excluded_nonconduit_link_ids=(),
        regulator_propagation_edge_count=0,
    )


def _physical_model(*, parallel: bool = True):
    graph, basis, v80 = _model()
    model = PhysicalConduitHydraulicEffectSurrogateV90(
        reference_model=v80.reference_model,
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        state_delta_scale=v80.state_delta_scale,
        flow_delta_scale=v80.flow_delta_scale,
        physics_dim=prepare_static_v80(graph).base.actuator_physics.shape[1],
        node_static_dim=graph.static_node_features.shape[1],
        actuator_count=109,
        physical_edge_assets=_assets(parallel=parallel),
        head_scale_m=1.0,
        depth_scale_m=1.0,
        gradient_scale=1.0,
        contract=DirectHydraulicEffectLossContractV90(hidden_dim=16, graph_blocks=2),
    )
    # Fix a transparent nonzero message operator.  With the two parallel rows,
    # the directed ``index_add`` at node 1 must receive two contributions.
    with torch.no_grad():
        model.physical_edge_condition[0].weight.zero_()
        model.physical_edge_source.weight.copy_(torch.eye(16))
        model.physical_edge_destination.weight.zero_()
        model.physical_edge_output.weight.copy_(torch.eye(16))
        model.multiscale_gate.weight.zero_()
        model.multiscale_gate.bias.zero_()
    return graph, model


def test_v90_physical_multiedge_diffusion_preserves_parallel_identity_and_exact_zero():
    graph, model = _physical_model(parallel=True)
    prepared = prepare_static_v80(graph)
    seed = torch.zeros(1, 1, 1, 14, 16)
    reference = torch.zeros(1, 1, 14, 6)
    zero, _, hops = model._multiscale_diffuse_v90(
        seed, prepared, reference_states_physical=reference
    )
    assert tuple(hops) == (1, 2, 4, 8)
    assert torch.equal(zero, seed)

    seed[..., 0, 0] = 1.0
    with torch.no_grad():
        model.physical_edge_condition[0].weight.zero_()
        model.physical_edge_condition[0].weight[:, 0] = 1.0
    parallel, _, _ = model._multiscale_diffuse_v90(
        seed, prepared, reference_states_physical=reference
    )
    _, single = _physical_model(parallel=False)
    with torch.no_grad():
        single.physical_edge_condition[0].weight.zero_()
        single.physical_edge_condition[0].weight[:, 0] = 1.0
    single_out, _, _ = single._multiscale_diffuse_v90(
        seed, prepared, reference_states_physical=reference
    )
    assert model.physical_edge_assets.directed_edge_count == 4
    assert len(set(model.physical_edge_assets.edge_to_link_id)) == 2
    assert parallel[..., 1, 0].abs().item() > single_out[..., 1, 0].abs().item()


def test_v90_physical_diffusion_uses_directed_causal_reference_hydraulics():
    graph, model = _physical_model(parallel=False)
    prepared = prepare_static_v80(graph)
    seed = torch.zeros(1, 1, 1, 14, 16)
    seed[..., 0, 0] = 1.0
    flat_reference = torch.zeros(1, 1, 14, 6)
    head_gradient = flat_reference.clone()
    head_gradient[..., 0, 1] = 2.0
    # Make the condition explicitly read the delta-head feature (static 33 +
    # dynamic feature position 4), proving direction-dependent causal context
    # affects the physical message without any target/future input.
    with torch.no_grad():
        model.physical_edge_condition[0].weight.zero_()
        model.physical_edge_condition[0].weight[:, 33 + 4] = 1.0
    flat, _, _ = model._multiscale_diffuse_v90(
        seed, prepared, reference_states_physical=flat_reference
    )
    graded, _, _ = model._multiscale_diffuse_v90(
        seed, prepared, reference_states_physical=head_gradient
    )
    assert not torch.equal(flat, graded)


def test_v90_physical_conduit_control_rejects_oracle_conditioning_level():
    graph, basis, v80 = _model()
    with pytest.raises(ValueError, match="predicted-reference-only"):
        PhysicalConduitHydraulicEffectSurrogateV90(
            reference_model=v80.reference_model,
            temporal_basis=basis.temporal_basis,
            control_block_steps=basis.horizon.control_block_steps,
            state_delta_scale=v80.state_delta_scale,
            flow_delta_scale=v80.flow_delta_scale,
            physics_dim=prepare_static_v80(graph).base.actuator_physics.shape[1],
            node_static_dim=graph.static_node_features.shape[1],
            actuator_count=109,
            conditioning_level=LEVEL_C,
            physical_edge_assets=_assets(),
            head_scale_m=1.0,
            depth_scale_m=1.0,
            gradient_scale=1.0,
            contract=DirectHydraulicEffectLossContractV90(hidden_dim=16, graph_blocks=2),
        )


def test_v90_base_ladder_keeps_oracle_level_available():
    graph, basis, v80 = _model()
    model = DirectHydraulicEffectSurrogateV90(
        reference_model=v80.reference_model,
        temporal_basis=basis.temporal_basis,
        control_block_steps=basis.horizon.control_block_steps,
        state_delta_scale=v80.state_delta_scale,
        flow_delta_scale=v80.flow_delta_scale,
        physics_dim=prepare_static_v80(graph).base.actuator_physics.shape[1],
        node_static_dim=graph.static_node_features.shape[1],
        actuator_count=109,
        conditioning_level=LEVEL_C,
        contract=DirectHydraulicEffectLossContractV90(hidden_dim=16, graph_blocks=2),
    )
    assert model.conditioning_level == LEVEL_C


def test_v90_physical_activation_recomputation_preserves_output_and_gradients():
    graph, checkpointed = _physical_model(parallel=True)
    _, direct = _physical_model(parallel=True)
    direct.load_state_dict(checkpointed.state_dict())
    prepared = prepare_static_v80(graph)
    reference = torch.randn(1, 2, 14, 6)
    seed_checkpointed = torch.randn(1, 3, 2, 14, 16, requires_grad=True)
    seed_direct = seed_checkpointed.detach().clone().requires_grad_(True)

    checkpointed_out, _, _ = checkpointed._multiscale_diffuse_v90(
        seed_checkpointed, prepared, reference_states_physical=reference
    )
    direct_out, _, _ = direct._physical_multiscale_v90(
        seed_direct,
        prepared,
        reference_states_physical=reference,
        activation_checkpointing=False,
    )
    checkpointed_out.square().mean().backward()
    direct_out.square().mean().backward()

    assert torch.allclose(checkpointed_out, direct_out, rtol=1e-5, atol=1e-6)
    assert torch.allclose(seed_checkpointed.grad, seed_direct.grad, rtol=1e-5, atol=1e-6)
    for (name_a, parameter_a), (name_b, parameter_b) in zip(
        checkpointed.named_parameters(), direct.named_parameters(), strict=True
    ):
        assert name_a == name_b
        if parameter_a.requires_grad and parameter_a.grad is not None:
            assert torch.allclose(parameter_a.grad, parameter_b.grad, rtol=1e-5, atol=1e-6)
