from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rtc.step2_edge_hydraulic_v44 import (
    STATIC_FEATURE_NAMES_V44,
    PhysicalLinkV44,
    build_physical_directed_edge_lineage_v44,
    causal_edge_dynamic_features_v44,
    normalize_edge_static_features_v44,
    parse_frozen_inp_physical_links_v44,
)
from rtc.step2_control_response_v433 import DifferentiableCounterfactualResponseModelV433
from rtc.step2_control_response_v44 import (
    DifferentiableCounterfactualResponseModelV44,
    edge_hydraulic_parameter_names_v44,
    set_trainable_edge_hydraulic_v44,
)


@pytest.fixture()
def tiny_inp(tmp_path: Path) -> Path:
    path = tmp_path / "lineage.inp"
    path.write_text(
        """[OPTIONS]\nFLOW_UNITS CMS\n\n[JUNCTIONS]\nN1 0 0\nN2 0 0\nN3 0 0\n\n[OUTFALLS]\nN4 0 FREE\n\n[CONDUITS]\nC1 N1 N2 100 0.013 1 1 0 0\nC2 N1 N2 120 0.014 1 1 0 0\nC3 N2 N3 80 0.012 1 1 0 0\n\n[PUMPS]\nP1 N3 N4 CURVE1 0 0\n\n[ORIFICES]\nO1 N1 N3 SIDE 1 0.5 NO 0\n\n[WEIRS]\nW1 N2 N4 TRANSVERSE 1 1 0 NO 0 0 NO\n\n[OUTLETS]\nL1 N3 N4 0 FUNCTIONAL 0\n\n[XSECTIONS]\nC1 CIRCULAR 1 0 0 1 1\nC2 RECT_OPEN 2 1 0 1 1\nC3 CIRCULAR 1 0 0 1 1\n\n[LOSSES]\nC1 0.1 0.2 0.0 NO\n\n[CURVES]\nCURVE1 PUMP4 0 0\n""",
        encoding="utf-8",
    )
    return path


def test_parallel_links_are_not_silently_collapsed(tiny_inp: Path):
    links = parse_frozen_inp_physical_links_v44(tiny_inp, ("N1", "N2", "N3", "N4"))
    assert len(links) == 7
    assert sum(link.link_type == "conduit" for link in links) == 3
    assert sum(link.link_type == "pump" for link in links) == 1
    assert sum(link.link_type == "orifice" for link in links) == 1
    assert sum(link.link_type == "weir" for link in links) == 1
    assert sum(link.link_type == "outlet" for link in links) == 1
    assert sum(link.unordered_node_pair == ("N1", "N2") for link in links) == 2


def test_every_physical_edge_has_link_lineage(tiny_inp: Path):
    links = parse_frozen_inp_physical_links_v44(tiny_inp, ("N1", "N2", "N3", "N4"))
    lineage = build_physical_directed_edge_lineage_v44(links, {name: i for i, name in enumerate(("N1", "N2", "N3", "N4"))})
    assert lineage.edge_index.shape == (2, 14)
    assert len(lineage.edge_to_link_id) == 14
    assert len(set(lineage.edge_to_link_id)) == 7
    assert set(lineage.orientation_signs) == {-1, 1}


def test_directed_reverse_edges_share_static_link_identity_and_flip_orientation(tiny_inp: Path):
    links = parse_frozen_inp_physical_links_v44(tiny_inp, ("N1", "N2", "N3", "N4"))
    lineage = build_physical_directed_edge_lineage_v44(links, {name: i for i, name in enumerate(("N1", "N2", "N3", "N4"))})
    for link_id in set(lineage.edge_to_link_id):
        indices = [i for i, value in enumerate(lineage.edge_to_link_id) if value == link_id]
        assert len(indices) == 2
        assert lineage.orientation_signs[indices[0]] == -lineage.orientation_signs[indices[1]]
        assert lineage.edge_static_features[indices[0]].tolist() == lineage.edge_static_features[indices[1]].tolist()


def test_edge_static_features_finite_and_normalized_train_only(tiny_inp: Path):
    links = parse_frozen_inp_physical_links_v44(tiny_inp, ("N1", "N2", "N3", "N4"))
    conduit = next(link for link in links if link.link_id == "C1")
    assert conduit.static_features[0] == pytest.approx(100.0)
    assert conduit.static_features[1] == pytest.approx(0.013)
    assert conduit.shape == "CIRCULAR"
    assert conduit.barrels == pytest.approx(1.0)
    lineage = build_physical_directed_edge_lineage_v44(links, {name: i for i, name in enumerate(("N1", "N2", "N3", "N4"))})
    normalized, stats = normalize_edge_static_features_v44(lineage.edge_static_features)
    assert np.isfinite(normalized).all()
    assert np.isfinite(stats.location).all()
    assert np.isfinite(stats.scale).all()


def test_dynamic_edge_features_are_causal_and_do_not_require_link_flow(tiny_inp: Path):
    links = parse_frozen_inp_physical_links_v44(tiny_inp, ("N1", "N2", "N3", "N4"))
    lineage = build_physical_directed_edge_lineage_v44(links, {name: i for i, name in enumerate(("N1", "N2", "N3", "N4"))})
    node_context = np.zeros((2, 4, 3), dtype=np.float32)
    node_context[:, :, 0] = np.arange(4, dtype=np.float32)
    dynamic = causal_edge_dynamic_features_v44(
        node_context,
        lineage,
        current_head=node_context[..., 0],
        current_depth=node_context[..., 1],
    )
    assert dynamic.shape == (2, 14, 4)
    assert np.isfinite(dynamic).all()
    assert "link_flow" not in lineage.dynamic_feature_names


def test_future_truth_is_not_an_input_to_dynamic_edge_features(tiny_inp: Path):
    links = parse_frozen_inp_physical_links_v44(tiny_inp, ("N1", "N2", "N3", "N4"))
    lineage = build_physical_directed_edge_lineage_v44(links, {name: i for i, name in enumerate(("N1", "N2", "N3", "N4"))})
    assert lineage.uses_future_truth is False
    assert lineage.uses_online_link_flow is False


def _model_fixture(*, horizon: int = 8, candidates: int = 2):
    from test_step2_state_topology_interaction_v43 import _fixture

    base, prepared, initial, rainfall, reference, candidate, previous, elapsed = _fixture(
        horizon=horizon, candidates=candidates
    )
    kwargs = dict(
        state_dim=base.state_dim,
        rainfall_dim=1,
        node_static_dim=prepared.static_node_features.shape[1],
        actuator_physics_dim=prepared.actuator_physics.shape[1],
        hidden_dim=base.hidden_dim,
        actuator_count=base.actuator_count,
        actuator_embedding_dim=base.actuator_identity.embedding_dim,
        temporal_embedding_dim=base.temporal_identity.embedding_dim,
        state_mean=base.state_mean,
        state_std=base.state_std,
        flow_std=base.flow_std,
        d2_state_scale=base.d2_state_scale,
        d3_state_scale=base.d3_state_scale,
        d2_flow_scale=base.d2_flow_scale,
        d3_flow_scale=base.d3_flow_scale,
        d2_tfv_scale=float(base.d2_tfv_scale),
        d3_tfv_scale=float(base.d3_tfv_scale),
        max_horizon_steps=base.max_horizon_steps,
        effect_rank=base.effect_rank,
        topology_blocks=base.topology_blocks,
    )
    v433 = DifferentiableCounterfactualResponseModelV433(**kwargs)
    v433.load_state_dict(base.state_dict(), strict=False)
    v44 = DifferentiableCounterfactualResponseModelV44(
        **kwargs, edge_feature_dim=len(STATIC_FEATURE_NAMES_V44)
    )
    v44.load_state_dict(v433.state_dict(), strict=False)
    node_ids = tuple(f"N{i}" for i in range(prepared.static_node_features.shape[0]))
    links = tuple(
        PhysicalLinkV44(
            link_id=f"L{i}",
            link_type="conduit",
            from_node=node_ids[int(src)],
            to_node=node_ids[int(dst)],
            static_features=np.r_[1.0, np.zeros(len(STATIC_FEATURE_NAMES_V44) - 1)].astype(np.float32),
            shape="CIRCULAR",
            raw_geometry=(1.0, 0.0, 0.0, 0.0),
            barrels=1.0,
        )
        for i, (src, dst) in enumerate(prepared.edge_index[:, ::2].t().tolist())
    )
    lineage = build_physical_directed_edge_lineage_v44(links, {name: i for i, name in enumerate(node_ids)})
    normalized, _ = normalize_edge_static_features_v44(lineage.edge_static_features)
    v44.configure_edge_hydraulic_v44(lineage, normalized)
    return v433, v44, prepared, initial, rainfall, reference, candidate, previous, elapsed


def _forward_v44(model, prepared, initial, rainfall, reference, candidate, previous, elapsed):
    return model.forward_group(
        initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D3"
    )


def test_v44_zero_init_equals_v433():
    baseline, model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _model_fixture()
    candidate[:, 0, :, :2] = (candidate[:, 0, :, :2] + 0.2).clamp(0.0, 1.0)
    baseline_out = baseline.forward_group(initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D3")
    model.edge_hydraulic_residual_active = True
    edge_out = _forward_v44(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    torch.testing.assert_close(edge_out.direct_delta_tfv_m3, baseline_out.direct_delta_tfv_m3, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(edge_out.delta_states_physical, baseline_out.delta_states_physical, atol=1e-7, rtol=1e-6)


def test_edge_residual_zero_action_and_single_action():
    baseline, model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _model_fixture(candidates=1)
    model.edge_hydraulic_residual_active = True
    zero = _forward_v44(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    model.edge_hydraulic_residual_active = False
    zero_off = _forward_v44(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    torch.testing.assert_close(zero.direct_delta_tfv_m3, zero_off.direct_delta_tfv_m3, atol=1e-7, rtol=1e-6)
    candidate[:, 0, 2:, 0] = (candidate[:, 0, 2:, 0] + 0.2).clamp(0.0, 1.0)
    model.edge_hydraulic_residual_active = True
    single = _forward_v44(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    model.edge_hydraulic_residual_active = False
    single_off = _forward_v44(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    torch.testing.assert_close(single.direct_delta_tfv_m3, single_off.direct_delta_tfv_m3, atol=1e-7, rtol=1e-6)


def test_edge_residual_gradient_finite_and_nonzero_for_multi_action():
    _, model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _model_fixture()
    candidate[:, 0, :, :2] = (candidate[:, 0, :, :2] + 0.2).clamp(0.0, 1.0)
    set_trainable_edge_hydraulic_v44(model, enabled=True)
    output = _forward_v44(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    objective = output.direct_delta_tfv_m3.sum() + output.trajectory_delta_tfv_m3.sum()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = torch.autograd.grad(objective, parameters, allow_unused=True)
    assert all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None) > 0.0


def test_edge_parameter_partition_is_explicit():
    _, model, *_ = _model_fixture()
    names = edge_hydraulic_parameter_names_v44(model)
    assert names
    set_trainable_edge_hydraulic_v44(model, enabled=True)
    assert all(parameter.requires_grad == name.startswith("edge_hydraulic_") for name, parameter in model.named_parameters())


def test_d2_prediction_invariant_after_edge_optimizer_step():
    _, model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _model_fixture()
    candidate[:, 0, :, :2] = (candidate[:, 0, :, :2] + 0.2).clamp(0.0, 1.0)
    before = model.forward_group(initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D2")
    set_trainable_edge_hydraulic_v44(model, enabled=True)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-2)
    output = _forward_v44(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    optimizer.zero_grad(set_to_none=True)
    output.direct_delta_tfv_m3.sum().backward()
    optimizer.step()
    after = model.forward_group(initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D2")
    torch.testing.assert_close(before.direct_delta_tfv_m3, after.direct_delta_tfv_m3, atol=1e-7, rtol=1e-6)


def test_edge_h72_forward_backward_finite():
    _, model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _model_fixture(horizon=72)
    candidate[:, 0, :, :2] = (candidate[:, 0, :, :2] + 0.2).clamp(0.0, 1.0)
    candidate.requires_grad_(True)
    output = _forward_v44(model, prepared, initial, rainfall, reference, candidate, previous, elapsed)
    objective = output.direct_delta_tfv_m3.sum() + output.trajectory_delta_tfv_m3.sum()
    gradient = torch.autograd.grad(objective, candidate)[0]
    assert torch.isfinite(output.direct_delta_tfv_m3).all()
    assert torch.isfinite(gradient).all()
