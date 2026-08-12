from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rtc.step2_edge_physics_v441 import (
    CONDUIT_FEATURE_NAMES_V441,
    DYNAMIC_FEATURE_NAMES_V441,
    build_conduit_directed_edge_lineage_v441,
    causal_edge_dynamic_features_v441,
    normalize_conduit_static_features_v441,
    parse_frozen_inp_physical_links_v441,
)
from rtc.step2_control_response_v441 import (
    DifferentiableCounterfactualResponseModelV441,
    edge_hydraulic_parameter_names_v441,
    set_trainable_edge_hydraulic_v441,
)


@pytest.fixture()
def physics_inp(tmp_path: Path) -> Path:
    path = tmp_path / "physics_contract.inp"
    path.write_text(
        """[OPTIONS]\nFLOW_UNITS CMS\nLINK_OFFSETS DEPTH\n\n[JUNCTIONS]\nN1 100 5\nN2 95 5\nN3 90 5\nN4 85 5\n\n[OUTFALLS]\nN5 80 FREE NO\n\n[CONDUITS]\nC1 N1 N2 100 0.013 1 2 0 0\nC2 N1 N2 200 0.014 2 3 0 0\nC3 N2 N3 50 0.012 0.5 0.25 0 0\n\n[PUMPS]\nP1 N3 N4 CURVE1 ON 0.25 0.75\n\n[ORIFICES]\nO1 N4 N5 SIDE 1.5 0.65 YES 2.5\n\n[WEIRS]\nW1 N2 N5 TRANSVERSE 0.8 1.7 NO 0.2 0.4 YES\n\n[XSECTIONS]\nC1 TRAPEZOIDAL 2 4 1.5 2.5\nC2 POWER 3 5 1.2 0\nC3 CIRCULAR 1.0 0 0 0\nO1 RECT_CLOSED 0.6 2.0 0 0\nW1 RECT_OPEN 0.4 3.0 0 0\n\n[LOSSES]\nC1 0.1 0.2 0.3 YES\nC2 0.0 0.1 0.2 NO\nC3 0.0 0.0 0.0 NO\n\n[CURVES]\nCURVE1 PUMP4 0 0\n""",
        encoding="utf-8",
    )
    return path


def _links(path: Path):
    return parse_frozen_inp_physical_links_v441(path, ("N1", "N2", "N3", "N4", "N5"))


def test_orifice_fields_and_xsection_are_separate(physics_inp: Path):
    orifice = next(link for link in _links(physics_inp) if link.link_id == "O1")
    assert orifice.orifice is not None
    assert orifice.orifice.discharge_coefficient == pytest.approx(0.65)
    assert orifice.orifice.flap_gate is True
    assert orifice.orifice.open_close_hours == pytest.approx(2.5)
    assert orifice.orifice.xsection_shape == "RECT_CLOSED"
    assert orifice.orifice.xsection_geometry[0] == pytest.approx(0.6)
    assert orifice.orifice.xsection_geometry[1] == pytest.approx(2.0)
    assert orifice.orifice.discharge_coefficient not in orifice.orifice.xsection_geometry


def test_weir_fields_and_xsection_are_separate(physics_inp: Path):
    weir = next(link for link in _links(physics_inp) if link.link_id == "W1")
    assert weir.weir is not None
    assert weir.weir.weir_type == "TRANSVERSE"
    assert weir.weir.crest_offset == pytest.approx(0.8)
    assert weir.weir.discharge_coefficient == pytest.approx(1.7)
    assert weir.weir.flap_gate is False
    assert weir.weir.end_contractions == pytest.approx(0.2)
    assert weir.weir.secondary_coefficient == pytest.approx(0.4)
    assert weir.weir.surcharge is True
    assert weir.weir.xsection_shape == "RECT_OPEN"
    assert weir.weir.xsection_geometry[0] == pytest.approx(0.4)
    assert weir.weir.xsection_geometry[1] == pytest.approx(3.0)


def test_pump_curve_is_not_a_shape(physics_inp: Path):
    pump = next(link for link in _links(physics_inp) if link.link_id == "P1")
    assert pump.pump is not None
    assert pump.pump.pump_curve_id == "CURVE1"
    assert pump.pump.initial_status == "ON"
    assert pump.pump.startup_depth == pytest.approx(0.25)
    assert pump.pump.shutoff_depth == pytest.approx(0.75)
    assert pump.shape == ""


def test_shape_units_and_dimensionless_parameters(physics_inp: Path):
    links = _links(physics_inp)
    trapezoid = next(link for link in links if link.link_id == "C1")
    power = next(link for link in links if link.link_id == "C2")
    assert trapezoid.shape_geometry == pytest.approx((2.0, 4.0, 1.5, 2.5))
    assert power.shape_geometry[0:2] == pytest.approx((3.0, 5.0))
    assert power.shape_geometry[2] == pytest.approx(1.2)
    assert power.shape_geometry[2] != pytest.approx(1.2 * 1000.0)


def test_missing_barrels_uses_swmm_default_one(physics_inp: Path, tmp_path: Path):
    text = physics_inp.read_text(encoding="utf-8").replace(
        "C3 CIRCULAR 1.0 0 0 0\n", "C3 CIRCULAR 1.0 0 0 0\n"
    )
    path = tmp_path / "missing_barrels.inp"
    path.write_text(text, encoding="utf-8")
    c3 = next(link for link in _links(path) if link.link_id == "C3")
    assert c3.barrels == pytest.approx(1.0)
    assert c3.barrels_defaulted is True


def test_link_offsets_depth_semantics_and_conduit_slope(physics_inp: Path):
    c1 = next(link for link in _links(physics_inp) if link.link_id == "C1")
    assert c1.link_offsets_semantics == "DEPTH"
    assert c1.upstream_link_invert_elevation_m == pytest.approx(101.0)
    assert c1.downstream_link_invert_elevation_m == pytest.approx(97.0)
    assert c1.invert_slope == pytest.approx(0.04)


def test_link_offsets_elevation_semantics(physics_inp: Path, tmp_path: Path):
    path = tmp_path / "elevation.inp"
    path.write_text(physics_inp.read_text(encoding="utf-8").replace("LINK_OFFSETS DEPTH", "LINK_OFFSETS ELEVATION"), encoding="utf-8")
    c1 = next(link for link in parse_frozen_inp_physical_links_v441(path, ("N1", "N2", "N3", "N4", "N5")) if link.link_id == "C1")
    assert c1.link_offsets_semantics == "ELEVATION"
    assert c1.upstream_link_invert_elevation_m == pytest.approx(1.0)
    assert c1.downstream_link_invert_elevation_m == pytest.approx(2.0)
    assert c1.inlet_offset_m == pytest.approx(-99.0)
    assert c1.outlet_offset_m == pytest.approx(-93.0)
    assert c1.invert_slope == pytest.approx(-0.01)


def test_regulator_zero_length_is_not_a_propagation_edge(physics_inp: Path):
    links = _links(physics_inp)
    assert all(link.length_m == 0.0 for link in links if link.link_type in {"pump", "orifice", "weir"})
    lineage = build_conduit_directed_edge_lineage_v441(links, {n: i for i, n in enumerate(("N1", "N2", "N3", "N4", "N5"))})
    assert len(lineage.edge_to_link_id) == 6
    assert set(lineage.edge_to_link_type) == {"conduit"}
    assert all(length > 0.0 for length in lineage.edge_lengths_m)


def test_parallel_conduits_preserve_identity(physics_inp: Path):
    links = _links(physics_inp)
    lineage = build_conduit_directed_edge_lineage_v441(links, {n: i for i, n in enumerate(("N1", "N2", "N3", "N4", "N5"))})
    assert lineage.edge_to_link_id.count("C1") == 2
    assert lineage.edge_to_link_id.count("C2") == 2
    assert lineage.edge_to_link_id.count("C3") == 2


def test_dynamic_edge_features_are_dimensionless_and_causal(physics_inp: Path):
    links = _links(physics_inp)
    lineage = build_conduit_directed_edge_lineage_v441(links, {n: i for i, n in enumerate(("N1", "N2", "N3", "N4", "N5"))})
    context = np.zeros((2, 5, 3), dtype=np.float32)
    head = np.arange(5, dtype=np.float32)[None].repeat(2, axis=0)
    depth = np.zeros_like(head)
    dynamic = causal_edge_dynamic_features_v441(
        context,
        lineage,
        current_head=head,
        current_depth=depth,
        head_scale_train=1.0,
        gradient_scale_train=0.01,
    )
    assert dynamic.shape == (2, 6, len(DYNAMIC_FEATURE_NAMES_V441))
    assert np.isfinite(dynamic).all()
    assert np.max(np.abs(dynamic)) < 20.0
    assert "future" not in " ".join(lineage.dynamic_feature_names).lower()


def test_static_features_are_finite_and_normalized(physics_inp: Path):
    links = _links(physics_inp)
    lineage = build_conduit_directed_edge_lineage_v441(links, {n: i for i, n in enumerate(("N1", "N2", "N3", "N4", "N5"))})
    assert lineage.edge_static_features.shape[1] == len(CONDUIT_FEATURE_NAMES_V441)
    normalized, stats = normalize_conduit_static_features_v441(lineage.edge_static_features)
    assert np.isfinite(normalized).all()
    assert np.isfinite(stats.location).all()
    assert np.isfinite(stats.scale).all()


def _model_fixture_v441(*, horizon: int = 8, candidates: int = 2):
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
    model = DifferentiableCounterfactualResponseModelV441(**kwargs)
    node_ids = tuple(f"N{i}" for i in range(prepared.static_node_features.shape[0]))
    from rtc.step2_edge_physics_v441 import PhysicalLinkV441

    links = tuple(
        PhysicalLinkV441(
            link_id=f"L{i}",
            link_type="conduit",
            from_node=node_ids[int(src)],
            to_node=node_ids[int(dst)],
            static_features=np.r_[np.log1p(1.0), np.zeros(len(CONDUIT_FEATURE_NAMES_V441) - 1)].astype(np.float32),
            shape="CIRCULAR",
            shape_geometry=(1.0, 0.0, 0.0, 0.0),
            barrels=1.0,
            barrels_defaulted=False,
            length_m=1.0,
            roughness_n=0.013,
            inlet_offset_m=0.0,
            outlet_offset_m=0.0,
            link_offsets_semantics="ELEVATION",
            upstream_link_invert_elevation_m=0.0,
            downstream_link_invert_elevation_m=0.0,
            invert_slope=0.0,
            losses=(0.0, 0.0, 0.0, 0.0),
            supported_for_propagation=True,
        )
        for i, (src, dst) in enumerate(prepared.edge_index[:, ::2].t().tolist())
    )
    lineage = build_conduit_directed_edge_lineage_v441(links, {name: i for i, name in enumerate(node_ids)})
    normalized, _ = normalize_conduit_static_features_v441(lineage.edge_static_features)
    model.configure_edge_hydraulic_v441(lineage, normalized, head_scale_train=1.0, gradient_scale_train=0.01)
    return model, prepared, initial, rainfall, reference, candidate, previous, elapsed


def test_v441_zero_init_equals_baseline_and_uses_conduit_only():
    from rtc.step2_control_response_v433 import DifferentiableCounterfactualResponseModelV433

    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _model_fixture_v441()
    baseline = DifferentiableCounterfactualResponseModelV433(
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
    baseline.load_state_dict(model.state_dict(), strict=False)
    candidate[:, 0, :, :2] = (candidate[:, 0, :, :2] + 0.2).clamp(0.0, 1.0)
    left = baseline.forward_group(initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D3")
    right = model.forward_group(initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D3")
    np.testing.assert_array_equal(model.edge_hydraulic_link_types_v441, ("conduit",) * len(model.edge_hydraulic_link_types_v441))
    import torch

    torch.testing.assert_close(left.direct_delta_tfv_m3, right.direct_delta_tfv_m3, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(left.delta_states_physical, right.delta_states_physical, atol=1e-7, rtol=1e-6)


def test_v441_edge_gradient_finite_nonzero_and_d2_invariant():
    import torch

    model, prepared, initial, rainfall, reference, candidate, previous, elapsed = _model_fixture_v441()
    candidate[:, 0, :, :2] = (candidate[:, 0, :, :2] + 0.2).clamp(0.0, 1.0)
    before = model.forward_group(initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D2")
    names = set_trainable_edge_hydraulic_v441(model, enabled=True)
    assert names and edge_hydraulic_parameter_names_v441(model)
    output = model.forward_group(initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D3")
    params = [p for p in model.parameters() if p.requires_grad]
    gradients = torch.autograd.grad(output.direct_delta_tfv_m3.sum(), params, allow_unused=True, retain_graph=True)
    assert all(g is None or torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients if g is not None) > 0.0
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    optimizer.zero_grad(set_to_none=True)
    output.direct_delta_tfv_m3.sum().backward()
    optimizer.step()
    after = model.forward_group(initial, rainfall, reference, candidate, previous, prepared, elapsed, source_kind="D2")
    torch.testing.assert_close(before.direct_delta_tfv_m3, after.direct_delta_tfv_m3, atol=1e-7, rtol=1e-6)
