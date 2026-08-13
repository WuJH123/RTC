from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from rtc.step2_physical_edge_v90 import (
    DYNAMIC_EDGE_FEATURE_NAMES_V90,
    STATIC_EDGE_FEATURE_NAMES_V90,
    build_conduit_physical_edge_assets_v90,
    causal_reference_dynamic_edge_features_v90,
)


@pytest.fixture()
def physical_inp(tmp_path: Path) -> Path:
    path = tmp_path / "physical_multiedge.inp"
    path.write_text(
        """[OPTIONS]\nFLOW_UNITS CMS\nLINK_OFFSETS DEPTH\n\n[JUNCTIONS]\nN1 10 5\nN2 8 5\nN3 6 5\nN4 4 5\n\n[OUTFALLS]\nN5 2 FREE NO\n\n[CONDUITS]\nC1 N1 N2 100 0.013 0 0 0 0\nC2 N1 N2 200 0.014 0 0 0 0\nC3 N2 N3 50 0.012 0 0 0 0\n\n[PUMPS]\nP1 N3 N4 CURVE1 ON 0.25 0.75\n\n[ORIFICES]\nO1 N4 N5 SIDE 1.5 0.65 YES 2.5\n\n[WEIRS]\nW1 N2 N5 TRANSVERSE 0.8 1.7 NO 0.2 0.4 YES\n\n[XSECTIONS]\nC1 CIRCULAR 1 0 0 0\nC2 RECT_OPEN 2 1 0 0\nC3 TRAPEZOIDAL 2 4 1.5 2.5\nO1 RECT_CLOSED 0.6 2.0 0 0\nW1 RECT_OPEN 0.4 3.0 0 0\n\n[LOSSES]\nC1 0.1 0.2 0.3 YES\nC2 0.0 0.1 0.2 NO\nC3 0.0 0.0 0.0 NO\n\n[CURVES]\nCURVE1 PUMP4 0 0\n""",
        encoding="utf-8",
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assets(path: Path):
    return build_conduit_physical_edge_assets_v90(
        inp_path=path,
        expected_inp_sha256=_sha256(path),
        node_ids=("N1", "N2", "N3", "N4", "N5"),
    )


def test_conduit_assets_preserve_parallel_identity_orientation_and_degrees(physical_inp: Path):
    assets = _assets(physical_inp)

    assert assets.inp_path == str(physical_inp.resolve())
    assert assets.inp_sha256 == _sha256(physical_inp)
    assert assets.physical_link_count == 6
    assert assets.conduit_physical_link_count == 3
    assert assets.edge_index.shape == (2, 6)
    assert assets.edge_to_link_id.count("C1") == 2
    assert assets.edge_to_link_id.count("C2") == 2
    assert assets.edge_to_link_id.count("C3") == 2
    assert assets.orientation_sign.tolist() == [1, -1, 1, -1, 1, -1]
    assert set(assets.edge_to_link_type) == {"conduit"}
    assert assets.regulator_propagation_edge_count == 0
    assert set(assets.excluded_regulator_link_ids) == {"P1", "O1", "W1"}
    assert assets.out_degree.tolist() == [2, 3, 1, 0, 0]
    assert assets.in_degree.tolist() == [2, 3, 1, 0, 0]


def test_conduit_assets_require_the_explicit_frozen_inp_sha(physical_inp: Path):
    with pytest.raises(ValueError, match="SHA256"):
        build_conduit_physical_edge_assets_v90(
            inp_path=physical_inp,
            expected_inp_sha256="0" * 64,
            node_ids=("N1", "N2", "N3", "N4", "N5"),
        )


def test_static_conduit_features_are_exactly_33_and_normalize_deterministically(physical_inp: Path):
    first = _assets(physical_inp)
    second = _assets(physical_inp)

    assert len(STATIC_EDGE_FEATURE_NAMES_V90) == 33
    assert first.static_features_raw.shape == (6, 33)
    assert first.static_features_normalized.shape == (6, 33)
    assert tuple(first.static_feature_names) == STATIC_EDGE_FEATURE_NAMES_V90
    assert torch.isfinite(first.static_features_raw).all()
    assert torch.isfinite(first.static_features_normalized).all()
    assert torch.isfinite(first.static_normalization_location).all()
    assert torch.isfinite(first.static_normalization_scale).all()
    assert torch.all(first.static_normalization_scale > 0.0)
    torch.testing.assert_close(first.static_features_normalized, second.static_features_normalized)


def test_dynamic_features_are_directed_differentiable_and_causal(physical_inp: Path):
    assets = _assets(physical_inp)
    head = torch.tensor([[[10.0, 7.0, 4.0, 1.0, 0.0], [12.0, 9.0, 6.0, 1.0, 0.0]]], requires_grad=True)
    depth = torch.tensor([[[1.0, 0.5, 0.2, 0.0, 0.0], [1.2, 0.7, 0.4, 0.0, 0.0]]], requires_grad=True)

    dynamic = causal_reference_dynamic_edge_features_v90(
        assets,
        reference_head_m=head,
        reference_depth_m=depth,
        head_scale_m=2.0,
        depth_scale_m=1.0,
        gradient_scale=0.01,
    )

    assert dynamic.shape == (1, 2, 6, len(DYNAMIC_EDGE_FEATURE_NAMES_V90))
    assert torch.isfinite(dynamic).all()
    # C1 forward is N1 -> N2 and reverse must use the reversed hydraulic state.
    assert dynamic[0, 0, 0, 4].item() == pytest.approx(1.5)
    assert dynamic[0, 0, 1, 4].item() == pytest.approx(-1.5)
    assert dynamic[0, 0, 0, -1].item() == pytest.approx(1.0)
    assert dynamic[0, 0, 1, -1].item() == pytest.approx(-1.0)
    dynamic[..., :-1].square().sum().backward()
    assert head.grad is not None and torch.count_nonzero(head.grad) > 0
    assert depth.grad is not None and torch.count_nonzero(depth.grad) > 0

    later_changed = head.detach().clone()
    later_changed[:, 1] += 100.0
    causal_changed = causal_reference_dynamic_edge_features_v90(
        assets,
        reference_head_m=later_changed,
        reference_depth_m=depth.detach(),
        head_scale_m=2.0,
        depth_scale_m=1.0,
        gradient_scale=0.01,
    )
    torch.testing.assert_close(dynamic.detach()[:, 0], causal_changed[:, 0])


def test_dynamic_features_fail_closed_on_nonpositive_scales(physical_inp: Path):
    assets = _assets(physical_inp)
    head = torch.ones((1, 5))
    depth = torch.ones((1, 5))
    with pytest.raises(ValueError, match="positive finite"):
        causal_reference_dynamic_edge_features_v90(
            assets,
            reference_head_m=head,
            reference_depth_m=depth,
            head_scale_m=0.0,
            depth_scale_m=1.0,
            gradient_scale=1.0,
        )
