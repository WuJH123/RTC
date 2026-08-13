from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch


def test_physical_edge_runner_refuses_a_model_that_cannot_receive_assets():
    """The mechanism runner must not silently fall back to the old topology path."""
    from scripts.run_step2_v90_physical_edge_d2 import _construct_physical_edge_model

    class NoPhysicalEdgeKeyword:
        def __init__(self, *, value):
            self.value = value

    with pytest.raises(TypeError, match="physical_edge_assets"):
        _construct_physical_edge_model(
            NoPhysicalEdgeKeyword,
            base_kwargs={"value": 1},
            physical_edge_assets=object(),
        )


def test_physical_edge_runner_passes_the_frozen_assets_to_the_model():
    from scripts.run_step2_v90_physical_edge_d2 import _construct_physical_edge_model

    assets = object()

    class ReceivesPhysicalEdgeAssets:
        def __init__(self, *, value, physical_edge_assets):
            self.value = value
            self.physical_edge_assets = physical_edge_assets

    model = _construct_physical_edge_model(
        ReceivesPhysicalEdgeAssets,
        base_kwargs={"value": 7},
        physical_edge_assets=assets,
    )

    assert model.value == 7
    assert model.physical_edge_assets is assets


def test_physical_edge_runner_emits_conduit_only_lineage_evidence():
    from scripts.run_step2_v90_physical_edge_d2 import _physical_edge_lineage

    assets = SimpleNamespace(
        contract="PROJECT7_STEP2_V90_CONDUIT_PHYSICAL_EDGE_ASSETS_V1",
        inp_path="E:/frozen.inp",
        inp_sha256="a" * 64,
        physical_link_count=1276,
        conduit_physical_link_count=1167,
        directed_edge_count=2334,
        static_normalization_sha256="b" * 64,
        regulator_propagation_edge_count=0,
        uses_future_truth=False,
        uses_online_link_flow=False,
    )

    report = _physical_edge_lineage(assets)

    assert report["conduit_only"] is True
    assert report["regulator_propagation_edge_count"] == 0
    assert report["uses_future_truth"] is False
    assert report["uses_online_link_flow"] is False
    assert report["directed_conduit_edge_count"] == 2334


def test_physical_edge_runner_derives_dynamic_scales_without_effect_targets():
    from scripts.run_step2_v90_physical_edge_d2 import _physical_dynamic_scales

    normalization = SimpleNamespace(state_std=[0.25, 3.0, 7.0, 9.0, 11.0, 13.0])
    assets = SimpleNamespace(edge_length_m=np.asarray([10.0, 20.0, 30.0], dtype=np.float32))

    scales = _physical_dynamic_scales(normalization, assets)

    assert scales == {
        "head_scale_m": 3.0,
        "depth_scale_m": 0.25,
        "gradient_scale": 3.0 / 20.0,
        "source": "TrainFit input-state std plus frozen conduit median length; no effect targets",
    }


def _checkpoint(*, graph="graph", cache="cache", basis="basis", design="design", split="split"):
    return {
        "lineage": {
            "graph_sha256": graph,
            "cache_manifest_sha256": cache,
            "basis_sha256_from_cache_lineage": basis,
            "design_sha256_from_cache_lineage": design,
        },
        "split_manifest_sha256": split,
    }


@pytest.mark.parametrize(
    ("component", "value_patch", "hydraulic_patch"),
    (
        ("graph_sha256", {"graph": "wrong-graph"}, {}),
        ("cache_manifest_sha256", {"cache": "wrong-cache"}, {}),
        ("basis_sha256", {"basis": "wrong-basis"}, {}),
        ("design_sha256", {}, {"design": "wrong-design"}),
        ("split_manifest_sha256", {}, {"split": "wrong-split"}),
    ),
)
def test_physical_edge_runner_fails_closed_on_v7_lineage_mismatch(
    component, value_patch, hydraulic_patch
):
    from scripts.run_step2_v90_physical_edge_d2 import _validate_v7_lineage

    value = _checkpoint(**value_patch)
    hydraulic = _checkpoint(**hydraulic_patch)
    with pytest.raises(ValueError, match=component):
        _validate_v7_lineage(
            value_checkpoint=value,
            hydraulic_checkpoint=hydraulic,
            graph_sha256="graph",
            cache_manifest_sha256="cache",
            cache_lineage={
                "v60_control_basis_sha256": "basis",
                "v60_design_contract_sha256": "design",
            },
        )


@pytest.mark.parametrize(
    ("seed", "holdout_fraction", "message"),
    ((41, 0.20, "seed"), (42, 0.25, "split")),
)
def test_physical_edge_runner_rejects_noncanonical_schedule(seed, holdout_fraction, message):
    from scripts.run_step2_v90_physical_edge_d2 import _validate_canonical_schedule

    with pytest.raises(ValueError, match=message):
        _validate_canonical_schedule(seed=seed, holdout_fraction=holdout_fraction)


def test_physical_edge_runner_checks_future_actions_for_every_retained_state_and_flow():
    from scripts.run_step2_v90_physical_edge_d2 import _assert_full_horizon_causality

    candidate = torch.zeros(1, 1, 5, 1)
    retained = torch.tensor([0, 2, 4])

    def forward(settings):
        # State/flow at retained j only depend on settings through raw timestep j.
        values = settings[:, :, retained, :]
        return SimpleNamespace(
            raw_delta_states_physical=values[..., None],
            raw_delta_flows_physical=values,
            horizon_indices=retained,
        )

    _assert_full_horizon_causality(
        candidate_settings=candidate,
        baseline_output=forward(candidate),
        forward=forward,
    )

    def leaky_flow(settings):
        output = forward(settings)
        return SimpleNamespace(
            raw_delta_states_physical=output.raw_delta_states_physical,
            raw_delta_flows_physical=output.raw_delta_flows_physical
            + settings[:, :, -1:, :],
            horizon_indices=retained,
        )

    with pytest.raises(RuntimeError, match="future candidate action affected.*flow"):
        _assert_full_horizon_causality(
            candidate_settings=candidate,
            baseline_output=leaky_flow(candidate),
            forward=leaky_flow,
        )
