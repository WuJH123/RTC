from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.direct_tfv_policy_return_hybrid_portfolio import (
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
    DIRECT_TFV_PROJECTED_GRADIENT_GENERATOR_CONTRACT,
    PROJECTED_GRADIENT_SOURCE,
    build_hybrid_policy_return_portfolio,
    build_projected_gradient_h10_proposal,
)


class _DifferentiableH10Value(torch.nn.Module):
    """Tiny differentiable value surface with a nonzero first-action gradient."""

    def __init__(self) -> None:
        super().__init__()
        weights = torch.zeros(109, dtype=torch.float32)
        weights[:6] = torch.tensor([4.0, -3.0, 2.0, -1.0, 5.0, -4.0])
        self.register_buffer("weights", weights)

    def forward(self, **kwargs):
        reference = kwargs["reference_settings"]
        candidate = kwargs["candidate_settings"]
        delta = (candidate[:, :2, :] - reference[:, :2, :]).mean(dim=1)
        score = 100.0 * torch.sum(delta * self.weights[None], dim=1)
        return SimpleNamespace(total_delta_tfv_m3=score)


def _graph() -> SimpleNamespace:
    names = (
        "min_setting",
        "max_setting",
        "is_pump",
        "is_orifice",
        "is_weir",
        "is_outlet",
    )
    physics = np.zeros((109, len(names)), dtype=np.float32)
    physics[:, 1] = 1.0
    for index in range(109):
        physics[index, 2 + (index % 4)] = 1.0
    return SimpleNamespace(
        actuator_ids=tuple(f"A{index:03d}" for index in range(109)),
        actuator_physics_feature_names=names,
        actuator_physics=physics,
        actuator_upstream=np.zeros(109, dtype=np.int64),
        actuator_downstream=np.ones(109, dtype=np.int64),
        node_ids=("UP", "DOWN"),
        static_node_feature_names=("max_depth_m", "storage_capacity_m3"),
        static_node_features=np.asarray(
            [[2.0, 1000.0], [2.0, 1000.0]], dtype=np.float32
        ),
    )


def _normalization() -> SimpleNamespace:
    return SimpleNamespace(
        state_mean=np.zeros(4, dtype=np.float32),
        state_std=np.ones(4, dtype=np.float32),
        rainfall_mean=np.zeros(1, dtype=np.float32),
        rainfall_std=np.ones(1, dtype=np.float32),
        flow_mean=np.zeros(109, dtype=np.float32),
        flow_std=np.ones(109, dtype=np.float32),
    )


def _inputs():
    state = torch.tensor(
        [[[1.6, 0.0, 0.0, 800.0], [0.3, 0.0, 0.0, 100.0]]],
        dtype=torch.float32,
    )
    rainfall = torch.ones((3, 72, 2, 1), dtype=torch.float32)
    flow = torch.zeros((1, 109), dtype=torch.float32)
    active = torch.full((109,), 0.5, dtype=torch.float32)
    return state, rainfall, flow, active


def test_projected_gradient_is_109channel_h10_only_and_support_bounded() -> None:
    state, rainfall, flow, active = _inputs()
    proposal = build_projected_gradient_h10_proposal(
        model=_DifferentiableH10Value(),
        normalization=_normalization(),
        graph=_graph(),
        current_state=state,
        rainfall_scenarios=rainfall,
        previous_actuator_flow=flow,
        active_target=active,
        first_radius=np.full(109, 0.15, dtype=np.float32),
        max_changed_facilities=4,
        max_delta_per_update=0.5,
        gradient_steps=1,
        step_fraction=0.25,
    )
    assert proposal.generator_contract == DIRECT_TFV_PROJECTED_GRADIENT_GENERATOR_CONTRACT
    assert proposal.produced_nonhold_candidate is True
    assert proposal.target is not None
    assert proposal.final_gradient_l2 > 0.0
    delta = torch.abs(proposal.target - active)
    assert int(torch.count_nonzero(delta > 1.0e-7).item()) <= 4
    assert float(delta.max()) <= 0.150001
    assert bool(torch.all((proposal.target >= 0.0) & (proposal.target <= 1.0)))


def test_hybrid_portfolio_adds_one_distinct_gradient_candidate_without_lbfgsb() -> None:
    state, rainfall, flow, active = _inputs()
    result = build_hybrid_policy_return_portfolio(
        model=_DifferentiableH10Value(),
        normalization=_normalization(),
        graph=_graph(),
        current_state=state,
        rainfall_scenarios=rainfall,
        previous_actuator_flow=flow,
        active_target=active,
        first_radius=np.full(109, 0.15, dtype=np.float32),
        max_changed_facilities=4,
        max_delta_per_update=0.5,
        probe_chunk_size=32,
        gradient_steps=1,
        gradient_step_fraction=0.25,
    )
    sources = tuple(candidate.source for candidate in result.candidates)
    assert 2 <= len(sources) <= 4
    assert PROJECTED_GRADIENT_SOURCE in sources
    assert any(source.startswith("STEP2_H10_PROBE_SCALE_") for source in sources)
    assert "TYPE_AWARE_HYDRAULIC_PRESSURE" in sources
    assert all("LBFGS" not in source.upper() for source in sources)
    for candidate in result.candidates:
        delta = torch.abs(candidate.target - active)
        assert candidate.changed_facility_count <= 4
        assert float(delta.max()) <= 0.150001
        assert bool(torch.all((candidate.target >= 0.0) & (candidate.target <= 1.0)))
    assert "82CONTROL_109REP" in DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT


def test_masked_gradient_and_all_candidates_leave_passive_channels_at_hold() -> None:
    state, rainfall, flow, active = _inputs()
    mask = np.zeros(109, dtype=bool)
    mask[:4] = True
    # The differentiable surface also has strong gradients on channels 4/5, but those channels are
    # deliberately passive and therefore may not move under the masked Practical controller.
    result = build_hybrid_policy_return_portfolio(
        model=_DifferentiableH10Value(),
        normalization=_normalization(),
        graph=_graph(),
        current_state=state,
        rainfall_scenarios=rainfall,
        previous_actuator_flow=flow,
        active_target=active,
        first_radius=np.full(109, 0.15, dtype=np.float32),
        max_changed_facilities=4,
        max_delta_per_update=0.5,
        probe_chunk_size=32,
        gradient_steps=2,
        gradient_step_fraction=0.25,
        supervisory_mask=mask,
    )
    assert result.projected_gradient.produced_nonhold_candidate is True
    assert result.learned_probe.probe_count <= 2 * int(mask.sum())
    for candidate in result.candidates:
        target = candidate.target.detach().cpu().numpy()
        assert np.allclose(target[~mask], active.detach().cpu().numpy()[~mask])
        changed = np.flatnonzero(np.abs(target - active.detach().cpu().numpy()) > 1.0e-7)
        assert set(changed.tolist()).issubset(set(np.flatnonzero(mask).tolist()))
