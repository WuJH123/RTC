from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from rtc.direct_tfv_policy_return_query_margin_v2 import (
    QueryConditionedPolicyReturnAdapterV2,
    build_query_margin_v2_features,
    freeze_step2_for_step3,
)


class _DummyStep2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.global_state_encoder = nn.Linear(4, 4, bias=False)
        self.rainfall_encoder = nn.Linear(1, 4, bias=False)

    def _rainfall_summary(self, rainfall: torch.Tensor) -> torch.Tensor:
        return rainfall.mean(dim=(1, 2))

    def _facility_context(self, **kwargs: torch.Tensor) -> torch.Tensor:
        state = kwargs["current_state"]
        return torch.zeros((state.shape[0], 109, 4), dtype=state.dtype, device=state.device)

    def _sequence_latent(self, common: torch.Tensor, settings: torch.Tensor) -> torch.Tensor:
        setting = settings.mean(dim=1)[..., None]
        return common + setting.expand(-1, -1, 4)


def _inputs() -> dict[str, object]:
    model = freeze_step2_for_step3(_DummyStep2())
    normalization = SimpleNamespace(
        state_mean=np.asarray(0.0, dtype=np.float32),
        state_std=np.asarray(1.0, dtype=np.float32),
        rainfall_mean=np.asarray(0.0, dtype=np.float32),
        rainfall_std=np.asarray(1.0, dtype=np.float32),
        flow_mean=np.asarray(0.0, dtype=np.float32),
        flow_std=np.asarray(1.0, dtype=np.float32),
    )
    graph = SimpleNamespace(
        actuator_upstream=np.zeros(109, dtype=np.int64),
        actuator_downstream=np.ones(109, dtype=np.int64),
        actuator_physics=np.ones((109, 1), dtype=np.float32),
    )
    mask = np.zeros(109, dtype=bool)
    mask[:82] = True
    active = torch.zeros(109)
    candidates = torch.zeros((2, 109))
    candidates[0, 0] = 0.5
    candidates[1, 1] = 1.0
    return {
        "step2_model": model,
        "normalization": normalization,
        "graph": graph,
        "current_state": torch.arange(6, dtype=torch.float32).reshape(3, 2) / 10.0,
        "rainfall_scenarios": torch.ones((2, 72, 3, 1), dtype=torch.float32),
        "previous_actuator_flow": torch.linspace(0.0, 1.0, 109),
        "active_target": active,
        "candidate_targets": candidates,
        "base_step2_scores_m3": torch.tensor([-1000.0, 500.0]),
        "candidate_sources": ["STEP2_H10_PROBE_SCALE_0.50", "TYPE_AWARE_HYDRAULIC_PRESSURE"],
        "supervisory_mask": mask,
        "target_scale_m3": 10000.0,
    }


def _adapter() -> tuple[QueryConditionedPolicyReturnAdapterV2, torch.Tensor, torch.Tensor]:
    context, candidates = build_query_margin_v2_features(**_inputs())
    adapter = QueryConditionedPolicyReturnAdapterV2(
        target_scale_m3=10000.0,
        context_dim=context.numel(),
        candidate_dim=candidates.shape[1],
        hidden_dim=16,
    )
    return adapter, context, candidates


def test_v16_features_reuse_frozen_step2_latent_instead_of_11_number_summary() -> None:
    kwargs = _inputs()
    model = kwargs["step2_model"]
    assert not any(parameter.requires_grad for parameter in model.parameters())
    context, candidates = build_query_margin_v2_features(**kwargs)
    assert context.ndim == 1
    assert candidates.ndim == 2
    assert context.numel() > 11
    assert candidates.shape[1] > 9
    assert candidates.shape[0] == 2
    assert torch.isfinite(context).all()
    assert torch.isfinite(candidates).all()


def test_v16_rank_selected_candidate_is_anchored_at_numeric_margin() -> None:
    adapter, context, candidates = _adapter()
    output = adapter(
        raw_rank_scores_m3=torch.tensor([5000.0, -2000.0]),
        context_features=context,
        candidate_features=candidates,
    )
    selected = int(output.selected_candidate_index)
    assert torch.isclose(output.relative_rank_normalized[selected], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(
        output.predicted_returns_m3[selected],
        output.query_best_margin_m3,
        atol=1e-4,
    )


def test_v16_hold_logit_is_exactly_the_deployed_margin_coordinate() -> None:
    adapter, context, candidates = _adapter()
    output = adapter(
        raw_rank_scores_m3=torch.tensor([1000.0, -1000.0]),
        context_features=context,
        candidate_features=candidates,
    )
    assert torch.isclose(output.hold_logit, output.margin_coordinate, atol=0.0, rtol=0.0)
    expected_margin = torch.sinh(output.margin_coordinate) * 10000.0
    assert torch.isclose(output.query_best_margin_m3, expected_margin, atol=1e-4, rtol=1e-5)
    assert bool((output.query_best_margin_m3 >= 0.0) == (output.hold_logit >= 0.0))


def test_v16_rank_and_margin_parameter_paths_are_stage_separated() -> None:
    adapter, _context, _candidates = _adapter()
    adapter.set_rank_stage()
    assert all(parameter.requires_grad for parameter in adapter.rank_parameters())
    assert all(not parameter.requires_grad for parameter in adapter.margin_parameters())
    adapter.set_margin_stage()
    assert all(not parameter.requires_grad for parameter in adapter.rank_parameters())
    assert all(parameter.requires_grad for parameter in adapter.margin_parameters())


def test_v16_passive_channels_remain_immutable() -> None:
    kwargs = _inputs()
    candidates = kwargs["candidate_targets"].clone()
    candidates[0, 100] = 1.0
    kwargs["candidate_targets"] = candidates
    try:
        build_query_margin_v2_features(**kwargs)
    except ValueError as exc:
        assert "passive" in str(exc).lower()
    else:
        raise AssertionError("V16 accepted a candidate that changed a passive channel")
