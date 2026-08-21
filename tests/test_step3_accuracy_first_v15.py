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


def test_v15_features_reuse_step2_latent_instead_of_11_number_summary() -> None:
    context, candidates = build_query_margin_v2_features(**_inputs())
    assert context.ndim == 1
    assert candidates.ndim == 2
    assert context.numel() > 11
    assert candidates.shape[1] > 9
    assert candidates.shape[0] == 2
    assert torch.isfinite(context).all()
    assert torch.isfinite(candidates).all()


def test_v15_step2_must_be_frozen_before_feature_extraction() -> None:
    kwargs = _inputs()
    model = _DummyStep2()
    kwargs["step2_model"] = model
    try:
        build_query_margin_v2_features(**kwargs)
    except ValueError as exc:
        assert "frozen" in str(exc).lower()
    else:
        raise AssertionError("V15 accepted a trainable Step2 representation")
    freeze_step2_for_step3(model)
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert model.training is False


def test_v15_rank_and_selected_hold_margin_are_aligned() -> None:
    context, candidates = build_query_margin_v2_features(**_inputs())
    adapter = QueryConditionedPolicyReturnAdapterV2(
        target_scale_m3=10000.0,
        context_dim=context.numel(),
        candidate_dim=candidates.shape[1],
        hidden_dim=16,
    )
    output = adapter(
        raw_rank_scores_m3=torch.tensor([5000.0, -2000.0]),
        context_features=context,
        candidate_features=candidates,
    )
    selected = int(output.selected_candidate_index)
    assert selected == int(torch.argmin(output.rank_scores_normalized))
    assert torch.isclose(output.relative_rank_normalized[selected], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(
        output.predicted_returns_m3[selected],
        output.query_best_margin_m3,
        atol=1e-4,
    )
    assert output.hold_logit.ndim == 0


def test_v15_selected_candidate_conditioning_is_permutation_invariant() -> None:
    context, candidates = build_query_margin_v2_features(**_inputs())
    adapter = QueryConditionedPolicyReturnAdapterV2(
        target_scale_m3=10000.0,
        context_dim=context.numel(),
        candidate_dim=candidates.shape[1],
        hidden_dim=16,
    )
    raw = torch.tensor([5000.0, -2000.0])
    first = adapter(
        raw_rank_scores_m3=raw,
        context_features=context,
        candidate_features=candidates,
    )
    perm = torch.tensor([1, 0])
    second = adapter(
        raw_rank_scores_m3=raw[perm],
        context_features=context,
        candidate_features=candidates[perm],
    )
    assert torch.isclose(first.query_best_margin_m3, second.query_best_margin_m3, atol=1e-5)
    assert torch.allclose(first.predicted_returns_m3[perm], second.predicted_returns_m3, atol=1e-5)


def test_v15_passive_channels_remain_immutable() -> None:
    kwargs = _inputs()
    candidates = kwargs["candidate_targets"].clone()
    candidates[0, 100] = 1.0
    kwargs["candidate_targets"] = candidates
    try:
        build_query_margin_v2_features(**kwargs)
    except ValueError as exc:
        assert "passive" in str(exc).lower()
    else:
        raise AssertionError("V15 accepted a candidate that changed a passive channel")
