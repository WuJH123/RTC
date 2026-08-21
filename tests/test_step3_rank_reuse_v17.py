from __future__ import annotations

import torch

from rtc.direct_tfv_policy_return_query_margin_v17 import (
    LEGACY_V15_CHECKPOINT_CONTRACT,
    QueryConditionedPolicyReturnAdapterV17,
    import_v15_rank_state,
    rank_state_sha256,
)


def _legacy_rank_state(adapter: QueryConditionedPolicyReturnAdapterV17) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for key, value in adapter.state_dict().items():
        if key.startswith("rank_context_encoder."):
            state["context_encoder." + key[len("rank_context_encoder.") :]] = value.detach().clone()
        elif key.startswith("rank_candidate_encoder."):
            state["candidate_encoder." + key[len("rank_candidate_encoder.") :]] = value.detach().clone()
        elif key.startswith("rank_adjustment."):
            state[key] = value.detach().clone()
    return state


def test_v17_imports_v15_rank_exactly_and_freezes_it() -> None:
    torch.manual_seed(7)
    source = QueryConditionedPolicyReturnAdapterV17(
        target_scale_m3=10_000.0,
        context_dim=5,
        candidate_dim=4,
        hidden_dim=8,
    )
    legacy_state = _legacy_rank_state(source)
    checkpoint = {
        "contract": LEGACY_V15_CHECKPOINT_CONTRACT,
        "context_dim": 5,
        "candidate_dim": 4,
        "query_margin_state_dict": legacy_state,
    }

    torch.manual_seed(99)
    target = QueryConditionedPolicyReturnAdapterV17(
        target_scale_m3=10_000.0,
        context_dim=5,
        candidate_dim=4,
        hidden_dim=8,
    )
    import_v15_rank_state(target, checkpoint)

    assert rank_state_sha256(target) == rank_state_sha256(source)
    assert all(not parameter.requires_grad for parameter in target.rank_parameters())


def test_v17_numeric_margin_sign_is_boundary_sign() -> None:
    torch.manual_seed(11)
    adapter = QueryConditionedPolicyReturnAdapterV17(
        target_scale_m3=10_000.0,
        context_dim=3,
        candidate_dim=2,
        hidden_dim=8,
    )
    for parameter in adapter.parameters():
        parameter.data.zero_()
    # Boundary bias alone controls sign. Magnitude stays strictly positive through softplus.
    adapter.boundary_head[-1].bias.data.fill_(2.0)
    positive = adapter(
        raw_rank_scores_m3=torch.tensor([0.0, 1.0]),
        context_features=torch.zeros(3),
        candidate_features=torch.zeros(2, 2),
    )
    assert float(positive.hold_logit) > 0.0
    assert float(positive.query_best_margin_m3) > 0.0

    adapter.boundary_head[-1].bias.data.fill_(-2.0)
    negative = adapter(
        raw_rank_scores_m3=torch.tensor([0.0, 1.0]),
        context_features=torch.zeros(3),
        candidate_features=torch.zeros(2, 2),
    )
    assert float(negative.hold_logit) < 0.0
    assert float(negative.query_best_margin_m3) < 0.0


def test_v17_magnitude_head_cannot_flip_boundary() -> None:
    torch.manual_seed(13)
    adapter = QueryConditionedPolicyReturnAdapterV17(
        target_scale_m3=10_000.0,
        context_dim=3,
        candidate_dim=2,
        hidden_dim=8,
    )
    for parameter in adapter.parameters():
        parameter.data.zero_()
    adapter.boundary_head[-1].bias.data.fill_(1.0)
    adapter.magnitude_head[-1].bias.data.fill_(20.0)
    output = adapter(
        raw_rank_scores_m3=torch.tensor([0.0, 1.0]),
        context_features=torch.zeros(3),
        candidate_features=torch.zeros(2, 2),
    )
    assert float(output.magnitude_coordinate) > 0.0
    assert float(output.query_best_margin_m3) > 0.0
