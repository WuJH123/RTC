from __future__ import annotations

import numpy as np
import pytest
import torch

from rtc.runtime import choose_first_move
from rtc.step2_causal_forecast_v120 import causal_rainfall_from_checkpoint_v120
from rtc.step2_policy_v120 import _project_executable_sequences_v120
from rtc.step2_v120_contract import (
    SOURCE_D2_AUTHORITATIVE_BRANCH_CENSUS,
    TARGETED_D3_AUTHORITATIVE_BRANCH_CENSUS,
    Step2V120Contract,
    v120_runtime_contract_sha256,
)


def test_v120_source_census_is_not_iid_training_population() -> None:
    contract = Step2V120Contract()
    contract.validate()
    assert SOURCE_D2_AUTHORITATIVE_BRANCH_CENSUS == 4800
    assert TARGETED_D3_AUTHORITATIVE_BRANCH_CENSUS == 3600
    assert contract.training_event_count == 18
    assert contract.internal_holdout_event_count == 4


def test_v120_runtime_hash_is_content_addressed_sha256() -> None:
    value = v120_runtime_contract_sha256()
    assert len(value) == 64
    int(value, 16)


def test_causal_rainfall_ignores_all_future_realised_values() -> None:
    first = np.zeros((72, 3, 1), dtype=np.float32)
    second = np.zeros_like(first)
    first[0, :, 0] = [4.0, 2.0, 1.0]
    second[0] = first[0]
    first[1:, :, 0] = 1000.0
    second[1:, :, 0] = 0.0
    a = causal_rainfall_from_checkpoint_v120(first, decay_per_step=0.92)
    b = causal_rainfall_from_checkpoint_v120(second, decay_per_step=0.92)
    assert np.array_equal(a, b)
    assert np.allclose(a[0], first[0])
    assert np.allclose(a[1], first[0] * 0.92)


def test_candidate_projection_occurs_before_scoring_and_preserves_hold() -> None:
    current = torch.tensor([0.4, 0.6])
    previous_target = torch.tensor([0.8, 0.2])
    hold = current.repeat(4, 1)
    aggressive = torch.tensor(
        [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]
    )
    projected, maximum = _project_executable_sequences_v120(
        torch.stack([hold, aggressive]),
        current_settings=current,
        previous_requested_settings=previous_target,
        min_settings=torch.zeros(2),
        max_settings=torch.ones(2),
        max_delta_per_update=0.5,
        control_block_steps=2,
    )
    assert torch.equal(projected[0], hold)
    assert torch.allclose(projected[1, 0], torch.tensor([0.3, 0.7]))
    assert torch.equal(projected[1, 0], projected[1, 1])
    assert float((projected[1, 2] - projected[1, 0]).abs().max()) <= 0.5 + 1.0e-7
    assert maximum > 0.0

    # The generic write-path projection must be a no-op for a sequence already
    # projected before Value scoring. This is the core score==execute invariant.
    decision = choose_first_move(
        optimized_sequence=projected[1].numpy(),
        surrogate_admissible=True,
        fallback_first_move=hold[0].numpy(),
        current_settings=current.numpy(),
        previous_requested_settings=previous_target.numpy(),
        min_settings=0.0,
        max_settings=1.0,
        max_delta_per_update=0.5,
    )
    assert np.allclose(decision.requested, projected[1, 0].numpy())
    assert decision.projected is False


def test_projection_fails_closed_if_hold_breaks_target_continuity() -> None:
    current = torch.tensor([0.0])
    previous_target = torch.tensor([1.0])
    hold = current.repeat(4, 1)[None]
    with pytest.raises(ValueError, match="HOLD reference"):
        _project_executable_sequences_v120(
            hold,
            current_settings=current,
            previous_requested_settings=previous_target,
            min_settings=torch.zeros(1),
            max_settings=torch.ones(1),
            max_delta_per_update=0.5,
            control_block_steps=2,
        )
