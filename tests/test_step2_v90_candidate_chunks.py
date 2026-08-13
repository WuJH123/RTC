"""Regression tests for memory-safe candidate execution without data changes."""
from __future__ import annotations

import pytest
import torch

from rtc.step2_optimization_v90 import candidate_batch_chunks_v90
from rtc.step2_train_response_v60 import V60GroupBatch


def _batch(candidates: int = 5) -> V60GroupBatch:
    return V60GroupBatch(
        source_kind="D2",
        group_name="D2::rain::event::checkpoint",
        initial_state=torch.zeros(1, 2, 6),
        rainfall=torch.zeros(1, 3, 1),
        reference_settings=torch.zeros(1, 3, 2),
        candidate_settings=torch.arange(candidates, dtype=torch.float32).reshape(1, candidates, 1, 1).expand(-1, -1, 3, 2).clone(),
        previous_actuator_flow=torch.zeros(1, 2),
        elapsed_seconds=torch.zeros(1),
        true_reference_states=torch.zeros(1, 3, 2, 6),
        true_candidate_states=torch.arange(candidates, dtype=torch.float32).reshape(1, candidates, 1, 1, 1).expand(-1, -1, 3, 2, 6).clone(),
        true_reference_flows=torch.zeros(1, 3, 2),
        true_candidate_flows=torch.arange(candidates, dtype=torch.float32).reshape(1, candidates, 1, 1).expand(-1, -1, 3, 2).clone(),
        true_delta_tfv_m3=torch.arange(candidates, dtype=torch.float32)[None],
    )


def test_candidate_chunks_partition_and_preserve_every_candidate_tensor():
    batch = _batch()
    chunks = list(candidate_batch_chunks_v90(batch, candidate_chunk_size=2))

    assert [chunk.candidate_settings.shape[1] for chunk in chunks] == [2, 2, 1]
    assert torch.equal(
        torch.cat([chunk.candidate_settings for chunk in chunks], dim=1),
        batch.candidate_settings,
    )
    assert torch.equal(
        torch.cat([chunk.true_candidate_states for chunk in chunks], dim=1),
        batch.true_candidate_states,
    )
    assert torch.equal(
        torch.cat([chunk.true_candidate_flows for chunk in chunks], dim=1),
        batch.true_candidate_flows,
    )
    assert torch.equal(
        torch.cat([chunk.true_delta_tfv_m3 for chunk in chunks], dim=1),
        batch.true_delta_tfv_m3,
    )
    assert chunks[0].reference_settings is batch.reference_settings


def test_candidate_chunks_reject_nonpositive_size():
    with pytest.raises(ValueError, match="positive"):
        list(candidate_batch_chunks_v90(_batch(), candidate_chunk_size=0))


def test_candidate_chunk_weights_reconstruct_full_candidate_mean_loss():
    batch = _batch()
    chunks = list(candidate_batch_chunks_v90(batch, candidate_chunk_size=2))
    full_loss = batch.candidate_settings.square().mean()
    weighted = sum(
        chunk.candidate_settings.square().mean()
        * (chunk.candidate_settings.shape[1] / batch.candidate_settings.shape[1])
        for chunk in chunks
    )
    assert torch.allclose(weighted, full_loss)
