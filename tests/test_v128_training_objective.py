from __future__ import annotations

import numpy as np
import torch

from rtc.step2_train_v128_exact import (
    _directed_pair_gradient_sum,
    _exact_reported_pair_loss,
    _informative_pair_totals,
    canonical_truth_tfv_delta_v128,
)


def test_v128_pair_totals_count_reference_and_all_candidate_pairs() -> None:
    ref, candidate, total = _informative_pair_totals(
        np.asarray([0.2, 5.0, 9.0]), threshold=1.0
    )
    assert ref == 2
    assert candidate == 3
    assert total == 5


def test_v128_pair_totals_exclude_near_ties_without_losing_partition_identity() -> None:
    ref, candidate, total = _informative_pair_totals(
        np.asarray([0.0, 0.5, 3.0, 3.4]), threshold=1.0
    )
    assert ref == 2
    assert candidate == 4
    assert total == 6


def test_v128_canonical_float32_truth_prevents_threshold_partition_drift() -> None:
    """Reproduce the class of 544/542 bug without depending on one study artifact.

    A deterministic search constructs float32 node-volume labels for which re-summing the
    exact same stored labels in NumPy float64 moves one reference/candidate delta across the
    1 m3 informative threshold.  The current canonical pathway must instead match the live
    torch.float32 reduction exactly.
    """
    rng = np.random.default_rng(20260815)
    mismatch = None
    for _ in range(50_000):
        ref = (rng.random(10) * 1000.0).astype(np.float32)
        cand_a = ref + rng.normal(0.0, 0.1, 10).astype(np.float32)
        cand_b = ref + rng.normal(0.0, 0.1, 10).astype(np.float32)
        truth = np.stack((ref, cand_a, cand_b), axis=0).astype(np.float32)
        tfv32, delta32 = canonical_truth_tfv_delta_v128(truth)
        tfv64 = truth.astype(np.float64).sum(axis=1)
        delta64 = tfv64[1:] - tfv64[0]
        class32 = (
            np.abs(delta32[0]) > np.float32(1.0),
            np.abs(delta32[1]) > np.float32(1.0),
            np.abs(delta32[0] - delta32[1]) > np.float32(1.0),
        )
        class64 = (
            abs(delta64[0]) > 1.0,
            abs(delta64[1]) > 1.0,
            abs(delta64[0] - delta64[1]) > 1.0,
        )
        if class32 != class64:
            mismatch = (truth, tfv32, delta32, class32, class64)
            break
    assert mismatch is not None, "deterministic fixture failed to expose precision partition drift"

    truth, tfv32, delta32, class32, class64 = mismatch
    live = torch.as_tensor(truth, dtype=torch.float32).sum(dim=1)
    live_delta = live[1:] - live[0]
    np.testing.assert_array_equal(tfv32, live.numpy())
    np.testing.assert_array_equal(delta32, live_delta.numpy())
    assert class32 != class64

    expected = _informative_pair_totals(delta32, threshold=1.0)
    observed = _informative_pair_totals(live_delta.numpy(), threshold=1.0)
    assert observed == expected


def test_v128_two_pass_directed_gradient_matches_full_unordered_pair_loss() -> None:
    truth = torch.tensor([5.0, 9.0, -2.0], dtype=torch.float64)
    pred = torch.tensor([1.2, 0.4, -0.7], dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(2.5, dtype=torch.float64)
    threshold = 1.0

    full_loss, ref_count, candidate_count = _exact_reported_pair_loss(
        truth_delta=truth,
        predicted_delta=pred,
        threshold=threshold,
        delta_scale=scale,
    )
    full_grad = torch.autograd.grad(full_loss, pred, retain_graph=True)[0]
    denominator = ref_count + candidate_count
    assert denominator > 0

    directed = pred.new_zeros(())
    observed_ref = 0
    observed_candidate_directed = 0
    for index in range(len(pred)):
        term, ref_seen, candidate_seen = _directed_pair_gradient_sum(
            live_truth=truth[index : index + 1],
            live_pred=pred[index : index + 1],
            live_positions=np.asarray([index + 1], dtype=np.int64),
            all_truth=truth,
            all_pred_detached=pred.detach(),
            threshold=threshold,
            delta_scale=scale,
        )
        directed = directed + term / float(denominator)
        observed_ref += ref_seen
        observed_candidate_directed += candidate_seen

    directed_grad = torch.autograd.grad(directed, pred)[0]
    assert observed_ref == ref_count
    assert observed_candidate_directed == 2 * candidate_count
    torch.testing.assert_close(directed_grad, full_grad, rtol=1e-12, atol=1e-12)
