from __future__ import annotations

import numpy as np
import torch

from rtc.step2_train_v128_exact import (
    _directed_pair_gradient_sum,
    _exact_reported_pair_loss,
    _informative_pair_totals,
)


def test_v128_pair_totals_count_reference_and_all_candidate_pairs() -> None:
    # Candidate deltas relative to the reference. With threshold=1:
    # reference pairs: 5 and 9 are informative -> 2
    # candidate pairs: |0.2-5|, |0.2-9|, |5-9| -> 3
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
    # Informative candidate comparisons are (0,3), (0,3.4), (0.5,3), (0.5,3.4).
    assert candidate == 4
    assert total == 6


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
