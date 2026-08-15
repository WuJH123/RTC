from __future__ import annotations

import numpy as np

from rtc.step2_train_v128 import _informative_pair_totals


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
