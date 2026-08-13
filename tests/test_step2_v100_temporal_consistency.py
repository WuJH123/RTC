import numpy as np

from rtc.step2_v100_temporal_consistency import (
    HORIZON_BUCKETS_V100,
    horizon_bucket_indices_v100,
    slope_pair_v100,
)


def test_slope_pair_uses_target_timestamps():
    values = np.asarray([[0.0, 2.0, 5.0]])
    slope = slope_pair_v100(values, np.asarray([300.0, 600.0, 900.0]))
    assert np.allclose(slope, [[2.0 / 300.0, 3.0 / 300.0]])


def test_temporal_buckets_are_fixed_and_exhaustive():
    buckets = horizon_bucket_indices_v100(np.arange(72), np.arange(72) * 300.0 + 300.0)
    assert tuple(buckets) == HORIZON_BUCKETS_V100
    assert [len(v) for v in buckets.values()] == [5, 18, 48]
    assert np.array_equal(np.concatenate(tuple(buckets.values())), np.arange(71))

