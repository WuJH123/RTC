import numpy as np

from rtc.step2_v100_no_flood_audit import (
    HORIZON_BUCKETS_V100,
    bucket_slices_v100,
    joint_no_flood_fractions_v100,
    no_flood_mask_v100,
    summarize_no_flood_values_v100,
)


def test_no_flood_mask_is_fixed_and_inclusive_at_epsilon():
    flood = np.asarray([0.0, 1e-8, -1e-8, 1.1e-8], dtype=np.float64)
    assert no_flood_mask_v100(flood, epsilon=1e-8).tolist() == [True, True, True, False]


def test_horizon_buckets_are_exhaustive_and_fixed():
    buckets = bucket_slices_v100(72, 300)
    assert tuple(buckets) == HORIZON_BUCKETS_V100
    assert [len(v) for v in buckets.values()] == [6, 18, 48]
    joined = np.concatenate([v for v in buckets.values()])
    assert np.array_equal(joined, np.arange(72))


def test_no_flood_summary_and_joint_fractions_use_sample_mask():
    flood = np.asarray([[0.0, 0.0], [0.2, 0.0]])
    depth = np.asarray([[0.1, 0.1], [0.0, 0.0]])
    volume = np.asarray([[0.2, 0.2], [0.0, 0.0]])
    flow = np.asarray([0.3, 0.0])
    mask = no_flood_mask_v100(flood, epsilon=1e-8)
    summary = summarize_no_flood_values_v100(depth, mask, epsilon=1e-8)
    assert summary["no_flood_count"] == 1
    assert summary["active_fraction"] == 1.0
    fractions = joint_no_flood_fractions_v100(
        depth, volume, flow, flood, epsilon=1e-8
    )
    assert fractions["depth_active_and_flood_inactive"] == 0.5
    assert fractions["volume_active_and_flood_inactive"] == 0.5
    assert fractions["flow_active_and_flood_inactive"] == 0.5
