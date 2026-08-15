from __future__ import annotations

import numpy as np
import torch

from rtc.step2_lazy_stream_v128 import LazyBranchArrayV128, select_to_device_v128_lazy


def test_lazy_branch_array_selects_logical_order_and_horizon_only() -> None:
    source = np.arange(5 * 6 * 2, dtype=np.float32).reshape(5, 6, 2)
    lazy = LazyBranchArrayV128(source=source, raw_branch_indices=np.asarray([3, 1, 4], dtype=np.int64))
    assert lazy.shape == (3, 6, 2)
    selected = lazy.select(np.asarray([2, 0]), horizon=4)
    expected = source[np.asarray([4, 3]), :4]
    np.testing.assert_array_equal(selected, expected)


def test_lazy_select_to_device_matches_expected_physical_microbatch() -> None:
    settings = torch.arange(3 * 6 * 2, dtype=torch.float32).reshape(3, 6, 2)
    state_source = np.arange(5 * 6 * 4 * 3, dtype=np.float32).reshape(5, 6, 4, 3)
    flow_source = np.arange(5 * 6 * 2, dtype=np.float32).reshape(5, 6, 2)
    order = np.asarray([3, 1, 4], dtype=np.int64)
    data = {
        "initial": torch.arange(4 * 3, dtype=torch.float32).reshape(1, 4, 3),
        "rainfall": torch.arange(6 * 4, dtype=torch.float32).reshape(1, 6, 4, 1),
        "previous_flow": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "settings": settings,
        "states": LazyBranchArrayV128(state_source, order),
        "flows": LazyBranchArrayV128(flow_source, order),
        "lazy_mmap_branch_stream": True,
    }
    chunk = select_to_device_v128_lazy(
        data,
        np.asarray([2, 0], dtype=np.int64),
        device=torch.device("cpu"),
        horizon=4,
        include_truth=True,
    )
    assert chunk["initial"].shape == (2, 4, 3)
    assert chunk["rainfall"].shape == (2, 4, 4, 1)
    torch.testing.assert_close(chunk["settings"], settings[[2, 0], :4])
    torch.testing.assert_close(
        chunk["states"], torch.as_tensor(state_source[[4, 3], :4])
    )
    torch.testing.assert_close(
        chunk["flows"], torch.as_tensor(flow_source[[4, 3], :4])
    )
