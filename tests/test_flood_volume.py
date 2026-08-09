from __future__ import annotations

import torch

from rtc.flood_volume import trapezoid_node_flood_volume


def test_trapezoid_volume_uses_current_and_future_flooding_rate() -> None:
    initial = torch.zeros(1, 1, 4)
    initial[..., 2] = 2.0
    future = torch.zeros(1, 2, 1, 4)
    future[:, 0, :, 2] = 4.0
    future[:, 1, :, 2] = 0.0
    volume = trapezoid_node_flood_volume(
        initial, future, flood_rate_index=2, dt_seconds=torch.tensor([10.0, 10.0])
    )
    # 0.5*(2+4)*10 + 0.5*(4+0)*10 = 50 m3.
    assert torch.allclose(volume, torch.tensor([[50.0]]))
