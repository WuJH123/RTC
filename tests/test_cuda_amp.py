from __future__ import annotations

import pytest
import torch

from rtc.models import GraphMessageBlock


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for AMP regression")
def test_graph_message_block_supports_cuda_amp() -> None:
    block = GraphMessageBlock(hidden_dim=8).cuda()
    x = torch.randn(2, 3, 8, device="cuda", dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], device="cuda", dtype=torch.long)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        result = block(x, edge_index)

    assert result.shape == x.shape
    assert result.device.type == "cuda"
