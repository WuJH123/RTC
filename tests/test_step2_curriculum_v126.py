from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.step2_curriculum_v126 import (
    anchor_advantage_loss_v126,
    derive_anchor_tfv_scale_v126,
)


class _Cache:
    def entry(self, name: str):
        flood = np.asarray(
            [
                [100.0, 20.0],
                [80.0, 10.0],
                [140.0, 30.0],
            ],
            dtype=np.float32,
        )
        return SimpleNamespace(
            indices=(0, 1, 2),
            reference_index=0,
            arrays={"exact_node_flood_volume_m3": flood},
        )


def test_d4_local_scale_uses_candidate_minus_reference_truth() -> None:
    scale = derive_anchor_tfv_scale_v126(_Cache(), ["D4::one"])
    # Absolute advantages are 30 and 50 m3; the hard physical floor is 100 m3.
    assert scale == 100.0


def test_advantage_loss_prefers_correct_benefit_order_and_sign() -> None:
    truth = torch.tensor([[-200.0, -50.0, 100.0]])
    good = torch.tensor([[-180.0, -40.0, 90.0]], requires_grad=True)
    bad = torch.tensor([[100.0, -20.0, -150.0]], requires_grad=True)
    good_loss, good_metrics = anchor_advantage_loss_v126(good, truth, scale_m3=100.0)
    bad_loss, bad_metrics = anchor_advantage_loss_v126(bad, truth, scale_m3=100.0)
    assert float(good_loss) < float(bad_loss)
    assert good_metrics["benefit_accuracy"] > bad_metrics["benefit_accuracy"]
    good_loss.backward()
    assert good.grad is not None and torch.isfinite(good.grad).all()
