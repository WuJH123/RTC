from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.step2_gradient_audit_v128_dev import _full_settings_gradient


class _LinearObjectiveModel:
    def objective_rollout(self, *, settings, **kwargs):
        del kwargs
        weights = torch.arange(settings.numel(), device=settings.device, dtype=settings.dtype).reshape_as(settings) + 1.0
        value = (settings * weights).sum(dim=(1, 2))
        return SimpleNamespace(optimization_tfv_m3=value)


def test_dev_full_settings_gradient_matches_linear_objective_exactly() -> None:
    graph = SimpleNamespace(actuator_ids=("a0", "a1"))
    base = np.zeros((72, 2), dtype=np.float32)
    gradient = _full_settings_gradient(
        _LinearObjectiveModel(),
        graph=graph,
        static={
            "up": torch.tensor([0, 0]),
            "down": torch.tensor([1, 1]),
            "physics": torch.zeros(2, 1),
            "static": torch.zeros(2, 1),
            "edges": torch.tensor([[0, 1], [1, 0]]),
        },
        initial=torch.zeros(1, 2, 1),
        rainfall=torch.zeros(1, 72, 2, 1),
        flow=torch.zeros(1, 2),
        base_sequence=base,
        flood_rate_index=0,
    )
    expected = np.arange(72 * 2, dtype=np.float64).reshape(72, 2) + 1.0
    np.testing.assert_array_equal(gradient, expected)
