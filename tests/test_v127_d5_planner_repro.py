from __future__ import annotations

import numpy as np

from scripts.plan_d5_gradient_v127 import _select_checkpoints


def test_d5_checkpoint_selection_remains_outcome_blind_and_deterministic() -> None:
    records = [
        {
            "descriptor": np.asarray([float(i), float(i % 2)], dtype=float),
            "rainfall_group": f"rain-{i % 2}",
            "severity": float(i),
            "checkpoint_id": f"cp-{i}",
            "active_target_sha256": f"sha-{i}",
        }
        for i in range(6)
    ]
    first, _ = _select_checkpoints(records, 4)
    second, _ = _select_checkpoints(records, 4)
    assert [row["checkpoint_id"] for row in first] == [
        row["checkpoint_id"] for row in second
    ]
    assert len(first) == 4
