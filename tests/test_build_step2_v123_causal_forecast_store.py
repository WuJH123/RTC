from __future__ import annotations

import numpy as np
import pytest

from scripts.build_step2_v123_causal_forecast_store import exact_history_from_no_control_v123


def test_exact_history_uses_checkpoint_and_twelve_preceding_frames() -> None:
    compact = {
        "elapsed_seconds": np.arange(0, 4501, 300, dtype=np.int64),
        "rainfall_mmhr": np.arange(16, dtype=np.float32)[:, None, None],
    }
    history = exact_history_from_no_control_v123(
        compact, checkpoint_elapsed_seconds=3600, history_steps=13, model_step_seconds=300
    )
    assert history.shape == (13, 1, 1)
    np.testing.assert_array_equal(history[:, 0, 0], np.arange(0, 13, dtype=np.float32))


def test_exact_history_fails_closed_on_missing_preaction_frame() -> None:
    compact = {
        "elapsed_seconds": np.asarray([0, 300, 900, 1200], dtype=np.int64),
        "rainfall_mmhr": np.zeros((4, 1, 1), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="causal history"):
        exact_history_from_no_control_v123(
            compact, checkpoint_elapsed_seconds=900, history_steps=4, model_step_seconds=300
        )
