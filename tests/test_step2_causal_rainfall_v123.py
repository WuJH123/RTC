from __future__ import annotations

import numpy as np
import pytest

from rtc.step2_causal_rainfall_v123 import (
    V123_CAUSAL_RAINFALL_CONTRACT,
    CausalForecastStoreV123,
    causal_forecast_from_history_v123,
)


def test_causal_forecast_uses_last_observed_frame_only() -> None:
    history = np.zeros((13, 2, 1), dtype=np.float32)
    history[-1, :, 0] = [4.0, 8.0]
    history[0, :, 0] = [100.0, 200.0]
    forecast = causal_forecast_from_history_v123(
        history, horizon_steps=3, decay_per_step=0.92
    )
    np.testing.assert_allclose(forecast[:, 0, 0], [4.0, 3.68, 3.3856], rtol=0, atol=1e-6)
    np.testing.assert_allclose(forecast[:, 1, 0], [8.0, 7.36, 6.7712], rtol=0, atol=1e-6)


def test_forecast_store_requires_group_event_checkpoint_and_hash_lineage() -> None:
    store = CausalForecastStoreV123(
        group_names=("D2::g",),
        event_ids=("e",),
        checkpoint_ids=("c",),
        checkpoint_elapsed_seconds=np.asarray([3600], dtype=np.int64),
        forecast_mmhr=np.zeros((1, 2, 1, 1), dtype=np.float32),
        history_sha256=("a" * 64,),
        forecast_sha256=("b" * 64,),
        forecast_contract=V123_CAUSAL_RAINFALL_CONTRACT,
        future_realized_rainfall_not_used=True,
    )
    store.validate()
    assert store.event_ids == ("e",)
    assert bool(store.future_realized_rainfall_not_used)


def test_forecast_store_rejects_missing_lineage() -> None:
    with pytest.raises(ValueError, match="event/checkpoint"):
        CausalForecastStoreV123(
            group_names=("D2::g",),
            event_ids=(),
            checkpoint_ids=(),
            checkpoint_elapsed_seconds=np.asarray([], dtype=np.int64),
            forecast_mmhr=np.zeros((1, 2, 1, 1), dtype=np.float32),
            history_sha256=("a" * 64,),
            forecast_sha256=("b" * 64,),
            forecast_contract=V123_CAUSAL_RAINFALL_CONTRACT,
            future_realized_rainfall_not_used=True,
        ).validate()
