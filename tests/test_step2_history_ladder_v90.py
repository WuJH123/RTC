"""Regression tests for the bounded V9 causal-history mechanism diagnostic."""
from __future__ import annotations

import numpy as np
import torch


def _runner_module():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_step2_v90_history_ladder.py"
    spec = importlib.util.spec_from_file_location("step2_history_ladder_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_endpoint_history_extraction_keeps_only_causal_endpoint_and_actuator_values() -> None:
    from rtc.step2_history_ladder_v90 import endpoint_history_features_v90

    states = np.zeros((13, 4, 6), dtype=np.float32)
    states[:, 1, :] = 1.0
    states[:, 3, :] = 3.0
    flows = np.arange(13 * 2, dtype=np.float32).reshape(13, 2)

    features = endpoint_history_features_v90(
        states,
        flows,
        actuator_index=1,
        upstream_index=1,
        downstream_index=3,
    )

    assert features.shape == (13, 13)
    assert np.array_equal(features[:, :6], states[:, 1, :])
    assert np.array_equal(features[:, 6:12], states[:, 3, :])
    assert np.array_equal(features[:, 12], flows[:, 1])


def test_history_encoder_is_exact_zero_for_absent_history() -> None:
    from rtc.step2_history_ladder_v90 import CausalEndpointHistoryEncoderV90

    torch.manual_seed(42)
    encoder = CausalEndpointHistoryEncoderV90()
    absent = torch.zeros(3, 13, 13)
    assert torch.equal(encoder(absent), torch.zeros(3, encoder.output_dim))


def test_local_history_effect_is_exact_zero_for_zero_action() -> None:
    from rtc.step2_history_ladder_v90 import LocalHistoryEffectModelV90

    torch.manual_seed(42)
    model = LocalHistoryEffectModelV90(
        base_feature_dim=8,
        action_feature_indices=(2, 3),
        action_zero_values=(1.5, -2.0),
        output_dim=7,
    )
    base = torch.randn(4, 8)
    # These are normalized feature values corresponding to physical delta-u=0.
    base[:, 2] = 1.5
    base[:, 3] = -2.0
    history = torch.randn(4, 13, 13)
    output = model(base, history)
    assert torch.equal(output, torch.zeros_like(output))


def test_reconstruction_summary_respects_wet_high_and_priority_masks() -> None:
    from rtc.step2_history_ladder_v90 import history_reconstruction_metrics_v90

    truth = np.zeros((1, 13, 3, 6), dtype=np.float32)
    predicted = truth.copy()
    truth[..., 0] = np.asarray([0.0, 0.5, 0.9], dtype=np.float32)
    predicted[..., 0] = np.asarray([0.0, 0.6, 0.7], dtype=np.float32)
    truth[..., 2] = 0.2
    predicted[..., 2] = 0.1
    truth[..., 3] = 10.0
    predicted[..., 3] = 8.0
    max_depth = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)

    result = history_reconstruction_metrics_v90(
        predicted,
        truth,
        max_depth_m=max_depth,
        priority_indices=np.asarray([2], dtype=np.int64),
    )

    assert result["depth_rmse_m"] > 0.0
    assert result["wet_depth_rmse_m"] > 0.0
    assert result["high_depth_rmse_m"] > 0.0
    assert result["priority_depth_rmse_m"] > 0.0
    assert result["flooding_rate_rmse_m3s"] > 0.0
    assert result["storage_volume_rmse_m3"] > 0.0


def test_history_source_contracts_remain_explicit() -> None:
    from rtc.step2_history_ladder_v90 import history_source_contract_v90

    assert history_source_contract_v90("none") == {
        "source_type": "none",
        "online_eligible": True,
        "oracle_diagnostic_only": False,
    }
    assert history_source_contract_v90("frozen_step1_reconstruction")["online_eligible"] is True
    oracle = history_source_contract_v90("oracle_past_swmm")
    assert oracle["online_eligible"] is False
    assert oracle["oracle_diagnostic_only"] is True


def test_history_group_gate_refuses_t5700_but_keeps_t7200_without_snapping() -> None:
    module = _runner_module()

    class Entry:
        def __init__(self, checkpoint_id: str):
            self.checkpoint_id = checkpoint_id
            self.event_id = "E"
            self.rainfall_group = "R"

    class Cache:
        def entry(self, name: str):
            return Entry(name)

    rows = [
        {
            "checkpoint_id": "t5700",
            "event_id": "E",
            "rainfall_group": "R",
            "scientific_split": "development",
            "development_fold": "train",
            "checkpoint_elapsed_seconds": "5700",
        },
        {
            "checkpoint_id": "t7200",
            "event_id": "E",
            "rainfall_group": "R",
            "scientific_split": "development",
            "development_fold": "train",
            "checkpoint_elapsed_seconds": "7200",
        },
    ]
    eligible, rejected = module.select_history_eligible_d2_groups_v90(
        Cache(), ["t5700", "t7200"], rows
    )
    assert eligible == ["t7200"]
    assert rejected["t5700"] == "insufficient_pre_action_history_for_13_frozen_step1_windows"
