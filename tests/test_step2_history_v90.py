from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from rtc.step2_history_v90 import (
    build_online_step1_history_v90,
    load_oracle_history_v90,
)


def _write_compact(path: Path, *, frames: int = 25) -> tuple[np.ndarray, np.ndarray]:
    elapsed = np.arange(frames, dtype=np.int64) * 300
    state = np.zeros((frames, 3, 6), dtype=np.float32)
    # Values identify both source time and node/channel in assertions below.
    state[:] = np.arange(frames, dtype=np.float32)[:, None, None]
    state += np.arange(3, dtype=np.float32)[None, :, None] * 0.01
    state += np.arange(6, dtype=np.float32)[None, None, :] * 0.001
    rainfall = np.full((frames, 3, 1), 2.0, dtype=np.float32)
    setting = np.full((frames, 2), 0.25, dtype=np.float32)
    flow = np.arange(frames * 2, dtype=np.float32).reshape(frames, 2) / 10.0
    np.savez_compressed(
        path,
        elapsed_seconds=elapsed,
        state_si=state,
        rainfall_mmhr=rainfall,
        current_setting=setting,
        actuator_flow_m3s=flow,
    )
    return state, flow


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_oracle_history_recovers_exact_13_pre_action_frames_and_lineage(tmp_path: Path) -> None:
    compact = tmp_path / "no_control.npz"
    state, flow = _write_compact(compact)
    checkpoint = 3600
    current = state[checkpoint // 300].copy()

    history = load_oracle_history_v90(
        compact,
        checkpoint_elapsed_seconds=checkpoint,
        cache_initial_state=current,
    )

    assert history.source_type == "oracle_past_swmm"
    assert history.states_physical.shape == (13, 3, 6)
    assert history.actuator_flows_physical.shape == (13, 2)
    assert history.elapsed_seconds.tolist() == list(range(0, 3601, 300))
    assert np.array_equal(history.states_physical[-1], current)
    assert np.array_equal(history.actuator_flows_physical[-1], flow[12])
    assert history.lineage["pre_action_only"] is True
    assert history.lineage["future_leakage"] == "NONE"
    assert history.lineage["compact_sha256"] == _sha(compact)


def test_oracle_history_fails_closed_for_missing_prefix_or_cache_mismatch(tmp_path: Path) -> None:
    compact = tmp_path / "short.npz"
    state, _ = _write_compact(compact, frames=12)
    with pytest.raises(ValueError, match="13-frame causal prefix"):
        load_oracle_history_v90(
            compact,
            checkpoint_elapsed_seconds=3300,
            cache_initial_state=state[-1],
        )

    complete = tmp_path / "complete.npz"
    state, _ = _write_compact(complete)
    with pytest.raises(ValueError, match="current frame differs"):
        load_oracle_history_v90(
            complete,
            checkpoint_elapsed_seconds=3600,
            cache_initial_state=state[12] + 1.0,
        )


def test_online_history_requires_t7200_to_construct_thirteen_step1_windows(tmp_path: Path) -> None:
    compact = tmp_path / "no_control.npz"
    state, _ = _write_compact(compact)
    checkpoint_path = tmp_path / "step1.pt"
    checkpoint_path.write_bytes(b"frozen-step1")
    sensor_layout = tmp_path / "sensors.txt"
    sensor_layout.write_text("N0\nN2\n", encoding="utf-8")

    def predictor(observed, mask, context):
        return observed[:, -1, :, :1].repeat(1, 1, 6)

    with pytest.raises(ValueError, match="at least 7200"):
        build_online_step1_history_v90(
            compact,
            checkpoint_elapsed_seconds=6900,
            cache_initial_state=state[23],
            sensor_indices=np.asarray([0, 2]),
            actuator_upstream=np.asarray([0, 1]),
            actuator_downstream=np.asarray([1, 2]),
            step1_checkpoint_path=checkpoint_path,
            sensor_layout_path=sensor_layout,
            predict_window=predictor,
        )


def test_online_history_uses_only_sparse_causal_inputs_and_records_frozen_lineage(tmp_path: Path) -> None:
    compact = tmp_path / "no_control.npz"
    state, flow = _write_compact(compact)
    checkpoint_path = tmp_path / "step1.pt"
    checkpoint_path.write_bytes(b"frozen-step1")
    sensor_layout = tmp_path / "sensors.txt"
    sensor_layout.write_text("N0\nN2\n", encoding="utf-8")
    received = []

    def predictor(observed, mask, context):
        # The pure helper passes batches exactly as the frozen Step1 contract expects.
        received.append((torch.is_grad_enabled(), observed.detach().clone(), mask.detach().clone()))
        return observed[:, -1, :, :1].repeat(1, 1, 6)

    checkpoint = 7200
    history = build_online_step1_history_v90(
        compact,
        checkpoint_elapsed_seconds=checkpoint,
        cache_initial_state=state[24],
        sensor_indices=np.asarray([0, 2]),
        actuator_upstream=np.asarray([0, 1]),
        actuator_downstream=np.asarray([1, 2]),
        step1_checkpoint_path=checkpoint_path,
        sensor_layout_path=sensor_layout,
        predict_window=predictor,
    )

    assert history.source_type == "frozen_step1_reconstruction"
    assert history.states_physical.shape == (13, 3, 6)
    assert history.actuator_flows_physical.shape == (13, 2)
    assert history.elapsed_seconds.tolist() == list(range(3600, 7201, 300))
    assert len(received) == 1  # vectorized 13-output inference, not 13 serial models.
    grad_enabled, observed, mask = received[0]
    assert grad_enabled is False
    assert observed.shape == (13, 13, 3, 2)
    assert torch.equal(mask[:, :, 1], torch.zeros_like(mask[:, :, 1]))
    assert torch.allclose(observed[:, :, 0], torch.as_tensor(state[:13, 0, :2]).expand(13, -1, -1)) is False
    # Each window's final sparse observation is its own causal output timestamp.
    assert torch.allclose(observed[:, -1, 0, 0], torch.as_tensor(state[12:25, 0, 0]))
    assert np.array_equal(history.actuator_flows_physical[-1], flow[-1])
    assert history.lineage["step1_checkpoint_sha256"] == _sha(checkpoint_path)
    assert history.lineage["sensor_layout_sha256"] == _sha(sensor_layout)
    assert history.lineage["current_sensor_observation_max_abs_difference"] == 0.0
    assert history.lineage["future_leakage"] == "NONE"


def test_online_history_fails_closed_when_current_sensor_observation_is_misaligned(tmp_path: Path) -> None:
    compact = tmp_path / "no_control.npz"
    state, _ = _write_compact(compact)
    checkpoint_path = tmp_path / "step1.pt"
    checkpoint_path.write_bytes(b"frozen-step1")
    sensor_layout = tmp_path / "sensors.txt"
    sensor_layout.write_text("N0\n", encoding="utf-8")

    def predictor(observed, mask, context):
        return observed[:, -1, :, :1].repeat(1, 1, 6)

    with pytest.raises(ValueError, match="current sparse sensor frame differs"):
        build_online_step1_history_v90(
            compact,
            checkpoint_elapsed_seconds=7200,
            cache_initial_state=state[24] + 1.0,
            sensor_indices=np.asarray([0]),
            actuator_upstream=np.asarray([0, 1]),
            actuator_downstream=np.asarray([1, 2]),
            step1_checkpoint_path=checkpoint_path,
            sensor_layout_path=sensor_layout,
            predict_window=predictor,
        )


def test_online_history_requires_the_frozen_step1_t0_contract(tmp_path: Path) -> None:
    compact = tmp_path / "shifted.npz"
    state, _ = _write_compact(compact)
    with np.load(compact, allow_pickle=False) as raw:
        payload = {name: raw[name] for name in raw.files}
    payload["elapsed_seconds"] = payload["elapsed_seconds"] + 300
    np.savez_compressed(compact, **payload)
    checkpoint_path = tmp_path / "step1.pt"
    checkpoint_path.write_bytes(b"frozen-step1")
    sensor_layout = tmp_path / "sensors.txt"
    sensor_layout.write_text("N0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="t=0"):
        build_online_step1_history_v90(
            compact,
            checkpoint_elapsed_seconds=7200,
            cache_initial_state=state[23],
            sensor_indices=np.asarray([0]),
            actuator_upstream=np.asarray([0, 1]),
            actuator_downstream=np.asarray([1, 2]),
            step1_checkpoint_path=checkpoint_path,
            sensor_layout_path=sensor_layout,
            predict_window=lambda observed, mask, context: observed[:, -1, :, :1].repeat(1, 1, 6),
        )
