"""Causal Train-only history assets for the V9 hydraulic-effect diagnostic.

This module deliberately separates two histories:

``oracle_past_swmm``
    Full no-control state history from an existing compact trajectory.  It is
    causal, but diagnostic-only because the full state is authoritative SWMM
    truth.

``frozen_step1_reconstruction``
    A full-state history reconstructed by a caller-supplied frozen Step1
    predictor from *only* sparse depth/head observations plus the same causal
    node context used by :class:`rtc.lazy_step1.CausalStep1TrajectoryDataset`.
    The helper does not load or train Step1; this keeps the asset construction
    testable and lets the guarded runner own checkpoint admission.

No function in this module starts SWMM, reads validation/final data, or permits
future frames.  The online path fails closed unless a checkpoint has enough
pre-action sensor history to construct all thirteen causal Step1 windows.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch

from .context_features import build_node_context


V90_HISTORY_FRAMES = 13
V90_HISTORY_FRAME_SECONDS = 300
V90_HISTORY_CONTRACT = "PROJECT7_STEP2_V90_CAUSAL_HISTORY_ASSET_V1"


@dataclass(frozen=True)
class HistoryAssetV90:
    """A complete causal 13-frame context with explicit source semantics."""

    source_type: Literal["oracle_past_swmm", "frozen_step1_reconstruction"]
    elapsed_seconds: np.ndarray
    states_physical: np.ndarray
    actuator_flows_physical: np.ndarray
    lineage: dict[str, Any]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer_seconds(value: int | float | np.integer, *, name: str) -> int:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric != float(int(numeric)):
        raise ValueError(f"{name} must be an integer number of seconds")
    return int(numeric)


def _timestamp_positions(elapsed_seconds: np.ndarray) -> dict[int, int]:
    elapsed = np.asarray(elapsed_seconds, dtype=np.int64).reshape(-1)
    if elapsed.size == 0 or len(np.unique(elapsed)) != elapsed.size:
        raise ValueError("compact elapsed_seconds must be non-empty and unique")
    if not np.all(np.diff(elapsed) == V90_HISTORY_FRAME_SECONDS):
        raise ValueError("compact elapsed_seconds must be a 300-second causal grid")
    return {int(value): index for index, value in enumerate(elapsed.tolist())}


def _required_output_elapsed(checkpoint_elapsed_seconds: int) -> np.ndarray:
    return np.arange(
        checkpoint_elapsed_seconds - (V90_HISTORY_FRAMES - 1) * V90_HISTORY_FRAME_SECONDS,
        checkpoint_elapsed_seconds + V90_HISTORY_FRAME_SECONDS,
        V90_HISTORY_FRAME_SECONDS,
        dtype=np.int64,
    )


def _load_required_compact(compact_path: str | Path) -> tuple[Path, dict[str, np.ndarray]]:
    path = Path(compact_path)
    if not path.is_file():
        raise ValueError(f"history compact is missing: {path}")
    required = {
        "elapsed_seconds",
        "state_si",
        "rainfall_mmhr",
        "current_setting",
        "actuator_flow_m3s",
    }
    with np.load(path, allow_pickle=False) as raw:
        missing = sorted(required - set(raw.files))
        if missing:
            raise ValueError(f"history compact lacks required arrays: {missing}")
        values = {name: np.asarray(raw[name], dtype=np.float32) for name in required - {"elapsed_seconds"}}
        values["elapsed_seconds"] = np.asarray(raw["elapsed_seconds"], dtype=np.int64)
    state = values["state_si"]
    if state.ndim != 3 or state.shape[-1] != 6:
        raise ValueError("history compact state_si must be [T,N,6]")
    frames, nodes, _ = state.shape
    if values["rainfall_mmhr"].shape[:2] != (frames, nodes):
        raise ValueError("history rainfall shape differs from state time/node schema")
    if values["current_setting"].shape != values["actuator_flow_m3s"].shape:
        raise ValueError("history setting/actuator-flow shapes differ")
    if values["current_setting"].shape[0] != frames:
        raise ValueError("history actuator time dimension differs from state")
    _timestamp_positions(values["elapsed_seconds"])
    if int(values["elapsed_seconds"][0]) != 0:
        raise ValueError("frozen Step1 causal history compact must include the t=0 frame")
    return path, values


def _take_exact_frames(
    elapsed_seconds: np.ndarray,
    *,
    requested_elapsed_seconds: np.ndarray,
    error_prefix: str,
) -> np.ndarray:
    positions = _timestamp_positions(elapsed_seconds)
    missing = [int(value) for value in requested_elapsed_seconds if int(value) not in positions]
    if missing:
        raise ValueError(
            f"{error_prefix} requires an exact 13-frame causal prefix; missing elapsed seconds {missing[:3]}"
        )
    return np.asarray([positions[int(value)] for value in requested_elapsed_seconds], dtype=np.int64)


def load_oracle_history_v90(
    compact_path: str | Path,
    *,
    checkpoint_elapsed_seconds: int | float | np.integer,
    cache_initial_state: np.ndarray,
) -> HistoryAssetV90:
    """Load diagnostic-only authoritative past state without future frames."""

    checkpoint = _integer_seconds(checkpoint_elapsed_seconds, name="checkpoint_elapsed_seconds")
    path, compact = _load_required_compact(compact_path)
    required = _required_output_elapsed(checkpoint)
    positions = _take_exact_frames(
        compact["elapsed_seconds"],
        requested_elapsed_seconds=required,
        error_prefix="oracle history",
    )
    states = np.asarray(compact["state_si"][positions], dtype=np.float32)
    flows = np.asarray(compact["actuator_flow_m3s"][positions], dtype=np.float32)
    current = np.asarray(cache_initial_state, dtype=np.float32)
    if current.shape != states[-1].shape:
        raise ValueError("oracle history cache_initial_state shape differs from compact current frame")
    max_difference = float(np.max(np.abs(states[-1] - current)))
    if max_difference != 0.0:
        raise ValueError(
            "oracle history current frame differs from cache_initial_state; "
            f"max_abs_difference={max_difference}"
        )
    return HistoryAssetV90(
        source_type="oracle_past_swmm",
        elapsed_seconds=required,
        states_physical=states,
        actuator_flows_physical=flows,
        lineage={
            "contract": V90_HISTORY_CONTRACT,
            "source_type": "oracle_past_swmm",
            "online_eligible": False,
            "oracle_diagnostic_only": True,
            "compact_path": str(path.resolve()),
            "compact_sha256": _sha256_file(path),
            "checkpoint_elapsed_seconds": checkpoint,
            "history_frames": V90_HISTORY_FRAMES,
            "frame_seconds": V90_HISTORY_FRAME_SECONDS,
            "pre_action_only": True,
            "future_leakage": "NONE",
            "current_frame_max_abs_difference": max_difference,
        },
    )


def _step1_sparse_inputs(
    compact: dict[str, np.ndarray],
    *,
    sensor_indices: np.ndarray,
    actuator_upstream: np.ndarray,
    actuator_downstream: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build exactly the sparse observations and node context used by Step1."""

    state = compact["state_si"]
    sensors = np.asarray(sensor_indices, dtype=np.int64).reshape(-1)
    if sensors.size == 0 or np.any(sensors < 0) or np.any(sensors >= state.shape[1]):
        raise ValueError("sensor_indices must be non-empty valid node indices")
    observed = np.zeros((state.shape[0], state.shape[1], 2), dtype=np.float32)
    mask = np.zeros_like(observed)
    observed[:, sensors] = state[:, sensors, :2]
    mask[:, sensors] = 1.0
    context = build_node_context(
        rainfall_mmhr=compact["rainfall_mmhr"],
        actuator_setting=compact["current_setting"],
        actuator_flow_m3s=compact["actuator_flow_m3s"],
        actuator_upstream=np.asarray(actuator_upstream, dtype=np.int64),
        actuator_downstream=np.asarray(actuator_downstream, dtype=np.int64),
        node_count=state.shape[1],
    )
    return observed, mask, np.asarray(context, dtype=np.float32)


def _online_window_positions(
    elapsed_seconds: np.ndarray,
    *,
    checkpoint_elapsed_seconds: int,
) -> tuple[np.ndarray, np.ndarray]:
    output_elapsed = _required_output_elapsed(checkpoint_elapsed_seconds)
    earliest_input = int(output_elapsed[0] - (V90_HISTORY_FRAMES - 1) * V90_HISTORY_FRAME_SECONDS)
    if earliest_input < 0:
        raise ValueError(
            "online 13-frame reconstruction requires checkpoint_elapsed_seconds "
            "at least 7200 seconds"
        )
    positions = _timestamp_positions(elapsed_seconds)
    requested_input = np.arange(
        earliest_input,
        checkpoint_elapsed_seconds + V90_HISTORY_FRAME_SECONDS,
        V90_HISTORY_FRAME_SECONDS,
        dtype=np.int64,
    )
    missing = [int(value) for value in requested_input if int(value) not in positions]
    if missing:
        raise ValueError(
            "online reconstruction requires contiguous causal sensor frames; "
            f"missing elapsed seconds {missing[:3]}"
        )
    output_positions = np.asarray([positions[int(value)] for value in output_elapsed], dtype=np.int64)
    windows = np.stack(
        [
            np.asarray(
                [positions[int(value)] for value in np.arange(
                    int(end) - (V90_HISTORY_FRAMES - 1) * V90_HISTORY_FRAME_SECONDS,
                    int(end) + V90_HISTORY_FRAME_SECONDS,
                    V90_HISTORY_FRAME_SECONDS,
                    dtype=np.int64,
                )],
                dtype=np.int64,
            )
            for end in output_elapsed
        ],
        axis=0,
    )
    if windows.shape != (V90_HISTORY_FRAMES, V90_HISTORY_FRAMES):
        raise RuntimeError("online Step1 window construction has the wrong fixed shape")
    return output_positions, windows


def _as_prediction_tensor(value: np.ndarray | torch.Tensor, *, batch: int, nodes: int) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.shape != (batch, nodes, 6):
        raise ValueError(
            "frozen Step1 predictor must return [batch,node,6]; "
            f"got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("frozen Step1 predictor returned non-finite reconstructed history")
    return tensor.detach().cpu()


def build_online_step1_history_v90(
    compact_path: str | Path,
    *,
    checkpoint_elapsed_seconds: int | float | np.integer,
    cache_initial_state: np.ndarray,
    sensor_indices: np.ndarray,
    actuator_upstream: np.ndarray,
    actuator_downstream: np.ndarray,
    step1_checkpoint_path: str | Path,
    sensor_layout_path: str | Path,
    predict_window: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], np.ndarray | torch.Tensor],
) -> HistoryAssetV90:
    """Construct an online-eligible history via vectorized frozen-Step1 inference.

    ``predict_window`` is intentionally injected.  A guarded caller should bind
    it to a frozen ``SparseStateEstimator`` with its static graph and edge
    tensors, e.g. ``lambda obs, mask, ctx: step1(obs, mask, static, edges, ctx)``.
    This helper guarantees the supplied model sees only causal sparse inputs.
    """

    checkpoint = _integer_seconds(checkpoint_elapsed_seconds, name="checkpoint_elapsed_seconds")
    path, compact = _load_required_compact(compact_path)
    step1_path = Path(step1_checkpoint_path)
    layout_path = Path(sensor_layout_path)
    if not step1_path.is_file() or not layout_path.is_file():
        raise ValueError("frozen Step1 checkpoint and sensor layout must both exist")
    output_positions, window_positions = _online_window_positions(
        compact["elapsed_seconds"], checkpoint_elapsed_seconds=checkpoint
    )
    observed, mask, context = _step1_sparse_inputs(
        compact,
        sensor_indices=sensor_indices,
        actuator_upstream=actuator_upstream,
        actuator_downstream=actuator_downstream,
    )
    sensors = np.asarray(sensor_indices, dtype=np.int64).reshape(-1)
    current = np.asarray(cache_initial_state, dtype=np.float32)
    if current.shape != compact["state_si"][output_positions[-1]].shape:
        raise ValueError("online history cache_initial_state shape differs from compact current frame")
    sensor_difference = float(
        np.max(
            np.abs(
                observed[output_positions[-1], sensors]
                - current[sensors, :2]
            )
        )
    )
    if sensor_difference != 0.0:
        raise ValueError(
            "online history current sparse sensor frame differs from cache_initial_state; "
            f"max_abs_difference={sensor_difference}"
        )

    # [history output frame, Step1 input time, node, feature].  The caller's
    # frozen Step1 model is evaluated once vectorized over all 13 output frames.
    observed_batch = torch.from_numpy(observed[window_positions])
    mask_batch = torch.from_numpy(mask[window_positions])
    context_batch = torch.from_numpy(context[window_positions])
    with torch.no_grad():
        reconstructed = predict_window(observed_batch, mask_batch, context_batch)
    states = _as_prediction_tensor(
        reconstructed,
        batch=V90_HISTORY_FRAMES,
        nodes=compact["state_si"].shape[1],
    ).numpy()
    output_elapsed = compact["elapsed_seconds"][output_positions]
    return HistoryAssetV90(
        source_type="frozen_step1_reconstruction",
        elapsed_seconds=np.asarray(output_elapsed, dtype=np.int64),
        states_physical=np.asarray(states, dtype=np.float32),
        actuator_flows_physical=np.asarray(
            compact["actuator_flow_m3s"][output_positions], dtype=np.float32
        ),
        lineage={
            "contract": V90_HISTORY_CONTRACT,
            "source_type": "frozen_step1_reconstruction",
            "online_eligible": True,
            "oracle_diagnostic_only": False,
            "compact_path": str(path.resolve()),
            "compact_sha256": _sha256_file(path),
            "step1_checkpoint_path": str(step1_path.resolve()),
            "step1_checkpoint_sha256": _sha256_file(step1_path),
            "sensor_layout_path": str(layout_path.resolve()),
            "sensor_layout_sha256": _sha256_file(layout_path),
            "checkpoint_elapsed_seconds": checkpoint,
            "history_frames": V90_HISTORY_FRAMES,
            "step1_window_frames": V90_HISTORY_FRAMES,
            "frame_seconds": V90_HISTORY_FRAME_SECONDS,
            "input_elapsed_start_seconds": int(window_positions[0, 0] * V90_HISTORY_FRAME_SECONDS),
            "input_elapsed_end_seconds": int(output_elapsed[-1]),
            "pre_action_only": True,
            "future_leakage": "NONE",
            "current_sensor_observation_max_abs_difference": sensor_difference,
            "full_state_current_equality_claimed": False,
        },
    )


__all__ = [
    "HistoryAssetV90",
    "V90_HISTORY_CONTRACT",
    "V90_HISTORY_FRAMES",
    "V90_HISTORY_FRAME_SECONDS",
    "build_online_step1_history_v90",
    "load_oracle_history_v90",
]
