from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReplayCheckpointReference:
    elapsed_seconds: int
    node_ids: tuple[str, ...]
    actuator_ids: tuple[str, ...]
    state_si: np.ndarray
    current_setting: np.ndarray
    swmm_engine_version: str
    source_metadata_path: str


def load_checkpoint_reference(
    metadata_path: str | Path, *, elapsed_seconds: int
) -> ReplayCheckpointReference:
    meta_path = Path(metadata_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("checkpoint reference metadata must be a JSON object")
    compact_name = meta.get("compact_file")
    if not compact_name:
        raise ValueError("checkpoint reference requires compact trajectory evidence")
    engine_version = str(meta.get("swmm_engine_version", "")).strip()
    if not engine_version:
        raise ValueError("checkpoint reference lacks SWMM engine version")
    compact = meta_path.parent / str(compact_name)
    with np.load(compact, allow_pickle=False) as raw:
        times = raw["elapsed_seconds"].astype(np.int64)
        matches = np.flatnonzero(times == int(elapsed_seconds))
        if matches.size != 1:
            raise ValueError(
                f"checkpoint reference requires exactly one sample at {elapsed_seconds}s: {compact}"
            )
        i = int(matches[0])
        return ReplayCheckpointReference(
            elapsed_seconds=int(elapsed_seconds),
            node_ids=tuple(raw["node_ids"].astype(str).tolist()),
            actuator_ids=tuple(raw["actuator_ids"].astype(str).tolist()),
            state_si=raw["state_si"][i].astype(np.float64),
            current_setting=raw["current_setting"][i].astype(np.float64),
            swmm_engine_version=engine_version,
            source_metadata_path=str(meta_path.resolve()),
        )


def verify_replayed_checkpoint(
    reference: ReplayCheckpointReference,
    *,
    elapsed_seconds: int,
    node_ids: tuple[str, ...],
    actuator_ids: tuple[str, ...],
    state_si: np.ndarray,
    current_setting: np.ndarray,
    swmm_engine_version: str,
    state_atol: float = 1e-5,
    setting_atol: float = 1e-7,
) -> dict[str, float | int | str | bool]:
    """Verify that a counterfactual branch truly reaches its saved No-control prefix state.

    A same-prefix action experiment is meaningful only when the branch reaches the same
    hydraulic state *before* the candidate is written. We compare the complete compact
    six-channel state and complete actuator readback vector, rather than relying on event
    names or a few network summary statistics.
    """

    if state_atol <= 0 or setting_atol <= 0:
        raise ValueError("replay tolerances must be positive")
    if int(elapsed_seconds) != reference.elapsed_seconds:
        raise ValueError("replayed checkpoint time differs from reference")
    if tuple(node_ids) != reference.node_ids:
        raise ValueError("replayed checkpoint node ordering differs from reference")
    if tuple(actuator_ids) != reference.actuator_ids:
        raise ValueError("replayed checkpoint actuator ordering differs from reference")
    if str(swmm_engine_version) != reference.swmm_engine_version:
        raise ValueError(
            "replayed checkpoint SWMM engine differs from reference: "
            f"{swmm_engine_version} != {reference.swmm_engine_version}"
        )
    state = np.asarray(state_si, dtype=np.float64)
    settings = np.asarray(current_setting, dtype=np.float64).reshape(-1)
    if state.shape != reference.state_si.shape:
        raise ValueError("replayed checkpoint state shape differs from reference")
    if settings.shape != reference.current_setting.shape:
        raise ValueError("replayed checkpoint setting shape differs from reference")
    if not np.isfinite(state).all() or not np.isfinite(settings).all():
        raise ValueError("replayed checkpoint contains non-finite values")
    state_error = float(np.max(np.abs(state - reference.state_si), initial=0.0))
    setting_error = float(
        np.max(np.abs(settings - reference.current_setting), initial=0.0)
    )
    if state_error > float(state_atol):
        raise ValueError(
            f"No-control replay state diverged before action: max_abs_error={state_error:.9g}"
        )
    if setting_error > float(setting_atol):
        raise ValueError(
            "No-control replay actuator readback diverged before action: "
            f"max_abs_error={setting_error:.9g}"
        )
    return {
        "contract": "EXACT_NO_CONTROL_PREFIX_REPLAY_V1",
        "passed": True,
        "checkpoint_elapsed_seconds": int(elapsed_seconds),
        "state_max_abs_error": state_error,
        "setting_max_abs_error": setting_error,
        "state_atol": float(state_atol),
        "setting_atol": float(setting_atol),
        "swmm_engine_version": str(swmm_engine_version),
        "reference_metadata_path": reference.source_metadata_path,
    }
