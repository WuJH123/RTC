"""Build the Train-only V123 checkpoint-keyed causal rainfall forecast store.

The frozen SWMM cache stores realised future rainfall for replay.  This script never
passes that future array to the Value model.  It joins each cache checkpoint to the
existing parent no-control compact trajectory, extracts exactly 13 observed 300-s
frames ending at the checkpoint, and applies the runtime PersistenceDecayForecast
central contract (current level, decay 0.92, multiplier 1.0).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from rtc.step2_causal_rainfall_v123 import (
    V123_CAUSAL_RAINFALL_CONTRACT,
    causal_forecast_from_history_v123,
)
from rtc.step2_train_response_v60 import V60TrainCache


def exact_history_from_no_control_v123(
    compact: Mapping[str, np.ndarray],
    *,
    checkpoint_elapsed_seconds: int,
    history_steps: int = 13,
    model_step_seconds: int = 300,
) -> np.ndarray:
    """Return [history,node,1] frames ``t-(H-1)dt ... t`` exactly."""
    if history_steps <= 0 or model_step_seconds <= 0:
        raise ValueError("V123 causal history contract is invalid")
    elapsed = np.asarray(compact["elapsed_seconds"], dtype=np.int64).reshape(-1)
    rainfall = np.asarray(compact["rainfall_mmhr"], dtype=np.float32)
    if rainfall.ndim != 3 or rainfall.shape[0] != elapsed.size or rainfall.shape[-1] != 1:
        raise ValueError("V123 no-control compact rainfall shape is invalid")
    target_times = int(checkpoint_elapsed_seconds) - np.arange(
        history_steps - 1, -1, -1, dtype=np.int64
    ) * int(model_step_seconds)
    lookup = {int(value): index for index, value in enumerate(elapsed.tolist())}
    missing = [int(value) for value in target_times if int(value) not in lookup]
    if missing:
        raise ValueError(
            "V123 causal history missing pre-action frame(s): "
            + ",".join(str(value) for value in missing)
        )
    indices = [lookup[int(value)] for value in target_times]
    history = np.ascontiguousarray(rainfall[indices], dtype=np.float32)
    if not np.isfinite(history).all() or np.any(history < -1e-7):
        raise ValueError("V123 no-control causal history is non-finite/negative")
    return np.clip(history, 0.0, None)


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _find_parent_no_control(root: Path, event_id: str) -> Path:
    candidates = sorted(
        path
        for path in root.rglob(f"{event_id}__no_control.compact.npz")
        if path.parent.name.lower() == "no_control"
    )
    if len(candidates) != 1:
        raise ValueError(
            f"V123 requires exactly one Train parent no-control compact for {event_id}; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def build_store(
    *,
    cache_manifest: str | Path,
    no_control_root: str | Path,
    output_path: str | Path,
    history_steps: int = 13,
    model_step_seconds: int = 300,
    horizon_steps: int = 72,
    decay_per_step: float = 0.92,
) -> dict[str, object]:
    cache = V60TrainCache(cache_manifest)
    names = sorted(cache.names("D2") + cache.targeted_d3_names())
    if not names:
        raise ValueError("V123 causal forecast store has no D2/D3 groups")
    root = Path(no_control_root).resolve()
    if not root.is_dir():
        raise ValueError(f"V123 parent no-control root is missing: {root}")

    forecast_rows: list[np.ndarray] = []
    histories_sha: list[str] = []
    forecasts_sha: list[str] = []
    group_names: list[str] = []
    event_ids: list[str] = []
    checkpoint_ids: list[str] = []
    checkpoint_elapsed: list[int] = []
    parent_paths: dict[str, str] = {}
    cached_first_mismatch = 0
    for name in names:
        entry = cache.entry(name)
        arrays, ref = entry.arrays, entry.reference_index
        event = str(entry.event_id)
        checkpoint = str(entry.checkpoint_id)
        elapsed = int(np.asarray(arrays["elapsed_seconds"])[ref, 0])
        parent = _find_parent_no_control(root, event)
        parent_paths[event] = str(parent)
        with np.load(parent, allow_pickle=False) as compact:
            compact_dict = {key: compact[key] for key in compact.files}
        history = exact_history_from_no_control_v123(
            compact_dict,
            checkpoint_elapsed_seconds=elapsed,
            history_steps=history_steps,
            model_step_seconds=model_step_seconds,
        )
        current = history[-1]
        cached_first = np.asarray(arrays["rainfall"][ref, 0], dtype=np.float32)
        if not np.allclose(current, cached_first, rtol=0.0, atol=1e-5):
            cached_first_mismatch += 1
        forecast = causal_forecast_from_history_v123(
            history, horizon_steps=horizon_steps, decay_per_step=decay_per_step
        )
        group_names.append(name)
        event_ids.append(event)
        checkpoint_ids.append(checkpoint)
        checkpoint_elapsed.append(elapsed)
        histories_sha.append(_sha256_array(history))
        forecasts_sha.append(_sha256_array(forecast))
        forecast_rows.append(forecast)

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        contract=np.asarray(V123_CAUSAL_RAINFALL_CONTRACT),
        group_names=np.asarray(group_names),
        event_ids=np.asarray(event_ids),
        checkpoint_ids=np.asarray(checkpoint_ids),
        checkpoint_elapsed_seconds=np.asarray(checkpoint_elapsed, dtype=np.int64),
        forecast_mmhr=np.stack(forecast_rows).astype(np.float32),
        history_sha256=np.asarray(histories_sha),
        forecast_sha256=np.asarray(forecasts_sha),
        forecast_contract=np.asarray(
            "PersistenceDecayForecast(history_steps_for_level=1,decay_per_step=0.92,scenario_multiplier=1.0)"
        ),
        future_realized_rainfall_not_used=np.asarray(True),
    )
    payload: dict[str, object] = {
        "contract": V123_CAUSAL_RAINFALL_CONTRACT,
        "store_path": str(output),
        "store_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "group_count": len(group_names),
        "d2_group_count": len([name for name in group_names if name.startswith("D2::")]),
        "d3_group_count": len([name for name in group_names if name.startswith("D3::")]),
        "event_count": len(set(event_ids)),
        "history_steps": history_steps,
        "model_step_seconds": model_step_seconds,
        "horizon_steps": horizon_steps,
        "decay_per_step": decay_per_step,
        "scenario_multiplier": 1.0,
        "forecast_contract": "PersistenceDecayForecast(history_steps_for_level=1,decay_per_step=0.92,scenario_multiplier=1.0)",
        "future_realized_rainfall_not_used": True,
        "cached_first_frame_mismatch_count": cached_first_mismatch,
        "parent_no_control_compacts": parent_paths,
    }
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V123 causal rainfall forecast store")
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--no-control-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(build_store(
        cache_manifest=args.cache_manifest,
        no_control_root=args.no_control_root,
        output_path=args.out,
    ), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
