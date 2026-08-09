from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .large_data_cli import run_d3_batch_main


def validate_d3_execution_contract(
    manifest_path: str | Path,
    *,
    control_block_seconds: int,
    stride_seconds: int,
) -> dict[str, int | str]:
    if control_block_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("D3 control block and stride must be positive")
    if control_block_seconds % stride_seconds:
        raise ValueError("D3 control block must be an integer multiple of model stride")
    frame = pd.read_csv(manifest_path)
    if frame.empty:
        raise ValueError("D3 manifest is empty")
    required = {
        "settings_sequence_json",
        "model_horizon_steps",
        "model_step_seconds",
        "control_update_seconds",
        "control_block_steps",
        "control_blocks",
        "d3_time_contract",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "D3 execution requires the controller-time-bound V2 design manifest; "
            f"missing columns: {missing}"
        )
    if set(frame["d3_time_contract"].astype(str)) != {
        "D3_MODEL_STEP_CONTROL_BLOCK_ALIGNMENT_V1"
    }:
        raise ValueError("D3 manifest uses an incompatible time contract")

    def one_int(column: str) -> int:
        values = pd.to_numeric(frame[column], errors="raise").astype(int).unique().tolist()
        if len(values) != 1:
            raise ValueError(f"D3 manifest has multiple {column} values: {values}")
        return int(values[0])

    model_step = one_int("model_step_seconds")
    control_update = one_int("control_update_seconds")
    block_steps = one_int("control_block_steps")
    control_blocks = one_int("control_blocks")
    model_horizon_steps = one_int("model_horizon_steps")
    if model_step != stride_seconds:
        raise ValueError(
            f"D3 runtime stride {stride_seconds}s differs from designed model step {model_step}s"
        )
    if control_update != control_block_seconds:
        raise ValueError(
            "D3 runtime control block differs from the frozen controller design: "
            f"runtime={control_block_seconds}s, designed={control_update}s"
        )
    if block_steps != control_block_seconds // stride_seconds:
        raise ValueError("D3 manifest control_block_steps is inconsistent with runtime cadences")
    if model_horizon_steps != control_blocks * block_steps:
        raise ValueError("D3 manifest model horizon does not equal control_blocks*control_block_steps")

    lengths = frame["settings_sequence_json"].map(lambda raw: len(json.loads(str(raw))))
    if not (lengths == control_blocks).all():
        raise ValueError("D3 sequence JSON length differs from the frozen control-block count")
    return {
        "contract": "D3_EXECUTION_TIME_GUARD_V1",
        "model_step_seconds": model_step,
        "control_update_seconds": control_update,
        "control_block_steps": block_steps,
        "control_blocks": control_blocks,
        "model_horizon_steps": model_horizon_steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--control-block-seconds", type=int, required=True)
    parser.add_argument("--stride-seconds", type=int, default=300)
    known, _ = parser.parse_known_args()
    evidence = validate_d3_execution_contract(
        known.manifest,
        control_block_seconds=known.control_block_seconds,
        stride_seconds=known.stride_seconds,
    )
    print(json.dumps(evidence, indent=2))
    run_d3_batch_main()


if __name__ == "__main__":
    main()
