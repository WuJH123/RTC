from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .causal_timing import CausalTimingContract, timing_from_controller_config
from .data_design import design_multi_actuator_rollouts
from .inp import ActuatorCatalog, discover_actuators


def _load_json(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("controller config must be a JSON object")
    return payload


def design_d3_manifest(
    *,
    checkpoints: pd.DataFrame,
    catalog: ActuatorCatalog,
    timing: CausalTimingContract,
    sequences_per_checkpoint: int = 8,
    perturbation_std: float = 0.20,
    change_probability: float = 0.25,
    seed: int = 42,
) -> pd.DataFrame:
    """Design D3 sequences with control-block and model-step horizons kept explicit."""

    timing.validate()
    control_blocks = timing.d3_control_blocks
    manifest = design_multi_actuator_rollouts(
        checkpoints,
        catalog,
        horizon_steps=control_blocks,
        sequences_per_checkpoint=sequences_per_checkpoint,
        perturbation_std=perturbation_std,
        change_probability=change_probability,
        seed=seed,
    ).copy()

    # ``design_multi_actuator_rollouts`` historically called the number of setting blocks
    # ``horizon_steps``.  Preserve the value only under an unambiguous name here, then stamp
    # the actual Step2 model horizon separately.
    if "horizon_steps" in manifest.columns:
        manifest = manifest.rename(columns={"horizon_steps": "control_blocks"})
    manifest["model_horizon_steps"] = int(timing.horizon_steps)
    manifest["model_step_seconds"] = int(timing.model_step_seconds)
    manifest["control_update_seconds"] = int(timing.control_update_seconds)
    manifest["control_block_steps"] = int(timing.control_block_steps)
    manifest["control_blocks"] = int(control_blocks)
    manifest["d3_time_contract"] = "D3_MODEL_STEP_CONTROL_BLOCK_ALIGNMENT_V1"

    expected = int(control_blocks)
    lengths = manifest["settings_sequence_json"].map(lambda raw: len(json.loads(str(raw))))
    if not (lengths == expected).all():
        raise RuntimeError("D3 sequence length differs from derived control-block count")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Design D3 multi-actuator sequences from the frozen controller time contract. "
            "The CLI derives control blocks from model horizon/model step/control update so "
            "Step2 horizon steps cannot be confused with supervisory action blocks."
        )
    )
    parser.add_argument("--inp", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--controller-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sequences-per-checkpoint", type=int, default=8)
    parser.add_argument("--perturbation-std", type=float, default=0.20)
    parser.add_argument("--change-probability", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    timing = timing_from_controller_config(_load_json(args.controller_config))
    timing.validate()
    catalog = discover_actuators(args.inp)
    checkpoints = pd.read_csv(args.checkpoints)
    manifest = design_d3_manifest(
        checkpoints=checkpoints,
        catalog=catalog,
        timing=timing,
        sequences_per_checkpoint=args.sequences_per_checkpoint,
        perturbation_std=args.perturbation_std,
        change_probability=args.change_probability,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    print(
        json.dumps(
            {
                "contract": "D3_DESIGN_V2_CONTROLLER_TIME_BOUND",
                "rows": int(len(manifest)),
                "checkpoints": int(manifest["checkpoint_id"].nunique()),
                "all_actuators_eligible": bool(manifest["all_actuators_eligible"].all()),
                "model_step_seconds": int(timing.model_step_seconds),
                "control_update_seconds": int(timing.control_update_seconds),
                "control_block_steps": int(timing.control_block_steps),
                "model_horizon_steps": int(timing.horizon_steps),
                "control_blocks": int(timing.d3_control_blocks),
                "model_horizon_seconds": int(timing.horizon_seconds),
                "out": str(out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
