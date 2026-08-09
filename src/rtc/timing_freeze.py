from __future__ import annotations

import argparse
import json
from pathlib import Path

from .causal_timing import CausalTimingContract
from .inp_runtime import sha256_file


def freeze_phase0_timing(
    *,
    phase0_summary_path: str | Path,
    model_step_seconds: int,
    control_update_seconds: int,
    history_steps: int,
    horizon_minutes: int,
    control_start_minutes: int,
    max_setting_delta_per_update: float | None = None,
) -> dict[str, object]:
    summary_path = Path(phase0_summary_path)
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Phase-0 summary must be a JSON object")
    if not str(raw.get("contract", "")).startswith("PHASE0_D2_STEP_RESPONSE_TIMESCALE_"):
        raise ValueError("not a current Phase-0 step-response timing summary")
    if raw.get("horizon_censored") is True:
        raise ValueError(
            "Phase-0 response peaks are horizon-censored; lengthen the pilot horizon before freezing production timing"
        )
    if min(model_step_seconds, control_update_seconds, history_steps, horizon_minutes) <= 0:
        raise ValueError("timing values must be positive")
    horizon_seconds = int(horizon_minutes) * 60
    if horizon_seconds % int(model_step_seconds):
        raise ValueError("horizon_minutes must contain an integer number of model steps")
    if max_setting_delta_per_update is not None and not 0.0 <= float(max_setting_delta_per_update) <= 1.0:
        raise ValueError("max_setting_delta_per_update must lie in [0,1] or be null")

    timing = CausalTimingContract(
        model_step_seconds=int(model_step_seconds),
        control_update_seconds=int(control_update_seconds),
        history_steps=int(history_steps),
        horizon_steps=horizon_seconds // int(model_step_seconds),
        control_start_minutes=int(control_start_minutes),
        record_stride_seconds=int(model_step_seconds),
    )
    timing.validate()
    # D3 must be able to represent the entire horizon in whole supervisory blocks.
    _ = timing.d3_control_blocks

    candidate = raw.get("candidate_production_timing")
    payload: dict[str, object] = {
        "contract": "RTC_PHASE0_TIMING_FREEZE_V1",
        "phase0_summary": str(summary_path.resolve()),
        "phase0_summary_sha256": sha256_file(summary_path),
        "phase0_contract": str(raw.get("contract")),
        "phase0_horizon_censored": bool(raw.get("horizon_censored", False)),
        "phase0_candidate_production_timing": candidate if isinstance(candidate, dict) else {},
        "timing": timing.as_dict(),
        # Keep the public controller-config field layout needed by rtc-design-d3.
        "model_step_seconds": timing.model_step_seconds,
        "control_update_seconds": timing.control_update_seconds,
        "record_stride_seconds": timing.record_stride_seconds,
        "control_start_minutes": timing.control_start_minutes,
        "controller": {
            "history_steps": timing.history_steps,
            "horizon_steps": timing.horizon_steps,
            "max_setting_delta_per_update": (
                None
                if max_setting_delta_per_update is None
                else float(max_setting_delta_per_update)
            ),
        },
        "status": "TIMING_ONLY_RESOLVED_NOT_FULL_PRODUCTION_POLICY_CONFIG",
        "instruction": (
            "Use this file to bind production-grid data cadence and D3 design. Before closed-loop Proposed runs, "
            "resolve forecast, optimizer, objective-near-optimality and runtime/readback fields in the production controller config."
        ),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze production data/controller timing from non-censored Phase-0 evidence"
    )
    parser.add_argument("--phase0-summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-step-seconds", type=int, required=True)
    parser.add_argument("--control-update-seconds", type=int, required=True)
    parser.add_argument("--history-steps", type=int, required=True)
    parser.add_argument("--horizon-minutes", type=int, required=True)
    parser.add_argument("--control-start-minutes", type=int, required=True)
    parser.add_argument("--max-setting-delta-per-update", type=float)
    args = parser.parse_args()

    payload = freeze_phase0_timing(
        phase0_summary_path=args.phase0_summary,
        model_step_seconds=args.model_step_seconds,
        control_update_seconds=args.control_update_seconds,
        history_steps=args.history_steps,
        horizon_minutes=args.horizon_minutes,
        control_start_minutes=args.control_start_minutes,
        max_setting_delta_per_update=args.max_setting_delta_per_update,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
