from __future__ import annotations

import argparse
import json
from pathlib import Path

from .causal_timing import CausalTimingContract
from .inp_runtime import sha256_file
from .project7_contract import (
    CONTROL_UPDATE_SECONDS,
    FIRST_PROPOSED_CONTROL_MINUTES,
    HISTORY_STEPS,
    MAX_SETTING_DELTA_PER_UPDATE,
    MODEL_STEP_SECONDS,
    PREDICTION_HORIZON_MINUTES,
)


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
    """Bind Phase-0 diagnostics to the user-frozen Project7 runtime grid.

    v0.6.9 no longer asks a sustained-step censor heuristic to choose the production horizon.
    The methodology-testbed contract fixes the prediction horizon at 360 minutes. Phase-0
    censoring and pulse/release recovery remain reported diagnostics and must not be hidden,
    but a late sustained depth response does not silently redefine the pre-registered horizon.
    """

    summary_path = Path(phase0_summary_path)
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Phase-0 summary must be a JSON object")
    if not str(raw.get("contract", "")).startswith("PHASE0_D2_STEP_RESPONSE_TIMESCALE_"):
        raise ValueError("not a current Phase-0 step-response timing summary")

    expected = {
        "model_step_seconds": MODEL_STEP_SECONDS,
        "control_update_seconds": CONTROL_UPDATE_SECONDS,
        "history_steps": HISTORY_STEPS,
        "horizon_minutes": PREDICTION_HORIZON_MINUTES,
        "control_start_minutes": FIRST_PROPOSED_CONTROL_MINUTES,
    }
    actual = {
        "model_step_seconds": int(model_step_seconds),
        "control_update_seconds": int(control_update_seconds),
        "history_steps": int(history_steps),
        "horizon_minutes": int(horizon_minutes),
        "control_start_minutes": int(control_start_minutes),
    }
    mismatch = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    if mismatch:
        raise ValueError(f"Project7 frozen timing mismatch: {mismatch}")
    if max_setting_delta_per_update is None:
        raise ValueError("Project7 timing freeze requires max_setting_delta_per_update=0.5")
    if abs(float(max_setting_delta_per_update) - MAX_SETTING_DELTA_PER_UPDATE) > 1e-12:
        raise ValueError("Project7 max setting delta must be 0.5 per 10-minute update")

    horizon_seconds = int(horizon_minutes) * 60
    timing = CausalTimingContract(
        model_step_seconds=int(model_step_seconds),
        control_update_seconds=int(control_update_seconds),
        history_steps=int(history_steps),
        horizon_steps=horizon_seconds // int(model_step_seconds),
        control_start_minutes=int(control_start_minutes),
        record_stride_seconds=int(model_step_seconds),
    )
    timing.validate()
    _ = timing.d3_control_blocks

    candidate = raw.get("candidate_production_timing")
    payload: dict[str, object] = {
        "contract": "RTC_PHASE0_TIMING_FREEZE_V2_PROJECT7_360MIN",
        "phase0_summary": str(summary_path.resolve()),
        "phase0_summary_sha256": sha256_file(summary_path),
        "phase0_contract": str(raw.get("contract")),
        "phase0_horizon_censored": bool(raw.get("horizon_censored", False)),
        "phase0_censor_role": "diagnostic_not_horizon_selection_gate",
        "horizon_selection_basis": "USER_FROZEN_IDEALIZED_METHODOLOGY_TESTBED_360MIN",
        "phase0_candidate_production_timing": candidate if isinstance(candidate, dict) else {},
        "timing": timing.as_dict(),
        "model_step_seconds": timing.model_step_seconds,
        "control_update_seconds": timing.control_update_seconds,
        "record_stride_seconds": timing.record_stride_seconds,
        "control_start_minutes": timing.control_start_minutes,
        "controller": {
            "history_steps": timing.history_steps,
            "horizon_steps": timing.horizon_steps,
            "max_setting_delta_per_update": float(max_setting_delta_per_update),
            "enforce_cross_decision_target_continuity": True,
            "enforce_sequential_horizon_continuity": True,
        },
        "status": "TIMING_ONLY_RESOLVED_NOT_FULL_PRODUCTION_POLICY_CONFIG",
        "instruction": (
            "The Project7 production grid is now frozen by study contract: 5-min model step, "
            "10-min control update, first Proposed decision at elapsed 60 min, 13-frame history, "
            "and 360-min prediction horizon. Preserve Phase-0 censor/pulse findings as diagnostics. "
            "Resolve only development forecast/optimizer/readback/runtime fields before closed-loop runs."
        ),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bind Phase-0 evidence to the frozen Project7 360-minute controller timing"
    )
    parser.add_argument("--phase0-summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-step-seconds", type=int, default=MODEL_STEP_SECONDS)
    parser.add_argument("--control-update-seconds", type=int, default=CONTROL_UPDATE_SECONDS)
    parser.add_argument("--history-steps", type=int, default=HISTORY_STEPS)
    parser.add_argument("--horizon-minutes", type=int, default=PREDICTION_HORIZON_MINUTES)
    parser.add_argument(
        "--control-start-minutes", type=int, default=FIRST_PROPOSED_CONTROL_MINUTES
    )
    parser.add_argument(
        "--max-setting-delta-per-update",
        type=float,
        default=MAX_SETTING_DELTA_PER_UPDATE,
    )
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
