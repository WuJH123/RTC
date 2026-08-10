from __future__ import annotations

from typing import Mapping

from .causal_timing import CausalTimingContract, timing_from_controller_config


PROJECT7_RUNTIME_CONTRACT = "PROJECT7_METHOD_TESTBED_RUNTIME_V1_360MIN_CONTINUOUS"
PRODUCTION_CONTROLLER_CONTRACT = "PRODUCTION_CONTROLLER_CONFIG_V5_TFV_FIRST_TEMPORAL_CONTINUITY"

MODEL_STEP_SECONDS = 300
CONTROL_UPDATE_SECONDS = 600
RECORD_STRIDE_SECONDS = 300
HISTORY_STEPS = 13
FIRST_PROPOSED_CONTROL_MINUTES = 60
PREDICTION_HORIZON_MINUTES = 360
PREDICTION_HORIZON_SECONDS = PREDICTION_HORIZON_MINUTES * 60
EFFECTIVE_WARMUP_MINUTES = 120
MAX_SETTING_DELTA_PER_UPDATE = 0.5

CLAIM_SCOPE = "IDEALIZED_METHODOLOGY_TESTBED_NOT_FIELD_DIGITAL_TWIN"
BASELINE_TRUE_STATE_ADVANTAGE = ("internal_rtc", "auto_rbc", "efd")


def validate_project7_runtime_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate the user-frozen Project7 methodology-testbed runtime contract.

    This contract intentionally separates event initialization from control timing:
    prepared events carry an effective 120-minute pre-rain warm-up, while the first
    Proposed decision occurs 60 minutes after the SWMM simulation START, once the
    13-frame, 5-minute causal history is complete. The prediction horizon is fixed at
    360 minutes and is not re-inferred from Phase-0 censor diagnostics.
    """

    if str(config.get("contract", "")) != PRODUCTION_CONTROLLER_CONTRACT:
        raise ValueError(
            "Project7 requires controller contract "
            f"{PRODUCTION_CONTROLLER_CONTRACT}"
        )
    timing = timing_from_controller_config(dict(config))
    timing.validate(require_full_history_before_first_control=True)

    expected = {
        "model_step_seconds": MODEL_STEP_SECONDS,
        "control_update_seconds": CONTROL_UPDATE_SECONDS,
        "record_stride_seconds": RECORD_STRIDE_SECONDS,
        "history_steps": HISTORY_STEPS,
        "control_start_minutes": FIRST_PROPOSED_CONTROL_MINUTES,
        "horizon_seconds": PREDICTION_HORIZON_SECONDS,
    }
    actual = {
        "model_step_seconds": timing.model_step_seconds,
        "control_update_seconds": timing.control_update_seconds,
        "record_stride_seconds": timing.record_stride_seconds,
        "history_steps": timing.history_steps,
        "control_start_minutes": timing.control_start_minutes,
        "horizon_seconds": timing.horizon_seconds,
    }
    mismatch = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in expected
        if actual[key] != expected[key]
    }
    if mismatch:
        raise ValueError(f"Project7 frozen runtime timing mismatch: {mismatch}")

    methodology = config.get("methodology_testbed")
    if not isinstance(methodology, Mapping):
        raise ValueError("controller config lacks methodology_testbed contract")
    if str(methodology.get("claim_scope", "")) != CLAIM_SCOPE:
        raise ValueError("Project7 claim scope must remain the idealized methodology testbed")
    if int(methodology.get("effective_warmup_minutes", -1)) != EFFECTIVE_WARMUP_MINUTES:
        raise ValueError("Project7 effective warm-up must be 120 minutes")
    if methodology.get("dwf_background_loading") is not True:
        raise ValueError("Project7 must retain DWF as idealized background loading")
    accepted = tuple(str(x) for x in methodology.get("baseline_true_state_advantage", []))
    if accepted != BASELINE_TRUE_STATE_ADVANTAGE:
        raise ValueError(
            "Project7 must explicitly disclose the Internal RTC/Auto-RBC/EFD true-state advantage"
        )

    controller = config.get("controller")
    if not isinstance(controller, Mapping):
        raise ValueError("controller config lacks controller section")
    delta = float(controller.get("max_setting_delta_per_update", -1.0))
    if abs(delta - MAX_SETTING_DELTA_PER_UPDATE) > 1e-12:
        raise ValueError("Project7 max setting delta must be 0.5 per 10-minute update")
    if controller.get("shift_previous_plan_warm_start") is not True:
        raise ValueError("Project7 requires shifted previous-plan warm start for rolling continuity")

    return {
        "contract": PROJECT7_RUNTIME_CONTRACT,
        "claim_scope": CLAIM_SCOPE,
        "effective_warmup_minutes": EFFECTIVE_WARMUP_MINUTES,
        "first_proposed_control_minutes": FIRST_PROPOSED_CONTROL_MINUTES,
        "prediction_horizon_minutes": PREDICTION_HORIZON_MINUTES,
        "model_step_seconds": MODEL_STEP_SECONDS,
        "control_update_seconds": CONTROL_UPDATE_SECONDS,
        "history_steps": HISTORY_STEPS,
        "max_setting_delta_per_update": MAX_SETTING_DELTA_PER_UPDATE,
        "dwf_background_loading": True,
        "baseline_true_state_advantage": list(BASELINE_TRUE_STATE_ADVANTAGE),
        "timing": timing.as_dict(),
    }


def frozen_timing_contract() -> CausalTimingContract:
    timing = CausalTimingContract(
        model_step_seconds=MODEL_STEP_SECONDS,
        control_update_seconds=CONTROL_UPDATE_SECONDS,
        history_steps=HISTORY_STEPS,
        horizon_steps=PREDICTION_HORIZON_SECONDS // MODEL_STEP_SECONDS,
        control_start_minutes=FIRST_PROPOSED_CONTROL_MINUTES,
        record_stride_seconds=RECORD_STRIDE_SECONDS,
    )
    timing.validate(require_full_history_before_first_control=True)
    return timing
