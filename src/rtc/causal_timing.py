from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CausalTimingContract:
    model_step_seconds: int
    control_update_seconds: int
    history_steps: int
    horizon_steps: int
    control_start_minutes: int
    record_stride_seconds: int

    @property
    def history_span_seconds(self) -> int:
        # t=0 is explicitly observed, so H frames cover (H-1) intervals.
        return (self.history_steps - 1) * self.model_step_seconds

    @property
    def horizon_seconds(self) -> int:
        return self.horizon_steps * self.model_step_seconds

    @property
    def control_start_seconds(self) -> int:
        return self.control_start_minutes * 60

    @property
    def control_block_steps(self) -> int:
        return self.control_update_seconds // self.model_step_seconds

    @property
    def d3_control_blocks(self) -> int:
        """Return the exact number of full control blocks spanning the model horizon.

        D3 stores one setting vector per supervisory control block but Step2 stores one state
        transition per model step.  Mixing those two notions of ``horizon_steps`` previously
        allowed a 24-step, 5-min model horizon to be misread as 24 ten-minute control blocks.
        Formal D3 therefore requires the frozen model horizon to contain an integer number of
        complete control blocks.
        """

        block_steps = self.control_block_steps
        if self.horizon_steps % block_steps:
            raise ValueError(
                "D3 requires the model horizon to be an integer number of control blocks: "
                f"horizon_steps={self.horizon_steps}, control_block_steps={block_steps}"
            )
        return self.horizon_steps // block_steps

    def validate(self, *, require_full_history_before_first_control: bool = True) -> None:
        if min(
            self.model_step_seconds,
            self.control_update_seconds,
            self.record_stride_seconds,
        ) <= 0:
            raise ValueError("model/control/record cadences must be positive")
        if self.history_steps < 2 or self.horizon_steps < 1:
            raise ValueError("history_steps must be >=2 and horizon_steps must be >=1")
        if self.control_start_minutes < 0:
            raise ValueError("control_start_minutes must be non-negative")
        if self.control_update_seconds % self.model_step_seconds:
            raise ValueError("control_update_seconds must be an integer multiple of model_step_seconds")
        if self.control_start_seconds % self.model_step_seconds:
            raise ValueError("first control epoch must align with the model/observation grid")
        # Formal RTC uses an event-clock control grid so decisions are reproducible and
        # comparable across strategies: e.g. 60,70,80... minutes for a 10-min controller.
        if self.control_start_seconds % self.control_update_seconds:
            raise ValueError("first control epoch must align with the control-update grid")
        if require_full_history_before_first_control and self.control_start_seconds < self.history_span_seconds:
            raise ValueError(
                "first control epoch occurs before a full causal Step1 history is available: "
                f"control_start={self.control_start_seconds}s, history_span={self.history_span_seconds}s"
            )
        if self.horizon_seconds < self.control_update_seconds:
            raise ValueError("prediction horizon must cover at least one complete control interval")

    def as_dict(self) -> dict[str, int | str]:
        self.validate()
        payload: dict[str, int | str] = {
            "contract": "CAUSAL_RTC_TIMING_V1_T0_INCLUDED",
            "model_step_seconds": self.model_step_seconds,
            "control_update_seconds": self.control_update_seconds,
            "history_steps": self.history_steps,
            "history_span_seconds": self.history_span_seconds,
            "horizon_steps": self.horizon_steps,
            "horizon_seconds": self.horizon_seconds,
            "control_start_minutes": self.control_start_minutes,
            "control_start_seconds": self.control_start_seconds,
            "control_block_steps": self.control_block_steps,
            "record_stride_seconds": self.record_stride_seconds,
            "initial_observation_elapsed_seconds": 0,
            "timeline": "observe current state -> reconstruct -> forecast causally -> optimize -> write target -> hold until next control epoch -> verify readback at next decision",
        }
        if self.horizon_steps % self.control_block_steps == 0:
            payload["d3_control_blocks"] = self.d3_control_blocks
        return payload


def timing_from_controller_config(config: dict[str, object]) -> CausalTimingContract:
    controller = config.get("controller")
    if not isinstance(controller, dict):
        raise ValueError("controller config must contain a controller object")
    required = ("history_steps", "horizon_steps")
    missing = [key for key in required if key not in controller]
    if missing:
        raise ValueError(f"controller config lacks explicit timing fields: {missing}")
    required_top = (
        "model_step_seconds",
        "control_update_seconds",
        "control_start_minutes",
    )
    missing_top = [key for key in required_top if key not in config]
    if missing_top:
        raise ValueError(f"controller config lacks explicit runtime timing fields: {missing_top}")
    model_step = int(config["model_step_seconds"])
    return CausalTimingContract(
        model_step_seconds=model_step,
        control_update_seconds=int(config["control_update_seconds"]),
        history_steps=int(controller["history_steps"]),
        horizon_steps=int(controller["horizon_steps"]),
        control_start_minutes=int(config["control_start_minutes"]),
        record_stride_seconds=int(config.get("record_stride_seconds", model_step)),
    )