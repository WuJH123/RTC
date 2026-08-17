"""Calibrated Direct-TFV receding MPC V5.

V5 keeps the V4 all-109 screening, q95 HOLD-relative action support and bounded H120 optimisation.
The policy change is evidence-driven admission: an optimised sequence is executable only when an
optimizer-aware one-sided residual upper bound remains below zero. The margin includes exact
same-prefix H360 SWMM residuals from optimizer-selected Development plans, so it addresses the
selection-induced optimism observed in continuous Step3 rather than relying on cached candidates
alone. The TFV-only objective is unchanged.

A plan whose first executed 10-minute block is identical to HOLD is also treated as HOLD. Future-only
benefit cannot be counted as an executed action because the controller re-observes before those
future blocks are reached.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .direct_tfv_admission import (
    DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT,
    admission_margin_m3,
)
from .step3_tfv_value_mpc_v3 import DirectTFVMPCResultV3
from .step3_tfv_value_mpc_v4 import DirectTFVMPCDesignV4, DirectTFVRecedingMPCV4


DIRECT_TFV_STEP3_CONTRACT = "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V5"


@dataclass(frozen=True)
class DirectTFVMPCResultV5(DirectTFVMPCResultV3):
    raw_optimized_predicted_delta_tfv_m3: float = 0.0
    admission_margin_m3: float = 0.0
    admission_upper_bound_m3: float = 0.0
    admission_margin_kind: str = "none"
    admission_passed: bool = False
    calibrated_admission_contract: str = DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT
    optimized_candidate_settings: torch.Tensor | None = None


class DirectTFVRecedingMPCV5(DirectTFVRecedingMPCV4):
    """V4 optimizer plus optimizer-aware one-sided HOLD-H360 action admission."""

    policy_mode = "direct_tfv_all109_receding_mpc_v5"
    policy_mode_contract = DIRECT_TFV_STEP3_CONTRACT

    def __init__(
        self,
        *,
        model: Any,
        graph: Any,
        normalization: Any,
        action_support: Mapping[str, Any],
        admission_calibration: Mapping[str, Any],
        design: DirectTFVMPCDesignV4 = DirectTFVMPCDesignV4(),
    ) -> None:
        super().__init__(
            model=model,
            graph=graph,
            normalization=normalization,
            action_support=action_support,
            design=design,
        )
        if str(admission_calibration.get("contract")) != DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT:
            raise ValueError("Direct-TFV V5 requires current optimizer-aware admission calibration")
        if admission_calibration.get("development_only") is not True:
            raise ValueError("Direct-TFV V5 admission calibration must be Development-only evidence")
        if str(admission_calibration.get("reference_semantics", "")) != "HOLD_ACTIVE_TARGET_H360":
            raise ValueError("Direct-TFV V5 admission calibration has the wrong reference semantics")
        if int(admission_calibration.get("optimizer_replay_count", 0)) < 4:
            raise ValueError("Direct-TFV V5 requires exact optimizer-replay residual evidence")
        for key in ("global_margin_m3", "dense_margin_m3"):
            value = float(admission_calibration.get(key, float("nan")))
            if not torch.isfinite(torch.tensor(value)) or value < 0.0:
                raise ValueError(f"Direct-TFV V5 admission calibration has invalid {key}")
        self.admission_calibration = dict(admission_calibration)

    @staticmethod
    def _wrap_result(
        result: DirectTFVMPCResultV3,
        *,
        settings: torch.Tensor,
        selected_source: str,
        executed_score_m3: float,
        raw_score_m3: float,
        margin_m3: float,
        upper_bound_m3: float,
        margin_kind: str,
        passed: bool,
        optimized_candidate_settings: torch.Tensor,
    ) -> DirectTFVMPCResultV5:
        values = dict(vars(result))
        values.update(
            {
                "settings": settings,
                "predicted_delta_tfv_m3": float(executed_score_m3),
                "selected_source": str(selected_source),
                "policy_mode": DirectTFVRecedingMPCV5.policy_mode,
                "policy_mode_contract": DirectTFVRecedingMPCV5.policy_mode_contract,
                "raw_optimized_predicted_delta_tfv_m3": float(raw_score_m3),
                "admission_margin_m3": float(margin_m3),
                "admission_upper_bound_m3": float(upper_bound_m3),
                "admission_margin_kind": str(margin_kind),
                "admission_passed": bool(passed),
                "calibrated_admission_contract": DIRECT_TFV_ADMISSION_CALIBRATION_CONTRACT,
                "optimized_candidate_settings": optimized_candidate_settings,
            }
        )
        return DirectTFVMPCResultV5(**values)

    def optimize(
        self,
        *,
        current_state: torch.Tensor,
        rainfall: torch.Tensor,
        previous_actuator_flow: torch.Tensor,
        current_settings: torch.Tensor,
        active_target: torch.Tensor,
    ) -> DirectTFVMPCResultV5:
        result = super().optimize(
            current_state=current_state,
            rainfall=rainfall,
            previous_actuator_flow=previous_actuator_flow,
            current_settings=current_settings,
            active_target=active_target,
        )
        optimized_candidate = result.settings
        raw_score = float(result.predicted_delta_tfv_m3)
        if result.selected_source != "DIRECT_TFV_RECEDING_LBFGSB":
            return self._wrap_result(
                result,
                settings=result.settings,
                selected_source=result.selected_source,
                executed_score_m3=float(result.predicted_delta_tfv_m3),
                raw_score_m3=raw_score,
                margin_m3=0.0,
                upper_bound_m3=float(result.predicted_delta_tfv_m3),
                margin_kind="none",
                passed=False,
                optimized_candidate_settings=optimized_candidate,
            )

        margin, kind = admission_margin_m3(
            self.admission_calibration, int(result.active_facility_count)
        )
        upper = float(raw_score + margin)
        hold = self._hold_sequence(active_target)
        if int(result.first_move_changed_facility_count) <= 0:
            return self._wrap_result(
                result,
                settings=hold,
                selected_source="HOLD_NO_EXECUTABLE_FIRST_MOVE",
                executed_score_m3=0.0,
                raw_score_m3=raw_score,
                margin_m3=margin,
                upper_bound_m3=upper,
                margin_kind=kind,
                passed=False,
                optimized_candidate_settings=optimized_candidate,
            )
        if upper >= 0.0:
            return self._wrap_result(
                result,
                settings=hold,
                selected_source="HOLD_CALIBRATED_TFV_UPPER_BOUND_NONNEGATIVE",
                executed_score_m3=0.0,
                raw_score_m3=raw_score,
                margin_m3=margin,
                upper_bound_m3=upper,
                margin_kind=kind,
                passed=False,
                optimized_candidate_settings=optimized_candidate,
            )
        return self._wrap_result(
            result,
            settings=result.settings,
            selected_source="DIRECT_TFV_RECEDING_LBFGSB",
            executed_score_m3=raw_score,
            raw_score_m3=raw_score,
            margin_m3=margin,
            upper_bound_m3=upper,
            margin_kind=kind,
            passed=True,
            optimized_candidate_settings=optimized_candidate,
        )


__all__ = [
    "DIRECT_TFV_STEP3_CONTRACT",
    "DirectTFVMPCResultV5",
    "DirectTFVRecedingMPCV5",
]
