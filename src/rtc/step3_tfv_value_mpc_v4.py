"""Support-aware Direct-TFV receding MPC V4.

V3 fixed the default active-set ceiling at the TrainFit joint q90 changed-facility count. The
Development audit showed the strongest H360-beneficial decisions binding that ceiling while a
release-dominated network rewards broader coordinated operation. V4 therefore uses q95 *only when
that q95 was derived from the frozen TrainFit action distribution*. Legacy V4 Step2 checkpoints do
not contain the additive density-support fields and fail back to q90, so no unsupported expansion is
silently enabled.

This is a support-geometry correction, not a tuned performance threshold: all 109 facilities are
still screened, the TFV-only objective is unchanged, the 0.5 setting slew is unchanged, and the
candidate is still executed only when its predicted delta TFV is negative.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

from .step2_tfv_support import DIRECT_TFV_JOINT_DENSITY_SUPPORT_CONTRACT
from .step3_tfv_value_mpc_v3 import (
    DirectTFVMPCDesignV3,
    DirectTFVMPCResultV3,
    DirectTFVRecedingMPC,
)


DIRECT_TFV_STEP3_CONTRACT = "PROJECT7_DIRECT_TFV_109ACT_RECEDING_MPC_V4"
SUPPORTED_ACTIVE_SET_QUANTILES = ("q90", "q95", "q99")


@dataclass(frozen=True)
class DirectTFVMPCDesignV4(DirectTFVMPCDesignV3):
    active_support_quantile: str = "q95"

    def validate(self) -> None:
        super().validate()
        if self.active_support_quantile not in SUPPORTED_ACTIVE_SET_QUANTILES:
            raise ValueError(
                f"active_support_quantile must be one of {SUPPORTED_ACTIVE_SET_QUANTILES}"
            )


class DirectTFVRecedingMPCV4(DirectTFVRecedingMPC):
    """V3 optimizer with a fail-closed, TrainFit-derived active-density ceiling."""

    policy_mode = "direct_tfv_all109_receding_mpc_v4"
    policy_mode_contract = DIRECT_TFV_STEP3_CONTRACT

    def __init__(
        self,
        *,
        model: Any,
        graph: Any,
        normalization: Any,
        action_support: Mapping[str, Any],
        design: DirectTFVMPCDesignV4 = DirectTFVMPCDesignV4(),
    ) -> None:
        super().__init__(
            model=model,
            graph=graph,
            normalization=normalization,
            action_support=action_support,
            design=design,
        )
        self.design = design

    def active_support_quantile_effective(self) -> str:
        requested = str(self.design.active_support_quantile)
        if requested == "q90":
            return requested
        extension = str(self.action_support.get("joint_density_extension_contract", ""))
        if extension != DIRECT_TFV_JOINT_DENSITY_SUPPORT_CONTRACT:
            return "q90"
        key = f"joint_changed_facility_count_{requested}"
        if key not in self.action_support:
            return "q90"
        return requested

    def active_support_ceiling(self) -> int:
        quantile = self.active_support_quantile_effective()
        key = f"joint_changed_facility_count_{quantile}"
        value = float(
            self.action_support.get(
                key,
                self.action_support.get("joint_changed_facility_count_q90", 1.0),
            )
        )
        observed_max = int(
            self.action_support.get(
                "joint_changed_facility_count_max",
                max(1, int(math.ceil(value))),
            )
        )
        return max(1, min(109, observed_max, int(math.ceil(value))))

    def _active_count(self, beneficial_count: int) -> int:
        if beneficial_count <= 0:
            return 0
        if self.design.active_facility_count > 0:
            requested = int(self.design.active_facility_count)
            # Even an explicit Development override may not exceed the maximum joint density
            # observed in TrainFit when the V2 density extension is available.
            extension = str(self.action_support.get("joint_density_extension_contract", ""))
            if extension == DIRECT_TFV_JOINT_DENSITY_SUPPORT_CONTRACT:
                requested = min(
                    requested,
                    int(self.action_support.get("joint_changed_facility_count_max", requested)),
                )
        else:
            requested = self.active_support_ceiling()
        return int(min(beneficial_count, min(109, requested)))

    def optimize(self, **kwargs: Any) -> DirectTFVMPCResultV3:
        """Run the inherited numerical solver but stamp the current scientific contract."""

        result = super().optimize(**kwargs)
        return replace(
            result,
            policy_mode=self.policy_mode,
            policy_mode_contract=self.policy_mode_contract,
        )


__all__ = [
    "DIRECT_TFV_STEP3_CONTRACT",
    "SUPPORTED_ACTIVE_SET_QUANTILES",
    "DirectTFVMPCDesignV4",
    "DirectTFVRecedingMPCV4",
]
