"""Authoritative-controller telemetry adapter for the Direct-TFV policy-return portfolio."""
from __future__ import annotations

from rtc.closed_loop import CausalObservation, ControllerAction
from rtc.controller_direct_tfv_safe import MemorySafeDirectTFVAuthoritativeController


DIRECT_TFV_PORTFOLIO_TELEMETRY_CONTRACT = (
    "PROJECT7_DIRECT_TFV_POLICY_RETURN_PORTFOLIO_TELEMETRY_V2_ADAPTER_SAFE"
)


def _runtime_policy_return_passed(result: object) -> bool:
    """Return the decision carried by the runtime wrapper without assuming a legacy field name."""
    return bool(
        getattr(
            result,
            "policy_return_admission_passed",
            getattr(result, "admission_passed", getattr(result, "candidate_valid", False)),
        )
    )


class PortfolioMemorySafeDirectTFVAuthoritativeController(
    MemorySafeDirectTFVAuthoritativeController
):
    """Preserve score==execute while reporting the actual portfolio selection.

    Historical portfolio telemetry read ``policy_return_admission_passed`` directly from the
    low-level Step3 result.  ``DirectTFVRuntimeMPCAdapter`` intentionally wraps that result into a
    smaller runtime dataclass and does not carry that historical field, even though it does preserve
    the equivalent ``admission_passed`` / ``candidate_valid`` decision.  Treat the latter as the
    canonical runtime fallback so telemetry cannot relabel a numerically executed ACTION as HOLD.
    This changes reporting only; the target settings have already been produced by the controller.
    """

    def decide(
        self, obs: CausalObservation, *, observation_already_recorded: bool = False
    ) -> ControllerAction:
        action = super().decide(obs, observation_already_recorded=observation_already_recorded)
        result = self._direct_mpc_adapter.last_result
        if result is None or not hasattr(result, "policy_return_portfolio_candidate_count"):
            return action
        diagnostics = dict(action.diagnostics or {})
        candidate_count = int(getattr(result, "policy_return_portfolio_candidate_count", 0))
        selected = str(getattr(result, "policy_return_portfolio_selected_source", "HOLD"))
        passed = _runtime_policy_return_passed(result)
        policy_mode_contract = str(getattr(self._direct_mpc_adapter.inner, "policy_mode_contract", ""))
        # V26/V27 select directly against HOLD=0.  They do not use the old calibrated one-sided
        # admission layer; leave that fact explicit rather than inheriting the generic legacy flag.
        direct_value_without_conformal = policy_mode_contract.startswith(
            ("PROJECT7_OPERATIONAL_DEVELOPMENT_V26_", "PROJECT7_OPERATIONAL_DEVELOPMENT_V27_")
        )
        diagnostics.update(
            {
                "direct_tfv_portfolio_telemetry_contract": DIRECT_TFV_PORTFOLIO_TELEMETRY_CONTRACT,
                "policy_return_portfolio_contract": str(
                    getattr(result, "policy_return_portfolio_contract", "")
                ),
                "policy_return_portfolio_candidate_count": candidate_count,
                "policy_return_portfolio_selected_source": selected,
                "policy_return_portfolio_sources": list(
                    getattr(result, "policy_return_portfolio_sources", ())
                ),
                "policy_return_portfolio_scores_m3": list(
                    getattr(result, "policy_return_portfolio_scores_m3", ())
                ),
                "policy_return_portfolio_upper_bounds_m3": list(
                    getattr(result, "policy_return_portfolio_upper_bounds_m3", ())
                ),
                "policy_return_admission_passed_runtime": passed,
                "calibrated_runtime_action_class": "ACTION" if passed else "HOLD",
                "calibrated_one_sided_admission_used": (
                    False if direct_value_without_conformal else diagnostics.get(
                        "calibrated_one_sided_admission_used", True
                    )
                ),
                "runtime_policy_mode_contract": policy_mode_contract,
            }
        )
        # The generic Direct-TFV adapter recognizes the historical L-BFGS-B source token.  Portfolio
        # policies use richer source labels, so restore canonical ACTION/HOLD telemetry after the
        # numerical command has already been produced.  This never changes the settings.
        source = (
            "MPC_DIRECT_TFV_RECEDING"
            if passed
            else "LATCH_PREVIOUS_TARGET_DIRECT_TFV_UPPER_BOUND"
        )
        return ControllerAction(settings=action.settings, source=source, diagnostics=diagnostics)


__all__ = [
    "DIRECT_TFV_PORTFOLIO_TELEMETRY_CONTRACT",
    "PortfolioMemorySafeDirectTFVAuthoritativeController",
    "_runtime_policy_return_passed",
]
