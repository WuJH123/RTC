"""Authoritative-controller telemetry adapter for the V14 policy-return portfolio."""
from __future__ import annotations

from rtc.closed_loop import CausalObservation, ControllerAction
from rtc.controller_direct_tfv_safe import MemorySafeDirectTFVAuthoritativeController


DIRECT_TFV_PORTFOLIO_TELEMETRY_CONTRACT = (
    "PROJECT7_DIRECT_TFV_POLICY_RETURN_PORTFOLIO_TELEMETRY_V1"
)


class PortfolioMemorySafeDirectTFVAuthoritativeController(
    MemorySafeDirectTFVAuthoritativeController
):
    """Preserve score==execute while reporting the actual portfolio selection."""

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
        passed = bool(getattr(result, "policy_return_admission_passed", False))
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
                "calibrated_runtime_action_class": "ACTION" if passed else "HOLD",
            }
        )
        # The generic Direct-TFV adapter recognizes the historical L-BFGS-B source token.  V14 uses
        # a richer portfolio source label, so restore the canonical ACTION/HOLD source after the
        # numerical command has already been produced.  This changes telemetry only, not settings.
        source = (
            "MPC_DIRECT_TFV_RECEDING"
            if passed
            else "LATCH_PREVIOUS_TARGET_DIRECT_TFV_UPPER_BOUND"
        )
        return ControllerAction(settings=action.settings, source=source, diagnostics=diagnostics)


__all__ = [
    "DIRECT_TFV_PORTFOLIO_TELEMETRY_CONTRACT",
    "PortfolioMemorySafeDirectTFVAuthoritativeController",
]
