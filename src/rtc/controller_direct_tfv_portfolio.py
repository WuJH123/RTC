"""Authoritative-controller telemetry adapter for the Direct-TFV policy-return portfolio."""
from __future__ import annotations

from rtc.closed_loop import CausalObservation, ControllerAction
from rtc.controller_direct_tfv_safe import MemorySafeDirectTFVAuthoritativeController


DIRECT_TFV_PORTFOLIO_TELEMETRY_CONTRACT = (
    "PROJECT7_DIRECT_TFV_POLICY_RETURN_PORTFOLIO_TELEMETRY_V3_STRUCTURED_V27"
)
_V27_PREFIX = "V27_DECISION_AWARE|"


def _runtime_policy_return_passed(result: object) -> bool:
    """Return the decision carried by the runtime wrapper without assuming a legacy field name."""
    return bool(
        getattr(
            result,
            "policy_return_admission_passed",
            getattr(result, "admission_passed", getattr(result, "candidate_valid", False)),
        )
    )


def _v27_structured_diagnostics(message: object) -> dict[str, str | bool | int | float]:
    """Decode the existing V27 diagnostic token into stable structured decision telemetry.

    The legacy string remains untouched for backward compatibility; new logs no longer require an
    audit script to recursively search arbitrary string-valued fields.
    """
    raw = str(message or "")
    if not raw.startswith(_V27_PREFIX):
        return {}
    values: dict[str, str] = {}
    for token in raw[len(_V27_PREFIX) :].split("|"):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        values[key] = value

    def as_bool(key: str) -> bool:
        return values.get(key, "").strip().lower() == "true"

    def as_int(key: str) -> int:
        try:
            return int(float(values[key]))
        except (KeyError, TypeError, ValueError):
            return 0

    def as_float(key: str) -> float | None:
        try:
            return float(values[key])
        except (KeyError, TypeError, ValueError):
            return None

    result: dict[str, str | bool | int | float] = {
        "v27_decision_diagnostic_contract": "PROJECT7_V27_STRUCTURED_DECISION_TELEMETRY_V1",
        "v27_report_clip_hit_candidate_count": as_int("clip_hits"),
        "v27_raw_report_clip_hit_candidate_count": as_int("raw_clip_hits"),
        "v27_q95_binding_candidate_count": as_int("q95_binding_candidates"),
        "v27_raw_best_source": values.get("raw_best", ""),
        "v27_supported_best_source": values.get("supported_best", ""),
        "v27_q95_selection_changed": as_bool("q95_selection_changed"),
        "v27_auto_rbc_shadow_present": as_bool("shadow_present"),
        "v27_auto_rbc_shadow_selected": as_bool("shadow_selected"),
        "v27_auto_rbc_shadow_duplicate": as_bool("shadow_duplicate"),
    }
    for source_key, target_key in (
        ("latent_min", "v27_supported_latent_min"),
        ("latent_max", "v27_supported_latent_max"),
        ("raw_best_latent", "v27_raw_best_latent"),
        ("supported_best_latent", "v27_supported_best_latent"),
    ):
        value = as_float(source_key)
        if value is not None:
            result[target_key] = value
    return result


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
        passed = _runtime_policy_return_passed(result)
        policy_mode_contract = str(getattr(self._direct_mpc_adapter.inner, "policy_mode_contract", ""))
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
                    False
                    if direct_value_without_conformal
                    else diagnostics.get("calibrated_one_sided_admission_used", True)
                ),
                "runtime_policy_mode_contract": policy_mode_contract,
            }
        )
        diagnostics.update(_v27_structured_diagnostics(getattr(result, "scipy_message", "")))
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
    "_v27_structured_diagnostics",
]
