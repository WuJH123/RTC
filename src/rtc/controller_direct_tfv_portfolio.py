"""Authoritative-controller telemetry adapter for the policy-return portfolio.

The adapter preserves score==execute and adds deployment identity only.  It never changes the
numerical action selected by the wrapped controller.  Future decisions carry a continuation SHA,
causal-context fingerprint, selected-action SHA and exact prior-command prefix SHA so they can be
joined to the existing exact-return bank without event-name inference.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from rtc.closed_loop import CausalObservation, ControllerAction
from rtc.controller_direct_tfv_safe import MemorySafeDirectTFVAuthoritativeController
from rtc.project7_v26_historical_supervision import (
    action_sha256,
    causal_context_sha256,
    normalize_context,
)


DIRECT_TFV_PORTFOLIO_TELEMETRY_CONTRACT = (
    "PROJECT7_DIRECT_TFV_POLICY_RETURN_PORTFOLIO_TELEMETRY_V3_RUNTIME_IDENTITY"
)
RUNTIME_DECISION_IDENTITY_CONTRACT = "PROJECT7_RUNTIME_DECISION_IDENTITY_V1"


def _canonical_sha(value: object) -> str:
    raw = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class PortfolioMemorySafeDirectTFVAuthoritativeController(
    MemorySafeDirectTFVAuthoritativeController
):
    """Preserve score==execute while reporting the actual portfolio selection and identity."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._runtime_prefix_actions: dict[int, dict[str, float]] = {}

    def _runtime_context_fingerprint(self, obs: CausalObservation) -> str:
        """Rebuild the same four-field causal context used by exact-return supervision.

        This is telemetry-only.  The duplicated frozen Step1 forward is not consumed by the policy
        and therefore cannot change the selected action.
        """
        if len(self.observed_history) < self.config.history_steps:
            return ""
        static = torch.as_tensor(
            self.graph.static_node_features,
            dtype=torch.float32,
            device=self.device,
        )
        edges = torch.as_tensor(self.graph.edge_index, dtype=torch.long, device=self.device)
        with torch.no_grad():
            state = self.step1(
                torch.as_tensor(
                    np.stack(self.observed_history)[None],
                    dtype=torch.float32,
                    device=self.device,
                ),
                torch.as_tensor(
                    np.stack(self.mask_history)[None],
                    dtype=torch.float32,
                    device=self.device,
                ),
                static,
                edges,
                torch.as_tensor(
                    np.stack(self.context_history)[None],
                    dtype=torch.float32,
                    device=self.device,
                ),
            )
        rain = self.forecast.forecast(
            np.stack(self.rainfall_history),
            horizon_steps=self.config.horizon_steps,
        )
        normalized = normalize_context(
            {
                "current_state": state.detach().cpu().to(torch.float32).numpy(),
                "rainfall_scenarios": np.asarray(rain, dtype=np.float32),
                "active_target": np.asarray(obs.actuator_target_setting, dtype=np.float32),
                "previous_actuator_flow": np.asarray(obs.actuator_flow_m3s, dtype=np.float32),
            }
        )
        return "" if normalized is None else causal_context_sha256(normalized)

    def _runtime_identity(
        self,
        *,
        obs: CausalObservation,
        action: ControllerAction,
    ) -> dict[str, str | bool]:
        ids = tuple(str(value) for value in self.graph.actuator_ids)
        if set(action.settings) != set(ids):
            raise ValueError("runtime identity requires the complete graph actuator target")
        selected = np.asarray([float(action.settings[aid]) for aid in ids], dtype=np.float32)
        prefix_payload = {
            str(elapsed): self._runtime_prefix_actions[elapsed]
            for elapsed in sorted(self._runtime_prefix_actions)
        }
        prefix_sha = _canonical_sha(prefix_payload)
        context_sha = self._runtime_context_fingerprint(obs)
        selected_sha = action_sha256(selected)
        self._runtime_prefix_actions[int(obs.elapsed_seconds)] = {
            aid: float(action.settings[aid]) for aid in ids
        }
        return {
            "runtime_decision_identity_contract": RUNTIME_DECISION_IDENTITY_CONTRACT,
            "runtime_recorded_prefix_action_sha256": prefix_sha,
            "runtime_causal_context_fingerprint_sha256": context_sha,
            "runtime_causal_context_fingerprint_present": len(context_sha) == 64,
            "runtime_selected_action_sha256": selected_sha,
        }

    def decide(
        self, obs: CausalObservation, *, observation_already_recorded: bool = False
    ) -> ControllerAction:
        action = super().decide(obs, observation_already_recorded=observation_already_recorded)
        diagnostics = dict(action.diagnostics or {})
        diagnostics.update(self._runtime_identity(obs=obs, action=action))

        result = self._direct_mpc_adapter.last_result
        if result is None or not hasattr(result, "policy_return_portfolio_candidate_count"):
            return ControllerAction(
                settings=action.settings,
                source=action.source,
                diagnostics=diagnostics,
            )

        candidate_count = int(getattr(result, "policy_return_portfolio_candidate_count", 0))
        selected = str(getattr(result, "policy_return_portfolio_selected_source", "HOLD"))
        # Later direct-value runtimes do not necessarily copy the historical
        # policy_return_admission_passed field into the generic adapter result. Falling back to the
        # generic admission/candidate validity is telemetry-only and reflects the command that was
        # actually returned by the numerical controller.
        passed = bool(
            getattr(
                result,
                "policy_return_admission_passed",
                getattr(result, "admission_passed", getattr(result, "candidate_valid", False)),
            )
        )
        inner = self._direct_mpc_adapter.inner
        continuation = str(
            getattr(inner, "policy_return_parent_continuation_sha256", "")
        ).lower()
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
                "runtime_continuation_policy_sha256": continuation,
                "runtime_continuation_policy_sha256_present": len(continuation) == 64,
            }
        )
        # The generic Direct-TFV adapter recognizes the historical L-BFGS-B source token. V14 uses
        # a richer portfolio source label, so restore the canonical ACTION/HOLD source after the
        # numerical command has already been produced. This changes telemetry only, not settings.
        source = (
            "MPC_DIRECT_TFV_RECEDING"
            if passed
            else "LATCH_PREVIOUS_TARGET_DIRECT_TFV_UPPER_BOUND"
        )
        return ControllerAction(settings=action.settings, source=source, diagnostics=diagnostics)


__all__ = [
    "DIRECT_TFV_PORTFOLIO_TELEMETRY_CONTRACT",
    "RUNTIME_DECISION_IDENTITY_CONTRACT",
    "PortfolioMemorySafeDirectTFVAuthoritativeController",
]
