"""V12.5 production selector: engineering anchor by default, learned override by evidence.

The V12.3 finite policies are retained unchanged for historical reproducibility.  V12.5
runs the same causal state, Value models, engineering projection and finite candidate
family twice over the *same* observation: one sparse-RBC anchor arm and one learned arm.
The learned arm may replace the anchor only when its TFV risk is better than the anchor
by a separately calibrated anchor-relative false-benefit margin.  PFV remains a
one-sided soft deterioration penalty *after* that TFV-primary admission gate, so PFV
improvement can never buy a TFV-worse override.

Continuous gradient search is intentionally outside this contract.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from .step2_policy_v123 import FirstMoveTFVPFVPolicyV123, FirstMoveTFVPFVResultV123

V125_POLICY_CONTRACT = "PROJECT7_V125_ANCHOR_DEFAULT_EVIDENCE_GATED_OVERRIDE_V1"
V125_OVERRIDE_CALIBRATION_CONTRACT = "PROJECT7_V125_ANCHOR_RELATIVE_TFV_FALSE_BENEFIT_V1"


@dataclass(frozen=True)
class AnchorOverrideResultV125:
    settings: torch.Tensor
    candidate_valid: bool
    selected_candidate_index: int
    raw_candidate_count: int
    first_move_group_count: int
    tail_only_noop_candidate_count: int
    scenario_count: int
    predicted_delta_tfv_m3: float
    predicted_delta_pfv_m3: float
    tfv_risk_m3: float
    pfv_risk_m3: float
    pfv_soft_excess_m3: float
    pfv_penalty_m3_equivalent: float
    objective_score_m3_equivalent: float
    selected_group_score_m3: float
    false_benefit_margin_m3: float
    scoring_projection_max: float
    knowledge_anchor_candidate_index: int
    knowledge_anchor_selected: bool
    knowledge_anchor_fallback_used: bool
    knowledge_anchor_first_move_delta_max: float
    anchor_tfv_risk_m3: float
    anchor_pfv_risk_m3: float
    anchor_objective_score_m3: float
    learned_tfv_risk_m3: float
    learned_pfv_risk_m3: float
    learned_objective_score_m3: float
    predicted_override_advantage_tfv_m3: float
    anchor_override_margin_m3: float
    learned_override_admitted: bool
    selected_source: str
    policy_mode: str = "anchor_override"
    policy_mode_contract: str = V125_POLICY_CONTRACT

    @property
    def candidate_count(self) -> int:
        return int(self.raw_candidate_count)

    @property
    def controller_source(self) -> str:
        return "MPC_V125"


def _finite_nonnegative(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def _selected_payload(result: FirstMoveTFVPFVResultV123) -> dict[str, Any]:
    return {
        "settings": result.settings,
        "selected_candidate_index": int(result.selected_candidate_index),
        "predicted_delta_tfv_m3": float(result.predicted_delta_tfv_m3),
        "predicted_delta_pfv_m3": float(result.predicted_delta_pfv_m3),
        "tfv_risk_m3": float(result.tfv_risk_m3),
        "pfv_risk_m3": float(result.pfv_risk_m3),
        "pfv_soft_excess_m3": float(result.pfv_soft_excess_m3),
        "pfv_penalty_m3_equivalent": float(result.pfv_penalty_m3_equivalent),
        "objective_score_m3_equivalent": float(result.objective_score_m3_equivalent),
        "selected_group_score_m3": float(result.selected_group_score_m3),
        "scoring_projection_max": float(result.scoring_projection_max),
    }


class AnchorOverridePolicyV125:
    """Fail-safe finite RTC selector with an engineering-anchor default.

    ``anchor_policy`` must be V123 ``anchor_only`` and ``learned_policy`` must be V123
    ``learned_only``.  They should share the same frozen model/basis/normalisation objects.
    The policy never projects after scoring and never performs continuous optimisation.
    """

    accepts_previous_requested_settings = True
    policy_mode = "anchor_override"
    policy_mode_contract = V125_POLICY_CONTRACT
    use_sparse_rbc_anchor = True
    knowledge_anchor_fallback = True

    def __init__(
        self,
        *,
        anchor_policy: FirstMoveTFVPFVPolicyV123,
        learned_policy: FirstMoveTFVPFVPolicyV123,
        anchor_override_margin_m3: float,
        require_objective_improvement: bool = True,
    ) -> None:
        if anchor_policy.policy_mode != "anchor_only":
            raise ValueError("V125 anchor_policy must use V123 anchor_only mode")
        if learned_policy.policy_mode != "learned_only":
            raise ValueError("V125 learned_policy must use V123 learned_only mode")
        # Scientific equivalence: both arms must use the exact same learned Value,
        # basis, normalisation and TFV/PFV objective; only the candidate prior differs.
        for name in ("model", "basis", "prepared", "normalization", "objective"):
            if getattr(anchor_policy, name) is not getattr(learned_policy, name):
                raise ValueError(f"V125 child policies must share {name}")
        if abs(anchor_policy.false_benefit_margin_m3 - learned_policy.false_benefit_margin_m3) > 1e-12:
            raise ValueError("V125 child policies must share the passive false-benefit margin")
        self.anchor_policy = anchor_policy
        self.learned_policy = learned_policy
        self.anchor_override_margin_m3 = _finite_nonnegative(
            anchor_override_margin_m3, name="V125 anchor override margin"
        )
        self.require_objective_improvement = bool(require_objective_improvement)
        # Compatibility attributes consumed by the rolling controller/runtime metadata.
        self.model = learned_policy.model
        self.basis = learned_policy.basis
        self.prepared = learned_policy.prepared
        self.normalization = learned_policy.normalization
        self.objective = learned_policy.objective
        self.false_benefit_margin_m3 = learned_policy.false_benefit_margin_m3
        self.graph = anchor_policy.graph

    def optimize(self, *args: Any, **kwargs: Any) -> AnchorOverrideResultV125:
        anchor = self.anchor_policy.optimize(*args, **kwargs)
        learned = self.learned_policy.optimize(*args, **kwargs)

        if anchor.settings.shape != learned.settings.shape:
            raise RuntimeError("V125 anchor/learned horizon shape mismatch")
        anchor_valid = bool(anchor.candidate_valid)
        learned_valid = bool(learned.candidate_valid)

        anchor_tfv = float(anchor.tfv_risk_m3) if anchor_valid else 0.0
        anchor_pfv = float(anchor.pfv_risk_m3) if anchor_valid else 0.0
        anchor_score = float(anchor.objective_score_m3_equivalent) if anchor_valid else 0.0
        learned_tfv = float(learned.tfv_risk_m3) if learned_valid else 0.0
        learned_pfv = float(learned.pfv_risk_m3) if learned_valid else 0.0
        learned_score = float(learned.objective_score_m3_equivalent) if learned_valid else 0.0
        relative_tfv = learned_tfv - anchor_tfv

        # Primary scientific gate: the learned candidate must improve TFV relative to
        # the engineering anchor by more than the separately calibrated model-error
        # budget.  PFV is never allowed to compensate for failure of this gate.
        relative_tfv_supported = bool(
            learned_valid
            and relative_tfv < -float(self.anchor_override_margin_m3)
        )
        objective_supported = bool(
            (not self.require_objective_improvement)
            or (learned_valid and learned_score < anchor_score)
        )
        override = bool(relative_tfv_supported and objective_supported)

        if override:
            selected = learned
            selected_source = "learned_override"
            payload = _selected_payload(learned)
            candidate_valid = True
        elif anchor_valid:
            selected = anchor
            selected_source = "anchor_default"
            payload = _selected_payload(anchor)
            candidate_valid = True
        else:
            # If the causal sparse-RBC anchor is exactly passive, a learned action may
            # still be used only when it already cleared V123's passive benefit gate.
            if learned_valid:
                selected = learned
                selected_source = "learned_from_passive_anchor"
                payload = _selected_payload(learned)
                candidate_valid = True
            else:
                selected = anchor
                selected_source = "passive"
                payload = _selected_payload(anchor)
                candidate_valid = False

        return AnchorOverrideResultV125(
            settings=payload["settings"].detach(),
            candidate_valid=bool(candidate_valid),
            selected_candidate_index=int(payload["selected_candidate_index"]),
            raw_candidate_count=int(anchor.raw_candidate_count),
            first_move_group_count=int(anchor.first_move_group_count),
            tail_only_noop_candidate_count=int(anchor.tail_only_noop_candidate_count),
            scenario_count=int(anchor.scenario_count),
            predicted_delta_tfv_m3=float(payload["predicted_delta_tfv_m3"]),
            predicted_delta_pfv_m3=float(payload["predicted_delta_pfv_m3"]),
            tfv_risk_m3=float(payload["tfv_risk_m3"]),
            pfv_risk_m3=float(payload["pfv_risk_m3"]),
            pfv_soft_excess_m3=float(payload["pfv_soft_excess_m3"]),
            pfv_penalty_m3_equivalent=float(payload["pfv_penalty_m3_equivalent"]),
            objective_score_m3_equivalent=float(payload["objective_score_m3_equivalent"]),
            selected_group_score_m3=float(payload["selected_group_score_m3"]),
            false_benefit_margin_m3=float(self.false_benefit_margin_m3),
            scoring_projection_max=max(
                float(anchor.scoring_projection_max), float(learned.scoring_projection_max)
            ),
            knowledge_anchor_candidate_index=int(anchor.knowledge_anchor_candidate_index),
            knowledge_anchor_selected=bool(selected_source == "anchor_default"),
            knowledge_anchor_fallback_used=bool(
                selected_source == "anchor_default" and learned_valid is False
            ),
            knowledge_anchor_first_move_delta_max=float(
                anchor.knowledge_anchor_first_move_delta_max
            ),
            anchor_tfv_risk_m3=anchor_tfv,
            anchor_pfv_risk_m3=anchor_pfv,
            anchor_objective_score_m3=anchor_score,
            learned_tfv_risk_m3=learned_tfv,
            learned_pfv_risk_m3=learned_pfv,
            learned_objective_score_m3=learned_score,
            predicted_override_advantage_tfv_m3=float(relative_tfv),
            anchor_override_margin_m3=float(self.anchor_override_margin_m3),
            learned_override_admitted=bool(override),
            selected_source=selected_source,
        )


__all__ = [
    "AnchorOverridePolicyV125",
    "AnchorOverrideResultV125",
    "V125_OVERRIDE_CALIBRATION_CONTRACT",
    "V125_POLICY_CONTRACT",
]
