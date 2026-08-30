"""Conservative two-estimator policy-improvement selector for Project7.

The selector addresses the observed scientific failure mode in which a strong causal Auto-RBC
candidate is present but a learned value ranker selects a different action.  Auto-RBC is treated as
a same-information baseline anchor.  A learned challenger may replace it only when *both* frozen
estimators used by Project7 agree that the challenger has lower TFV return than the anchor.

This is a conservative engineering heuristic inspired by baseline-bootstrapped safe policy
improvement; it is not a theorem that SWMM truth will improve.  Publication promotion therefore
still requires fresh authoritative Development evidence under the separate scientific gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


CONSERVATIVE_SELECTOR_CONTRACT = (
    "PROJECT7_V30_AUTO_RBC_ANCHORED_DUAL_ESTIMATOR_PARETO_IMPROVEMENT_V1"
)


@dataclass(frozen=True)
class CandidateScore:
    source: str
    latent_score: float
    base_step2_score_m3: float


@dataclass(frozen=True)
class ConservativeSelection:
    mode: str
    selected_source: str
    execute_hold: bool
    challenger_selected: bool
    anchor_is_hold: bool
    reason: str


def _finite(value: float) -> bool:
    value = float(value)
    return value == value and value not in (float("inf"), float("-inf"))


def _dominates(a: CandidateScore, b: CandidateScore, tolerance: float) -> bool:
    return (
        float(a.latent_score) < float(b.latent_score) - tolerance
        and float(a.base_step2_score_m3) < float(b.base_step2_score_m3) - tolerance
    )


def select_conservative_improvement(
    *,
    challenger: CandidateScore | None,
    anchor: CandidateScore | None,
    tolerance: float = 1.0e-9,
) -> ConservativeSelection:
    """Select HOLD, Auto-RBC anchor, or one learned challenger.

    Scores are deltas versus HOLD, so zero is the HOLD reference for both estimators.  If an Auto-RBC
    action exists, the challenger must strictly Pareto-dominate the anchor in both estimators before
    it can replace the anchor.  HOLD replaces an actionable anchor only if both estimators say the
    anchor is non-beneficial.  If Auto-RBC itself produces HOLD, a challenger must beat zero in both
    estimators.
    """

    tol = float(tolerance)
    if tol < 0.0:
        raise ValueError("conservative selector tolerance must be non-negative")
    for score in (challenger, anchor):
        if score is None:
            continue
        if not score.source:
            raise ValueError("candidate score requires a source")
        if not _finite(score.latent_score) or not _finite(score.base_step2_score_m3):
            raise ValueError("conservative selector received a non-finite score")

    hold = CandidateScore(source="HOLD", latent_score=0.0, base_step2_score_m3=0.0)
    if anchor is None:
        if challenger is not None and _dominates(challenger, hold, tol):
            return ConservativeSelection(
                mode="CONSENSUS_OVERRIDE_HOLD_ANCHOR",
                selected_source=challenger.source,
                execute_hold=False,
                challenger_selected=True,
                anchor_is_hold=True,
                reason="challenger strictly beats HOLD in both frozen estimators",
            )
        return ConservativeSelection(
            mode="HOLD_ANCHOR",
            selected_source="HOLD",
            execute_hold=True,
            challenger_selected=False,
            anchor_is_hold=True,
            reason="no challenger has dual-estimator evidence to beat HOLD",
        )

    if challenger is not None and _dominates(challenger, anchor, tol):
        return ConservativeSelection(
            mode="CONSENSUS_OVERRIDE_RBC",
            selected_source=challenger.source,
            execute_hold=False,
            challenger_selected=True,
            anchor_is_hold=False,
            reason="challenger strictly Pareto-dominates Auto-RBC anchor",
        )

    # Only dual agreement may suppress the baseline anchor.  One-model disagreement falls back to
    # the causal Auto-RBC action rather than trusting the learned selector.
    if (
        float(anchor.latent_score) >= -tol
        and float(anchor.base_step2_score_m3) >= -tol
    ):
        return ConservativeSelection(
            mode="CONSENSUS_HOLD_OVER_RBC",
            selected_source="HOLD",
            execute_hold=True,
            challenger_selected=False,
            anchor_is_hold=False,
            reason="both frozen estimators rate Auto-RBC no better than HOLD",
        )

    return ConservativeSelection(
        mode="RBC_ANCHOR_FALLBACK",
        selected_source=anchor.source,
        execute_hold=False,
        challenger_selected=False,
        anchor_is_hold=False,
        reason="learned challenger lacks dual-estimator dominance; preserve Auto-RBC anchor",
    )


__all__ = [
    "CONSERVATIVE_SELECTOR_CONTRACT",
    "CandidateScore",
    "ConservativeSelection",
    "select_conservative_improvement",
]
