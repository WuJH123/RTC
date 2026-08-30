from __future__ import annotations

import pytest

from rtc.project7_conservative_policy_improvement import (
    CandidateScore,
    select_conservative_improvement,
)


def score(source: str, latent: float, base: float) -> CandidateScore:
    return CandidateScore(source=source, latent_score=latent, base_step2_score_m3=base)


def test_challenger_must_improve_both_estimators_over_rbc_anchor() -> None:
    selected = select_conservative_improvement(
        challenger=score("LEARNED", -2.0, -30.0),
        anchor=score("AUTO_RBC", -1.0, -20.0),
    )
    assert selected.challenger_selected is True
    assert selected.selected_source == "LEARNED"
    assert selected.mode == "CONSENSUS_OVERRIDE_RBC"


def test_one_estimator_disagreement_falls_back_to_rbc_anchor() -> None:
    selected = select_conservative_improvement(
        challenger=score("LEARNED", -2.0, -10.0),
        anchor=score("AUTO_RBC", -1.0, -20.0),
    )
    assert selected.challenger_selected is False
    assert selected.execute_hold is False
    assert selected.selected_source == "AUTO_RBC"
    assert selected.mode == "RBC_ANCHOR_FALLBACK"


def test_both_estimators_can_suppress_nonbeneficial_rbc_to_hold() -> None:
    selected = select_conservative_improvement(
        challenger=score("LEARNED", 0.5, -1.0),
        anchor=score("AUTO_RBC", 1.0, 2.0),
    )
    assert selected.execute_hold is True
    assert selected.mode == "CONSENSUS_HOLD_OVER_RBC"


def test_hold_anchor_requires_challenger_to_beat_zero_in_both_estimators() -> None:
    rejected = select_conservative_improvement(
        challenger=score("LEARNED", -2.0, 1.0),
        anchor=None,
    )
    assert rejected.execute_hold is True
    accepted = select_conservative_improvement(
        challenger=score("LEARNED", -2.0, -1.0),
        anchor=None,
    )
    assert accepted.execute_hold is False
    assert accepted.challenger_selected is True
    assert accepted.mode == "CONSENSUS_OVERRIDE_HOLD_ANCHOR"


def test_ties_do_not_override_anchor() -> None:
    selected = select_conservative_improvement(
        challenger=score("LEARNED", -1.0, -20.0),
        anchor=score("AUTO_RBC", -1.0, -20.0),
    )
    assert selected.selected_source == "AUTO_RBC"
    assert selected.challenger_selected is False


def test_nonfinite_scores_fail_closed() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        select_conservative_improvement(
            challenger=score("LEARNED", float("nan"), -1.0),
            anchor=None,
        )
