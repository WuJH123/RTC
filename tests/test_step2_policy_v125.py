from __future__ import annotations

from types import SimpleNamespace

import torch

from rtc.step2_policy_v123 import FirstMoveTFVPFVResultV123
from rtc.step2_policy_v125 import AnchorOverridePolicyV125


def _result(*, tfv: float, pfv: float, score: float, valid: bool, value: float, anchor: bool = False):
    return FirstMoveTFVPFVResultV123(
        settings=torch.full((4, 3), value, dtype=torch.float32),
        candidate_valid=valid,
        selected_candidate_index=7 if valid else 0,
        raw_candidate_count=10,
        first_move_group_count=8,
        tail_only_noop_candidate_count=1,
        scenario_count=1,
        predicted_delta_tfv_m3=tfv if valid else 0.0,
        predicted_delta_pfv_m3=pfv if valid else 0.0,
        tfv_risk_m3=tfv if valid else 0.0,
        pfv_risk_m3=pfv if valid else 0.0,
        pfv_soft_excess_m3=max(pfv, 0.0) if valid else 0.0,
        pfv_penalty_m3_equivalent=max(pfv, 0.0) if valid else 0.0,
        objective_score_m3_equivalent=score if valid else 0.0,
        false_benefit_margin_m3=10.0,
        scoring_projection_max=0.0,
        knowledge_anchor_candidate_index=9 if anchor else -1,
        knowledge_anchor_selected=anchor,
        knowledge_anchor_fallback_used=False,
        knowledge_anchor_first_move_delta_max=0.2 if anchor else 0.0,
        policy_mode="anchor_only" if anchor else "learned_only",
    )


class _Child:
    def __init__(self, mode: str, result, shared) -> None:
        self.policy_mode = mode
        self.result = result
        self.false_benefit_margin_m3 = 10.0
        self.graph = object() if mode == "anchor_only" else None
        for name in ("model", "basis", "prepared", "normalization", "objective"):
            setattr(self, name, shared)

    def optimize(self, *args, **kwargs):
        return self.result


def _policy(anchor, learned, *, margin=25.0):
    shared = object()
    return AnchorOverridePolicyV125(
        anchor_policy=_Child("anchor_only", anchor, shared),
        learned_policy=_Child("learned_only", learned, shared),
        anchor_override_margin_m3=margin,
    )


def test_anchor_is_default_when_learned_beats_passive_but_not_anchor() -> None:
    anchor = _result(tfv=-100.0, pfv=20.0, score=-80.0, valid=True, value=0.6, anchor=True)
    learned = _result(tfv=-70.0, pfv=-30.0, score=-100.0, valid=True, value=0.4)
    result = _policy(anchor, learned).optimize()
    assert result.selected_source == "anchor_default"
    assert not result.learned_override_admitted
    assert result.knowledge_anchor_selected
    assert torch.all(result.settings == 0.6)


def test_pfv_improvement_cannot_buy_worse_tfv() -> None:
    anchor = _result(tfv=-100.0, pfv=100.0, score=0.0, valid=True, value=0.6, anchor=True)
    learned = _result(tfv=-90.0, pfv=-100.0, score=-190.0, valid=True, value=0.4)
    result = _policy(anchor, learned, margin=0.0).optimize()
    assert result.selected_source == "anchor_default"
    assert not result.learned_override_admitted


def test_learned_override_requires_anchor_relative_margin_and_objective() -> None:
    anchor = _result(tfv=-100.0, pfv=10.0, score=-90.0, valid=True, value=0.6, anchor=True)
    learned = _result(tfv=-140.0, pfv=5.0, score=-135.0, valid=True, value=0.4)
    result = _policy(anchor, learned, margin=25.0).optimize()
    assert result.selected_source == "learned_override"
    assert result.learned_override_admitted
    assert result.predicted_override_advantage_tfv_m3 == -40.0
    assert torch.all(result.settings == 0.4)


def test_passive_anchor_can_use_learned_candidate_that_clears_passive_gate() -> None:
    anchor = _result(tfv=0.0, pfv=0.0, score=0.0, valid=False, value=0.5, anchor=True)
    learned = _result(tfv=-80.0, pfv=0.0, score=-80.0, valid=True, value=0.3)
    result = _policy(anchor, learned, margin=25.0).optimize()
    assert result.selected_source == "learned_override"
    assert result.candidate_valid
