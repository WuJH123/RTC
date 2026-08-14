from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from rtc.step3_candidates_v123 import (
    FirstMoveCandidateDesignV123,
    candidate_coefficients_v123,
    unique_executable_first_moves_v123,
)
from rtc.step3_objective_v123 import TFVPFVObjectiveV123, tfv_pfv_score_v123


class _Basis:
    def __init__(self):
        self.temporal_basis_count = 3
        self.group_count = 3
        self.temporal_basis = np.asarray(
            [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        self.grouping = SimpleNamespace(
            zone_id_by_actuator=np.asarray([0, 0, 1], dtype=np.int64),
            group_id_by_actuator=np.asarray([0, 1, 2], dtype=np.int64),
        )

    def validate(self):
        return None


def _objective() -> TFVPFVObjectiveV123:
    return TFVPFVObjectiveV123(
        pfv_soft_margin_m3=100.0,
        pfv_scale_m3=1000.0,
        tfv_scale_m3=10000.0,
        pfv_penalty_weight=0.5,
    )


def test_pfv_improvement_does_not_buy_worse_tfv() -> None:
    # Candidate 0 has worse TFV but much better PFV. Because PFV is one-sided, its
    # improvement cannot create a negative reward that overrules the TFV objective.
    tfv = torch.tensor([[1000.0, -500.0]])
    pfv = torch.tensor([[-10000.0, 0.0]])
    out = tfv_pfv_score_v123(tfv, pfv, movement=None, contract=_objective())
    assert out["score_m3_equivalent"][0] == 1000.0
    assert out["score_m3_equivalent"][1] == -500.0


def test_pfv_deterioration_is_soft_not_hard() -> None:
    tfv = torch.tensor([[-10000.0, -2000.0]])
    pfv = torch.tensor([[1100.0, 0.0]])
    out = tfv_pfv_score_v123(tfv, pfv, movement=None, contract=_objective())
    # First candidate is penalized by 0.5*10000*(1000/1000)=5000 m3, but remains
    # feasible and can still win because its TFV benefit is large enough.
    assert out["pfv_penalty_m3_equivalent"][0] == 5000.0
    assert out["score_m3_equivalent"][0] == -5000.0
    assert out["score_m3_equivalent"][0] < out["score_m3_equivalent"][1]


def test_pfv_below_margin_has_no_penalty() -> None:
    tfv = torch.tensor([[-1000.0]])
    pfv = torch.tensor([[99.0]])
    out = tfv_pfv_score_v123(tfv, pfv, movement=None, contract=_objective())
    assert out["pfv_penalty_m3_equivalent"].item() == 0.0
    assert out["score_m3_equivalent"].item() == -1000.0


def test_candidate_design_has_no_tail_only_nonhold_coefficients() -> None:
    coeff = candidate_coefficients_v123(
        _Basis(),
        design=FirstMoveCandidateDesignV123(max_cross_zone_pairs=2),
    )
    assert np.allclose(coeff[0], 0.0)
    assert np.all(np.max(np.abs(coeff[1:, 0, :]), axis=1) > 0.0)


def test_executable_first_move_dedup_reports_tail_only_collisions() -> None:
    reference = torch.zeros((4, 1))
    candidates = torch.tensor(
        [
            [[0.0], [0.0], [0.0], [0.0]],
            [[0.0], [0.0], [1.0], [1.0]],
            [[0.5], [0.5], [0.5], [0.5]],
        ]
    )
    kept, report = unique_executable_first_moves_v123(
        candidates, reference, control_block_steps=2
    )
    assert kept.shape[0] == 2
    assert report["raw_candidate_count"] == 3
    assert report["unique_first_move_count"] == 2
    assert report["tail_only_or_passive_like_count"] == 1
