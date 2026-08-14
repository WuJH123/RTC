"""First-move-rich candidate design for Project7 V12.3.

V12.2 scored 97 full-horizon sequences but only 11--13 distinct executable first
moves in the T5 development run; most candidates differed only in an unexecuted tail.
V12.3 makes first-action coverage an explicit design invariant.  Every non-HOLD seed
has a non-zero coefficient in temporal basis 0, which is the only basis component with
unit support at the first 10-min control block of the frozen hat basis.

This remains a finite-policy search until the Step2 gradient evidence gate passes.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch

from .step2_control_basis_v60 import ControlBasisV60

V123_CANDIDATE_CONTRACT = "PROJECT7_V123_FIRST_MOVE_RICH_FINITE_SHOOTING_V1"


@dataclass(frozen=True)
class FirstMoveCandidateDesignV123:
    single_group_magnitudes: tuple[float, ...] = (0.35, 0.65)
    pair_magnitude: float = 0.50
    include_same_zone_pairs: bool = True
    include_cross_zone_pairs: bool = True
    max_cross_zone_pairs: int = 24
    seed: int = 42

    def validate(self) -> None:
        if not self.single_group_magnitudes:
            raise ValueError("V123 requires at least one single-group magnitude")
        if any(not 0.0 < float(value) <= 1.0 for value in self.single_group_magnitudes):
            raise ValueError("V123 single-group magnitudes must lie in (0,1]")
        if not 0.0 < float(self.pair_magnitude) <= 1.0:
            raise ValueError("V123 pair magnitude must lie in (0,1]")
        if self.max_cross_zone_pairs < 0:
            raise ValueError("V123 max_cross_zone_pairs cannot be negative")


def _add_unique(rows: list[np.ndarray], seen: set[bytes], value: np.ndarray) -> None:
    value = np.asarray(value, dtype=np.float32)
    key = value.tobytes()
    if key not in seen:
        rows.append(value)
        seen.add(key)


def candidate_coefficients_v123(
    basis: ControlBasisV60,
    *,
    design: FirstMoveCandidateDesignV123 = FirstMoveCandidateDesignV123(),
) -> np.ndarray:
    """Generate HOLD plus deterministic first-move-active multi-facility seeds."""
    basis.validate()
    design.validate()
    k, g = basis.temporal_basis_count, basis.group_count
    if k <= 0 or g <= 0:
        raise ValueError("V123 control basis is empty")
    # Frozen hat basis should make basis-0 the direct first-block degree of freedom.
    first_support = np.asarray(basis.temporal_basis[0], dtype=np.float64)
    if not np.isclose(first_support[0], 1.0, atol=1e-6) or np.any(np.abs(first_support[1:]) > 1e-6):
        raise ValueError("V123 requires temporal basis 0 to exclusively control the first block")

    rows = [np.zeros((k, g), dtype=np.float32)]
    seen = {rows[0].tobytes()}

    # 1) Ensure every control group has both opening/closing first-move directions at
    # two amplitudes.  This is the minimum coverage missing from V12.2 online shooting.
    for group in range(g):
        for magnitude in design.single_group_magnitudes:
            for sign in (-1.0, 1.0):
                coeff = np.zeros((k, g), dtype=np.float32)
                coeff[0, group] = float(sign * magnitude)
                _add_unique(rows, seen, coeff)

    zones = np.asarray(basis.grouping.zone_id_by_actuator, dtype=np.int64)
    group_ids = np.asarray(basis.grouping.group_id_by_actuator, dtype=np.int64)
    zone_by_group = np.full(g, -1, dtype=np.int64)
    for group in range(g):
        values = np.unique(zones[group_ids == group])
        if values.size != 1:
            raise ValueError("V123 group spans multiple spatial zones")
        zone_by_group[group] = int(values[0])

    # 2) Same-zone pair moves capture coordinated conveyance/storage operation.
    if design.include_same_zone_pairs:
        for zone in sorted(set(zone_by_group.tolist())):
            groups = np.flatnonzero(zone_by_group == zone).tolist()
            for a, b in combinations(groups, 2):
                for sign_a, sign_b in ((1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)):
                    coeff = np.zeros((k, g), dtype=np.float32)
                    coeff[0, a] = float(sign_a * design.pair_magnitude)
                    coeff[0, b] = float(sign_b * design.pair_magnitude)
                    _add_unique(rows, seen, coeff)

    # 3) A bounded deterministic sample of cross-zone pairs lets the finite policy
    # redistribute storage/conveyance between regions without exploding candidate count.
    if design.include_cross_zone_pairs and design.max_cross_zone_pairs:
        pairs = [(a, b) for a, b in combinations(range(g), 2) if zone_by_group[a] != zone_by_group[b]]
        rng = np.random.default_rng(int(design.seed))
        if len(pairs) > design.max_cross_zone_pairs:
            chosen = rng.choice(len(pairs), size=design.max_cross_zone_pairs, replace=False)
            pairs = [pairs[int(i)] for i in sorted(chosen.tolist())]
        for a, b in pairs:
            for sign in (-1.0, 1.0):
                coeff = np.zeros((k, g), dtype=np.float32)
                coeff[0, a] = float(sign * design.pair_magnitude)
                coeff[0, b] = float(sign * design.pair_magnitude)
                _add_unique(rows, seen, coeff)

    result = np.stack(rows)
    # Structural invariant: no non-HOLD seed is allowed to hide all action in the tail.
    if result.shape[0] <= 1 or np.any(np.max(np.abs(result[1:, 0, :]), axis=1) <= 0.0):
        raise RuntimeError("V123 generated a tail-only non-HOLD seed")
    return result


def unique_executable_first_moves_v123(
    candidate_settings: torch.Tensor,
    reference_settings: torch.Tensor,
    *,
    control_block_steps: int,
    atol: float = 1.0e-6,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Deduplicate after engineering projection by the command that can execute now."""
    if candidate_settings.ndim != 3:
        raise ValueError("V123 candidates must be [C,H,A]")
    if reference_settings.ndim == 3 and reference_settings.shape[0] == 1:
        reference_settings = reference_settings[0]
    if reference_settings.ndim != 2 or reference_settings.shape != candidate_settings.shape[1:]:
        raise ValueError("V123 reference/candidate sequence shapes differ")
    if control_block_steps <= 0:
        raise ValueError("V123 control_block_steps must be positive")

    first = candidate_settings[:, :control_block_steps].mean(dim=1)
    reference_first = reference_settings[:control_block_steps].mean(dim=0)
    quant = torch.round(first / float(atol)).to(torch.int64).detach().cpu().numpy()
    reference_key = torch.round(reference_first / float(atol)).to(torch.int64).detach().cpu().numpy().tobytes()
    keep: list[int] = []
    seen: set[bytes] = set()
    passive_like = 0
    for index, row in enumerate(quant):
        key = row.tobytes()
        if key == reference_key:
            passive_like += 1
        if key not in seen:
            seen.add(key)
            keep.append(index)
    if not keep:
        raise RuntimeError("V123 executable first-move dedup removed every candidate")
    kept = candidate_settings[torch.as_tensor(keep, dtype=torch.long, device=candidate_settings.device)]
    return kept, {
        "raw_candidate_count": int(candidate_settings.shape[0]),
        "unique_first_move_count": int(len(keep)),
        "tail_only_or_passive_like_count": int(max(passive_like - 1, 0)),
    }


__all__ = [
    "FirstMoveCandidateDesignV123",
    "V123_CANDIDATE_CONTRACT",
    "candidate_coefficients_v123",
    "unique_executable_first_moves_v123",
]
