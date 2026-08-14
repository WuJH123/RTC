"""Deterministic TrainFit-only D4 action-support planning for Project7.

D4 is an outcome-free *design* layer.  It densifies supervision around the verified
sparse-state RBC first move.  V2 additionally freezes rainfall-group fit/audit roles and
isolates first-move credit: all candidates at one checkpoint share the exact same anchor
continuation after the executable 10-minute block.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Sequence

import numpy as np

D4_ACTION_SUPPORT_CONTRACT_V125 = "PROJECT7_STEP2_D4_ACTION_SUPPORT_V125_V2_COMMON_CONTINUATION"
D4_SPLIT_SALT_V125 = "PROJECT7_V125_D4_RAINFALL_SPLIT_V1"


@dataclass(frozen=True)
class D4ActionSupportContractV125:
    """Frozen development-only D4 neighbourhood design."""

    max_checkpoints: int = 48
    anchor_scales: tuple[float, ...] = (0.50, 0.75, 1.00)
    local_fraction: float = 0.25
    max_active_groups: int = 3
    max_delta_per_update: float = 0.50
    audit_fraction: float = 0.25

    def validate(self) -> None:
        if self.max_checkpoints <= 0:
            raise ValueError("D4 max_checkpoints must be positive")
        if not self.anchor_scales or any(not 0.0 < x <= 1.0 for x in self.anchor_scales):
            raise ValueError("D4 anchor scales must lie in (0,1]")
        if not 0.0 < self.local_fraction <= 0.5:
            raise ValueError("D4 local_fraction must lie in (0,0.5]")
        if self.max_active_groups <= 0:
            raise ValueError("D4 max_active_groups must be positive")
        if not 0.0 < self.max_delta_per_update <= 1.0:
            raise ValueError("D4 max_delta_per_update must lie in (0,1]")
        if not 0.0 < self.audit_fraction < 0.5:
            raise ValueError("D4 audit_fraction must lie in (0,0.5)")


def deterministic_d4_rainfall_roles_v125(
    rainfall_groups: Sequence[str],
    *,
    audit_fraction: float = 0.25,
    salt: str = D4_SPLIT_SALT_V125,
) -> dict[str, str]:
    """Freeze outcome-blind fit/audit roles at rainfall-group level.

    The assignment depends only on group identity and the frozen salt.  It must be run
    before any D4 SWMM outcome is generated.  All rows from one rainfall group share one
    role, preventing branch/checkpoint leakage.
    """
    groups = sorted(set(str(x) for x in rainfall_groups))
    if len(groups) < 2:
        raise ValueError("D4 fit/audit split requires at least two rainfall groups")
    fraction = float(audit_fraction)
    if not 0.0 < fraction < 0.5:
        raise ValueError("D4 audit fraction must lie in (0,0.5)")
    audit_count = min(len(groups) - 1, max(1, int(math.ceil(len(groups) * fraction))))
    ranked = sorted(
        groups,
        key=lambda group: (
            hashlib.sha256(f"{salt}|{group}".encode("utf-8")).hexdigest(),
            group,
        ),
    )
    audit = set(ranked[:audit_count])
    return {group: ("audit" if group in audit else "fit") for group in groups}


def common_anchor_continuation_sequence_v125(
    first_move_target: np.ndarray,
    anchor_sequence: np.ndarray,
    *,
    control_block_steps: int = 2,
) -> np.ndarray:
    """Build the D4 counterfactual with candidate-only first 10 min and common tail."""
    target = np.asarray(first_move_target, dtype=np.float32).reshape(-1)
    anchor = np.asarray(anchor_sequence, dtype=np.float32)
    block = int(control_block_steps)
    if anchor.ndim != 2 or anchor.shape[1] != target.size or block <= 0 or block > anchor.shape[0]:
        raise ValueError("D4 common-continuation shape mismatch")
    if not np.isfinite(anchor).all() or not np.isfinite(target).all():
        raise ValueError("D4 common-continuation values must be finite")
    result = anchor.copy()
    result[:block] = target[None, :]
    # Strong invariant: candidate identity must disappear after the executable block.
    if block < result.shape[0] and not np.array_equal(result[block:], anchor[block:]):
        raise RuntimeError("D4 common continuation was not preserved exactly")
    return result


def action_sequence_sha256_v125(sequence: np.ndarray) -> str:
    values = np.asarray(sequence, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("D4 action sequence must be a finite [time,actuator] matrix")
    return hashlib.sha256(values.astype("<f4", copy=False).tobytes(order="C")).hexdigest()


def _bounded_target(
    current: np.ndarray,
    delta: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    max_delta_per_update: float,
) -> np.ndarray:
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    delta = np.asarray(delta, dtype=np.float64).reshape(-1)
    lower = np.asarray(lower, dtype=np.float64).reshape(-1)
    upper = np.asarray(upper, dtype=np.float64).reshape(-1)
    if not (current.shape == delta.shape == lower.shape == upper.shape):
        raise ValueError("D4 target arrays must have identical actuator shape")
    if np.any(lower > upper):
        raise ValueError("D4 actuator bounds are invalid")
    bounded_delta = np.clip(delta, -float(max_delta_per_update), float(max_delta_per_update))
    return np.clip(current + bounded_delta, lower, upper).astype(np.float32)


def knowledge_neighbourhood_first_moves_v125(
    current: np.ndarray,
    anchor_target: np.ndarray,
    group_id_by_actuator: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    contract: D4ActionSupportContractV125 = D4ActionSupportContractV125(),
) -> list[tuple[str, np.ndarray]]:
    """Return a small deterministic first-move neighbourhood around the RBC anchor."""
    contract.validate()
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    anchor = np.asarray(anchor_target, dtype=np.float64).reshape(-1)
    groups = np.asarray(group_id_by_actuator, dtype=np.int64).reshape(-1)
    if current.shape != anchor.shape or groups.size != current.size:
        raise ValueError("D4 current/anchor/group actuator mismatch")
    anchor_delta = np.clip(
        anchor - current,
        -float(contract.max_delta_per_update),
        float(contract.max_delta_per_update),
    )

    raw: list[tuple[str, np.ndarray]] = [("hold", current.astype(np.float32))]
    for scale in contract.anchor_scales:
        raw.append((
            f"anchor_scale_{scale:.2f}",
            _bounded_target(
                current, anchor_delta * float(scale), lower, upper,
                max_delta_per_update=contract.max_delta_per_update,
            ),
        ))

    unique_groups = sorted(set(int(x) for x in groups.tolist()))
    strength = []
    for gid in unique_groups:
        mask = groups == gid
        strength.append((float(np.sum(np.abs(anchor_delta[mask]))), gid))
    active_groups = [
        gid for value, gid in sorted(strength, key=lambda x: (-x[0], x[1])) if value > 1e-8
    ][: int(contract.max_active_groups)]
    for gid in active_groups:
        mask = groups == gid
        component = np.zeros_like(anchor_delta)
        component[mask] = anchor_delta[mask]
        for sign, label in ((-1.0, "minus"), (1.0, "plus")):
            raw.append((
                f"anchor_group_{gid}_{label}{int(round(contract.local_fraction * 100)):02d}",
                _bounded_target(
                    current,
                    anchor_delta + sign * float(contract.local_fraction) * component,
                    lower,
                    upper,
                    max_delta_per_update=contract.max_delta_per_update,
                ),
            ))

    result: list[tuple[str, np.ndarray]] = []
    seen: set[bytes] = set()
    for family, target in raw:
        key = np.round(np.asarray(target, dtype=np.float64), 7).astype(np.float32).tobytes()
        if key in seen:
            continue
        seen.add(key)
        result.append((family, np.asarray(target, dtype=np.float32)))
    return result


def action_support_gap_v125(
    current: np.ndarray,
    anchor_target: np.ndarray,
    candidate_first_moves: np.ndarray,
    *,
    max_delta_per_update: float = 0.50,
) -> dict[str, float]:
    current = np.asarray(current, dtype=np.float64).reshape(-1)
    anchor = np.asarray(anchor_target, dtype=np.float64).reshape(-1)
    candidates = np.asarray(candidate_first_moves, dtype=np.float64)
    if candidates.ndim != 2 or candidates.shape[1] != current.size or anchor.size != current.size:
        raise ValueError("D4 support-gap actuator shape mismatch")
    scale = max(float(max_delta_per_update), 1e-6)
    anchor_delta = (anchor - current) / scale
    candidate_delta = (candidates - current[None, :]) / scale
    distance = np.mean(np.abs(candidate_delta - anchor_delta[None, :]), axis=1)
    nearest = int(np.argmin(distance))
    active = np.abs(anchor_delta) > 1e-7
    if np.any(active):
        direction_agreement = float(np.mean(
            np.sign(candidate_delta[nearest, active]) == np.sign(anchor_delta[active])
        ))
    else:
        direction_agreement = 1.0
    return {
        "nearest_anchor_l1_normalized": float(distance[nearest]),
        "nearest_candidate_index": float(nearest),
        "nearest_direction_agreement": direction_agreement,
        "anchor_mean_abs_delta_normalized": float(np.mean(np.abs(anchor_delta))),
        "anchor_active_fraction": float(np.mean(active)),
    }


def select_gap_balanced_checkpoints_v125(
    records: Sequence[dict[str, object]], *, max_checkpoints: int,
) -> list[dict[str, object]]:
    if max_checkpoints <= 0:
        raise ValueError("D4 selection budget must be positive")
    if not records:
        return []
    by_rain: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_rain.setdefault(str(record["rainfall_group"]), []).append(dict(record))
    for rain in by_rain:
        by_rain[rain].sort(key=lambda r: (
            -float(r["nearest_anchor_l1_normalized"]),
            str(r.get("event_id", "")),
            str(r.get("checkpoint_id", "")),
        ))
    rain_order = sorted(by_rain, key=lambda rain: (
        -float(by_rain[rain][0]["nearest_anchor_l1_normalized"]), rain
    ))
    selected: list[dict[str, object]] = []
    cursor = {rain: 0 for rain in rain_order}
    while len(selected) < min(max_checkpoints, len(records)):
        progressed = False
        for rain in rain_order:
            i = cursor[rain]
            if i >= len(by_rain[rain]):
                continue
            selected.append(by_rain[rain][i])
            cursor[rain] = i + 1
            progressed = True
            if len(selected) >= max_checkpoints:
                break
        if not progressed:
            break
    return selected


def first_move_family_summary_v125(plans: Iterable[tuple[str, np.ndarray]]) -> dict[str, int]:
    result: dict[str, int] = {}
    for family, _ in plans:
        result[family] = result.get(family, 0) + 1
    return dict(sorted(result.items()))


__all__ = [
    "D4_ACTION_SUPPORT_CONTRACT_V125",
    "D4_SPLIT_SALT_V125",
    "D4ActionSupportContractV125",
    "action_sequence_sha256_v125",
    "action_support_gap_v125",
    "common_anchor_continuation_sequence_v125",
    "deterministic_d4_rainfall_roles_v125",
    "first_move_family_summary_v125",
    "knowledge_neighbourhood_first_moves_v125",
    "select_gap_balanced_checkpoints_v125",
]
