"""Deterministic execution profiles for fast Project7 Step2 development.

The scientific model/time/action contract is shared across profiles.  ``smoke`` and ``dev``
only reduce Development-data coverage and training repetitions so code/architecture ideas can
be rejected quickly.  They are never valid Policy-Lock/Final evidence.  ``full`` preserves the
current V128 training census and curriculum exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Sequence

from .step2_train_v127_control import V127ControlTrainingDesign

V128_EXECUTION_PROFILE_CONTRACT = "PROJECT7_V128_SMOKE_DEV_FULL_EXECUTION_PROFILES_V1"


@dataclass(frozen=True)
class V128ExecutionProfile:
    name: str
    d2_fit_groups: int | None
    d3_fit_groups: int | None
    d4_fit_groups: int | None
    d2_holdout_eval_groups: int | None
    d3_holdout_eval_groups: int | None
    d4_audit_eval_groups: int | None
    hydraulic_epochs: int
    teacher_stride: int
    rollout_horizons: tuple[int, ...]
    rollout_candidates_per_group: int
    objective_epochs: int
    scientific_claim_allowed: bool
    final_checkpoint_allowed: bool
    expensive_downstream_allowed: bool

    @property
    def is_full(self) -> bool:
        return self.name == "full"


PROFILES: dict[str, V128ExecutionProfile] = {
    "smoke": V128ExecutionProfile(
        name="smoke",
        d2_fit_groups=4,
        d3_fit_groups=4,
        d4_fit_groups=4,
        d2_holdout_eval_groups=2,
        d3_holdout_eval_groups=2,
        d4_audit_eval_groups=2,
        hydraulic_epochs=1,
        teacher_stride=1,
        rollout_horizons=(12,),
        rollout_candidates_per_group=1,
        objective_epochs=1,
        scientific_claim_allowed=False,
        final_checkpoint_allowed=False,
        expensive_downstream_allowed=False,
    ),
    "dev": V128ExecutionProfile(
        name="dev",
        d2_fit_groups=24,
        d3_fit_groups=24,
        d4_fit_groups=12,
        d2_holdout_eval_groups=8,
        d3_holdout_eval_groups=8,
        d4_audit_eval_groups=8,
        hydraulic_epochs=2,
        teacher_stride=2,
        rollout_horizons=(12, 24),
        rollout_candidates_per_group=2,
        objective_epochs=1,
        scientific_claim_allowed=False,
        final_checkpoint_allowed=False,
        expensive_downstream_allowed=False,
    ),
    "full": V128ExecutionProfile(
        name="full",
        d2_fit_groups=None,
        d3_fit_groups=None,
        d4_fit_groups=None,
        d2_holdout_eval_groups=None,
        d3_holdout_eval_groups=None,
        d4_audit_eval_groups=None,
        hydraulic_epochs=4,
        teacher_stride=4,
        rollout_horizons=(12, 24),
        rollout_candidates_per_group=2,
        objective_epochs=3,
        scientific_claim_allowed=True,
        final_checkpoint_allowed=True,
        expensive_downstream_allowed=True,
    ),
}


def get_execution_profile(name: str) -> V128ExecutionProfile:
    key = str(name).strip().lower()
    if key not in PROFILES:
        raise ValueError(f"unknown V128 execution profile: {name!r}; expected smoke/dev/full")
    return PROFILES[key]


def _stable_key(name: str, *, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{name}".encode("utf-8")).hexdigest()


def deterministic_subset(
    names: Sequence[str], count: int | None, *, salt: str
) -> list[str]:
    ordered = sorted((str(name) for name in names), key=lambda x: (_stable_key(x, salt=salt), x))
    if count is None:
        return ordered
    if count <= 0:
        raise ValueError("development subset count must be positive or None")
    if len(ordered) < count:
        raise ValueError(
            f"development subset {salt!r} requested {count} groups but only {len(ordered)} exist"
        )
    return ordered[: int(count)]


def profile_groups(
    profile: V128ExecutionProfile,
    *,
    fit_d2: Sequence[str],
    fit_d3: Sequence[str],
    hold_d2: Sequence[str],
    hold_d3: Sequence[str],
    d4_fit: Sequence[str],
    d4_audit: Sequence[str],
    one_group: bool = False,
) -> dict[str, list[str]]:
    def n(value: int | None) -> int | None:
        return 1 if one_group else value

    return {
        "fit_d2": deterministic_subset(fit_d2, n(profile.d2_fit_groups), salt=f"{profile.name}:fit_d2"),
        "fit_d3": deterministic_subset(fit_d3, n(profile.d3_fit_groups), salt=f"{profile.name}:fit_d3"),
        "hold_d2": deterministic_subset(hold_d2, n(profile.d2_holdout_eval_groups), salt=f"{profile.name}:hold_d2"),
        "hold_d3": deterministic_subset(hold_d3, n(profile.d3_holdout_eval_groups), salt=f"{profile.name}:hold_d3"),
        "d4_fit": deterministic_subset(d4_fit, n(profile.d4_fit_groups), salt=f"{profile.name}:d4_fit"),
        "d4_audit": deterministic_subset(d4_audit, n(profile.d4_audit_eval_groups), salt=f"{profile.name}:d4_audit"),
    }


def apply_profile_to_design(
    base: V127ControlTrainingDesign,
    profile: V128ExecutionProfile,
) -> V127ControlTrainingDesign:
    result = replace(
        base,
        hydraulic_epochs=int(profile.hydraulic_epochs),
        teacher_stride=int(profile.teacher_stride),
        rollout_horizons=tuple(profile.rollout_horizons),
        rollout_candidates_per_group=int(profile.rollout_candidates_per_group),
        objective_epochs=int(profile.objective_epochs),
    )
    result.validate()
    return result


__all__ = [
    "PROFILES",
    "V128_EXECUTION_PROFILE_CONTRACT",
    "V128ExecutionProfile",
    "apply_profile_to_design",
    "deterministic_subset",
    "get_execution_profile",
    "profile_groups",
]
