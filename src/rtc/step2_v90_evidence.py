"""Fail-closed acceptance guard for Project7 Step2 V9 diagnostic evidence.

A state-sufficiency result is scientific evidence only when it is reproducible from the
current diagnostic implementation and frozen Train-only lineage.  This module prevents
older/pre-merge JSON summaries from being silently treated as canonical after the V9
branch changes.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .step2_v90_contract import (
    DirectHydraulicEffectLossContractV90,
    LEVEL_A,
    LEVEL_B,
    LEVEL_C,
    V90_CONTRACT,
)

V90_EVIDENCE_GUARD_CONTRACT = "PROJECT7_STEP2_V90_EVIDENCE_GUARD_V1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_LEVELS = (LEVEL_A, LEVEL_B, LEVEL_C)
_REQUIRED_BOUNDARY_FALSE = (
    "swmm_run",
    "validation_accessed",
    "final_accessed",
    "formal_accessed",
)
_REQUIRED_LINEAGE_SHA_KEYS = (
    "implementation_sha256",
    "graph_sha256",
    "cache_manifest_sha256",
    "value_checkpoint_sha256",
    "hydraulic_checkpoint_sha256",
    "fit_d2_group_digest",
)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"V9 evidence requires mapping: {name}")
    return value


def _require_sha256(value: Any, name: str) -> str:
    text = str(value).lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"V9 evidence has invalid sha256 for {name}")
    return text


def _validate_training_history(
    history: Mapping[str, Any], *, expected_epochs: int
) -> None:
    for level in _LEVELS:
        rows = history.get(level)
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise ValueError(f"V9 evidence lacks training history for {level}")
        if len(rows) != expected_epochs:
            raise ValueError(
                f"V9 evidence {level} epochs={len(rows)} != canonical {expected_epochs}"
            )
        epochs: list[int] = []
        for row in rows:
            if not isinstance(row, Mapping) or "epoch" not in row:
                raise ValueError(f"V9 evidence has malformed history row for {level}")
            epochs.append(int(row["epoch"]))
        if epochs != list(range(1, expected_epochs + 1)):
            raise ValueError(f"V9 evidence has non-canonical epoch sequence for {level}")


def validate_state_sufficiency_evidence_v90(
    payload: Mapping[str, Any],
    *,
    expected_git_head: str | None = None,
) -> dict[str, Any]:
    """Validate that a V9 A/B/C report is canonical for the current code lineage.

    This is deliberately stricter than merely parsing JSON.  In particular, reports
    produced before a branch merge, reports without a git SHA, or reports generated
    with a non-canonical D2 diagnostic schedule are rejected rather than interpreted.
    """
    report = _require_mapping(payload, "payload")
    if str(report.get("contract")) != V90_CONTRACT:
        raise ValueError(
            f"V9 evidence contract {report.get('contract')!r} != {V90_CONTRACT!r}"
        )
    if report.get("development_only") is not True:
        raise ValueError("V9 evidence must be explicitly development-only")
    if report.get("production_compatible") is not False:
        raise ValueError("V9 diagnostic evidence must be production-incompatible")
    if report.get("oracle_level_c_forbidden_online") is not True:
        raise ValueError("V9 evidence must keep oracle Level C forbidden online")
    for key in _REQUIRED_BOUNDARY_FALSE:
        if report.get(key) is not False:
            raise ValueError(f"V9 evidence boundary violation or missing flag: {key}")

    lineage = _require_mapping(report.get("lineage"), "lineage")
    git_head = str(lineage.get("git_head", "")).lower()
    if not _GIT_SHA_RE.fullmatch(git_head):
        raise ValueError("V9 evidence lacks an exact 40-character git_head")
    if expected_git_head is not None:
        expected = str(expected_git_head).lower()
        if not _GIT_SHA_RE.fullmatch(expected):
            raise ValueError("expected_git_head must be a 40-character commit SHA")
        if git_head != expected:
            raise ValueError(
                f"stale V9 evidence: report git_head={git_head}, expected={expected}"
            )

    for key in _REQUIRED_LINEAGE_SHA_KEYS:
        _require_sha256(lineage.get(key), f"lineage.{key}")
    if int(lineage.get("fit_d2_group_count", 0)) <= 0:
        raise ValueError("V9 evidence has no TrainFit D2 groups")
    if int(lineage.get("seed", -1)) != 42:
        raise ValueError("V9 evidence seed must remain frozen at 42")

    preflight = _require_mapping(report.get("preflight"), "preflight")
    ladder = _require_mapping(report.get("ladder"), "ladder")
    history = _require_mapping(report.get("training_history"), "training_history")
    for level in _LEVELS:
        if level not in preflight or level not in ladder:
            raise ValueError(f"V9 evidence is missing A/B/C level: {level}")
        checks = _require_mapping(preflight[level], f"preflight.{level}")
        if checks.get("signed_state_exact_zero") is not True:
            raise ValueError(f"V9 {level} failed signed-state exact-zero preflight")
        if checks.get("signed_flow_exact_zero") is not True:
            raise ValueError(f"V9 {level} failed signed-flow exact-zero preflight")
        if checks.get("reference_frozen") is not True:
            raise ValueError(f"V9 {level} modified the frozen V7 Hydraulic reference")

    diagnostic_epochs = DirectHydraulicEffectLossContractV90().d2_pretrain_epochs
    _validate_training_history(history, expected_epochs=diagnostic_epochs)
    schedule = _require_mapping(report.get("diagnostic_schedule"), "diagnostic_schedule")
    if str(schedule.get("source")) != "TRAINFIT_D2_ONLY":
        raise ValueError("V9 evidence diagnostic_schedule.source must be TRAINFIT_D2_ONLY")
    if int(schedule.get("epochs_per_level", -1)) != diagnostic_epochs:
        raise ValueError(
            "V9 evidence diagnostic_schedule.epochs_per_level must equal canonical "
            f"{diagnostic_epochs}"
        )

    decision = _require_mapping(report.get("decision"), "decision")
    if not str(decision.get("decision", "")).strip():
        raise ValueError("V9 evidence lacks a scientific decision")

    return {
        "contract": V90_EVIDENCE_GUARD_CONTRACT,
        "accepted": True,
        "git_head": git_head,
        "fit_d2_group_count": int(lineage["fit_d2_group_count"]),
        "seed": int(lineage["seed"]),
        "diagnostic_d2_epochs": int(diagnostic_epochs),
        "decision": str(decision["decision"]),
        "scope": "development_train_only_no_swmm",
    }


__all__ = [
    "V90_EVIDENCE_GUARD_CONTRACT",
    "validate_state_sufficiency_evidence_v90",
]
