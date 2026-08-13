"""Fail-closed data/evidence contract for Project7 V120.

The already-generated source census and the split-eligible training cache are two
different populations.  This module makes both explicit and reusable by the
trainer, promotion, runtime loader and Policy Lock.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

D2_SOURCE_AUDIT_CONTRACT = "PROJECT7_V120_D2_SOURCE_CENSUS_AUDIT_V1"
SOURCE_D2_BRANCHES = 4800
TRAIN_D2_BRANCHES = 3600
TARGETED_D3_BRANCHES = 3600
D2_GROUPS = 144
D3_GROUPS = 144
SOURCE_D2_GROUPS = 192
DEVELOPMENT_VALIDATION_D2_GROUPS = 48
BRANCHES_PER_GROUP = 25
CANDIDATES_PER_GROUP = 24
INTERNAL_HOLDOUT_FRACTION = 0.20
STATE_DOMAIN_CONTRACT = "PROJECT7_V120_ORACLE_TRAIN_STEP1_RUNTIME_STATE_DOMAIN_V1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    if len(text) != 64:
        return False
    try:
        int(text, 16)
    except ValueError:
        return False
    return True


def verify_d2_source_audit(
    path: str | Path,
    *,
    split_contract_path: str | Path,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("contract") != D2_SOURCE_AUDIT_CONTRACT:
        raise ValueError("V120 requires D2 source-census audit V1")
    expected = {
        "unique_authoritative_branches": SOURCE_D2_BRANCHES,
        "source_group_count": SOURCE_D2_GROUPS,
        "branches_per_group": BRANCHES_PER_GROUP,
        "frozen_train_eligible_branches": TRAIN_D2_BRANCHES,
        "frozen_train_group_count": D2_GROUPS,
        "development_validation_branches": SOURCE_D2_BRANCHES - TRAIN_D2_BRANCHES,
        "development_validation_group_count": DEVELOPMENT_VALIDATION_D2_GROUPS,
        "final_branches": 0,
        "frozen_train_event_count": 18,
        "development_validation_event_count": 6,
    }
    for key, value in expected.items():
        if int(payload.get(key, -1)) != value:
            raise ValueError(f"D2 source audit {key}={payload.get(key)!r}, expected {value}")
    if not _is_sha256(payload.get("source_index_sha256")):
        raise ValueError("D2 source audit lacks a valid source-index SHA256")
    if payload.get("validation_branches_excluded_from_training") is not True:
        raise ValueError("D2 source audit does not prove Validation exclusion")
    if payload.get("final_untouched") is not True:
        raise ValueError("D2 source audit does not prove Final untouched")
    if str(payload.get("split_contract_sha256", "")) != sha256_file(split_contract_path):
        raise ValueError("D2 source audit belongs to another scientific split")
    return payload


def validate_canonical_cache_population(
    cache: Any,
    d2_names: Sequence[str],
    d3_names: Sequence[str],
) -> dict[str, Any]:
    d2 = list(d2_names)
    d3 = list(d3_names)
    if (len(d2), len(d3)) != (D2_GROUPS, D3_GROUPS):
        raise ValueError(
            f"V120 canonical cache requires {D2_GROUPS}/{D3_GROUPS} D2/D3 groups; "
            f"got {len(d2)}/{len(d3)}"
        )

    def inspect(names: list[str], label: str) -> tuple[int, int]:
        branches = 0
        candidates = 0
        for name in names:
            entry = cache.entry(name)
            count = len(entry.indices)
            candidate_count = count - 1
            if count != BRANCHES_PER_GROUP or candidate_count != CANDIDATES_PER_GROUP:
                raise ValueError(
                    f"{label} group {name} requires 1 reference + {CANDIDATES_PER_GROUP} "
                    f"candidates; got {count} branches"
                )
            arrays = entry.arrays
            if "exact_node_flood_volume_m3" not in arrays:
                raise ValueError(f"{label} group {name} lacks authoritative exact flood volume")
            ref = int(entry.reference_index)
            truth = np.asarray(arrays["exact_node_flood_volume_m3"], dtype=np.float64)
            if truth.shape[0] <= ref or not np.isfinite(truth[list(entry.indices)]).all():
                raise ValueError(f"{label} group {name} has invalid authoritative TFV truth")
            branches += count
            candidates += candidate_count
        return branches, candidates

    d2_branches, d2_candidates = inspect(d2, "D2")
    d3_branches, d3_candidates = inspect(d3, "D3")
    if d2_branches != TRAIN_D2_BRANCHES:
        raise ValueError(f"V120 requires Train18 D2={TRAIN_D2_BRANCHES} branches; got {d2_branches}")
    if d3_branches != TARGETED_D3_BRANCHES:
        raise ValueError(f"V120 requires targeted D3={TARGETED_D3_BRANCHES} branches; got {d3_branches}")
    return {
        "d2_groups": len(d2),
        "d2_branches": d2_branches,
        "d2_candidates": d2_candidates,
        "d3_groups": len(d3),
        "d3_branches": d3_branches,
        "d3_candidates": d3_candidates,
        "branches_per_group": BRANCHES_PER_GROUP,
        "candidates_per_group": CANDIDATES_PER_GROUP,
    }


def validate_internal_holdout_fraction(value: float) -> float:
    result = float(value)
    if not np.isclose(result, INTERNAL_HOLDOUT_FRACTION, rtol=0.0, atol=1.0e-12):
        raise ValueError("V120 internal holdout fraction is frozen at 0.20")
    return result


def finite_auxiliary_value_metrics(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    """Fail only on undefined/corrupt D2 diagnostics; do not invent a D2 science threshold."""
    required = (
        "rank",
        "pairwise",
        "sign_accuracy",
        "top1_rate",
        "spread_ratio",
        "response_ratio",
        "mean_regret_m3",
    )
    reasons = [key for key in required if not np.isfinite(float(metrics.get(key, np.nan)))]
    return not reasons, [f"non-finite D2 metric: {key}" for key in reasons]


__all__ = [
    "BRANCHES_PER_GROUP",
    "CANDIDATES_PER_GROUP",
    "D2_GROUPS",
    "D2_SOURCE_AUDIT_CONTRACT",
    "D3_GROUPS",
    "DEVELOPMENT_VALIDATION_D2_GROUPS",
    "INTERNAL_HOLDOUT_FRACTION",
    "SOURCE_D2_BRANCHES",
    "SOURCE_D2_GROUPS",
    "STATE_DOMAIN_CONTRACT",
    "TARGETED_D3_BRANCHES",
    "TRAIN_D2_BRANCHES",
    "finite_auxiliary_value_metrics",
    "sha256_file",
    "validate_canonical_cache_population",
    "validate_internal_holdout_fraction",
    "verify_d2_source_audit",
]
