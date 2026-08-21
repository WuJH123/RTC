"""Calibration helpers for the current masked three-family H10 policy-return portfolio."""
from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping, Sequence

from .direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
    derive_policy_return_admission,
    validate_policy_return_record,
)
from .direct_tfv_policy_return_hybrid_portfolio import DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT


DIRECT_TFV_POLICY_RETURN_PORTFOLIO_ADMISSION_CONTRACT = (
    "PROJECT7_DIRECT_TFV_POLICY_RETURN_PORTFOLIO_MATCHED_ADMISSION_V5_THREE_FAMILY_82CONTROL_109REP"
)
CURRENT_THREE_FAMILY_SOURCES = (
    "STEP2_H10_PROBE_SCALE_0.50",
    "STEP2_H10_PROBE_SCALE_1.00",
    "TYPE_AWARE_HYDRAULIC_PRESSURE",
)
CURRENT_POLICY_RETURN_LEARNING_ROLES = (
    "policy_return_train",
    "policy_return_validation",
    "policy_return_calibration",
)
_LEARNING_TRUTH_GATES = (
    "shared_hold_branch",
    "same_prefix_verified",
    "same_continuation_policy_verified",
    "passive_setting_channels_unchanged",
    "target_write_readback_verified",
    "engineering_bounds_verified",
    "candidate_manifest_support_lineage_verified",
)
_LEARNING_PROVENANCE_HASHES = (
    "candidate_manifest_sha256",
    "parent_decisions_sha256",
    "source_inp_sha256",
    "asset_manifest_sha256",
    "graph_sha256",
    "base_step2_sha256",
    "sequence_support_sha256",
    "supervisory_control_sha256",
)


def _canonical_sha256(value: Any) -> bool:
    raw = str(value or "").strip().lower()
    return len(raw) == 64 and all(ch in "0123456789abcdef" for ch in raw)


def validate_policy_return_portfolio_record(record: Mapping[str, Any]) -> None:
    """Validate one current three-family portfolio row.

    Exact-query provenance/readback gates are enforced by the authoritative query runner before a
    record is emitted. This reusable validator keeps the stable dataset/calibration schema while
    fail-closing candidate-family drift.
    """
    role = str(record.get("data_role", ""))
    if role == "policy_return_development_diagnostic":
        probe = dict(record)
        probe["data_role"] = "policy_return_train"
        validate_policy_return_record(probe)
        if (
            record.get("development_diagnostic_only") is not True
            or record.get("eligible_for_learning_dataset") is not False
        ):
            raise ValueError(
                "Development mechanism truth must remain explicitly ineligible for learning"
            )
    else:
        validate_policy_return_record(record)
    if str(record.get("candidate_portfolio_contract", "")) != DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT:
        raise ValueError("policy-return portfolio record has the wrong current three-family contract")
    if str(record.get("action_encoding_contract", "")) != DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING:
        raise ValueError("policy-return portfolio record has the wrong H10 action encoding")
    if int(record.get("supervisory_control_dimension", -1)) != 82:
        raise ValueError("policy-return portfolio record has the wrong supervisory-control dimension")
    if int(record.get("model_action_channel_count", -1)) != 109:
        raise ValueError("policy-return portfolio record lost the 109-channel model representation")
    mask_sha = str(record.get("supervisory_mask_sha256", "")).strip().lower()
    if len(mask_sha) != 64 or any(ch not in "0123456789abcdef" for ch in mask_sha):
        raise ValueError("policy-return portfolio record lacks a canonical supervisory-mask SHA")
    if record.get("passive_setting_channels_unchanged") is not True:
        raise ValueError("policy-return portfolio record changed passive setting channels")
    source = str(record.get("candidate_source", "")).strip()
    if source == "SUPPORT_CONSTRAINED_GRADIENT_H10":
        raise ValueError("current three-family calibration must not contain projected-gradient rows")
    if source not in CURRENT_THREE_FAMILY_SOURCES:
        raise ValueError(f"policy-return portfolio record has unexpected candidate family: {source}")
    query_set = str(record.get("query_set_id", "")).strip().lower()
    if len(query_set) != 64 or any(ch not in "0123456789abcdef" for ch in query_set):
        raise ValueError("policy-return portfolio record lacks a canonical query_set_id")
    if record.get("projected_gradient_online") not in (False, None):
        raise ValueError("current policy-return portfolio record unexpectedly enables projected gradient")
    if record.get("online_lbfgsb_used") not in (False, None):
        raise ValueError("current policy-return portfolio record unexpectedly uses L-BFGS-B")


def validate_policy_return_learning_record(record: Mapping[str, Any]) -> None:
    """Fail closed before exact truth can enter train/validation/calibration datasets.

    Development mechanism rows are intentionally valid portfolio records, but they must never cross
    the learning boundary. Learning rows additionally require the exact-query truth firewall emitted
    by the current authoritative query runner: shared HOLD, same prefix/continuation, readback,
    engineering/support lineage, current finite three-family execution and canonical provenance SHAs.
    """
    validate_policy_return_portfolio_record(record)
    role = str(record.get("data_role", ""))
    if role not in CURRENT_POLICY_RETURN_LEARNING_ROLES:
        raise ValueError("policy-return learning firewall accepts only train/validation/calibration rows")
    if record.get("development_diagnostic_only") is not False:
        raise ValueError("policy-return learning row is marked as Development diagnostic truth")
    if record.get("eligible_for_learning_dataset") is not True:
        raise ValueError("policy-return record is not explicitly eligible for learning")
    for key in _LEARNING_TRUTH_GATES:
        if record.get(key) is not True:
            raise ValueError(f"policy-return learning truth failed authoritative gate: {key}")
    if record.get("projected_gradient_online") is not False:
        raise ValueError("policy-return learning truth must explicitly keep projected gradient offline")
    if record.get("online_lbfgsb_used") is not False:
        raise ValueError("policy-return learning truth must explicitly keep L-BFGS-B offline")
    for key in _LEARNING_PROVENANCE_HASHES:
        if not _canonical_sha256(record.get(key)):
            raise ValueError(f"policy-return learning truth lacks canonical provenance {key}")
    for key in ("candidate_flow_routing_error_pct", "hold_flow_routing_error_pct"):
        try:
            value = float(record.get(key, float("nan")))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"policy-return learning truth lacks finite {key}") from exc
        if not math.isfinite(value):
            raise ValueError(f"policy-return learning truth lacks finite {key}")


def derive_policy_return_portfolio_admission(
    *,
    records: Sequence[Mapping[str, Any]],
    expected_rainfall_groups: Sequence[str],
    policy_return_checkpoint_sha256: str,
    continuation_policy_sha256: str,
    coverage: float,
) -> dict[str, Any]:
    """Calibrate one-sided residuals on the same three-family H10 set used online."""
    if not records:
        raise ValueError("portfolio admission received no calibration records")
    query_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    mask_shas: set[str] = set()
    for row in records:
        validate_policy_return_portfolio_record(row)
        query_counts[str(row["query_set_id"])] += 1
        source_counts[str(row["candidate_source"])] += 1
        mask_shas.add(str(row["supervisory_mask_sha256"]).lower())
    if len(mask_shas) != 1:
        raise ValueError("portfolio admission mixes supervisory-control mask lineages")
    multi = sum(count >= 2 for count in query_counts.values())
    if multi <= 0:
        raise ValueError("portfolio admission requires same-prefix multi-candidate query sets")
    sources = set(source_counts)
    if "TYPE_AWARE_HYDRAULIC_PRESSURE" not in sources:
        raise ValueError("portfolio calibration lacks the type-aware hydraulic-pressure family")
    if not any(source.startswith("STEP2_H10_PROBE_SCALE_") for source in sources):
        raise ValueError("portfolio calibration lacks the supported Step2 H10-probe family")
    if not sources.issubset(set(CURRENT_THREE_FAMILY_SOURCES)):
        raise ValueError("current three-family calibration contains a non-current candidate family")

    payload = derive_policy_return_admission(
        records=records,
        expected_rainfall_groups=expected_rainfall_groups,
        policy_return_checkpoint_sha256=policy_return_checkpoint_sha256,
        continuation_policy_sha256=continuation_policy_sha256,
        coverage=coverage,
    )
    payload.update(
        {
            "portfolio_admission_contract": DIRECT_TFV_POLICY_RETURN_PORTFOLIO_ADMISSION_CONTRACT,
            "candidate_portfolio_contract": DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
            "action_encoding_contract": DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
            "supervisory_control_dimension": 82,
            "model_action_channel_count": 109,
            "supervisory_mask_sha256": next(iter(mask_shas)),
            "passive_setting_channels_unchanged": True,
            "query_set_count": len(query_counts),
            "multi_candidate_query_set_count": int(multi),
            "candidate_source_counts": dict(sorted(source_counts.items())),
            "required_candidate_families_present": {
                "step2_h10_probe_direction": True,
                "type_aware_hydraulic_pressure": True,
            },
            "candidate_family_contract": list(CURRENT_THREE_FAMILY_SOURCES),
            "projected_gradient_online": False,
            "projected_gradient_calibration_required": False,
            "ranking_calibration_scope": (
                "same-prefix H10 three-family candidate portfolio on the frozen native supervisory "
                "subspace after first-move support projection, q95 joint-sequence contraction and "
                "deduplication; rainfall-group max residual is the independent split-conformal unit"
            ),
            "online_lbfgsb_used": False,
        }
    )
    return payload


__all__ = [
    "CURRENT_POLICY_RETURN_LEARNING_ROLES",
    "CURRENT_THREE_FAMILY_SOURCES",
    "DIRECT_TFV_POLICY_RETURN_PORTFOLIO_ADMISSION_CONTRACT",
    "derive_policy_return_portfolio_admission",
    "validate_policy_return_learning_record",
    "validate_policy_return_portfolio_record",
]
