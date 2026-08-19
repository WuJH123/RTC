"""Calibration helpers for the current masked hybrid H10 policy-return candidate portfolio."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ACTION_ENCODING,
    derive_policy_return_admission,
    validate_policy_return_record,
)
from .direct_tfv_policy_return_hybrid_portfolio import (
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT,
    PROJECTED_GRADIENT_SOURCE,
)


DIRECT_TFV_POLICY_RETURN_PORTFOLIO_ADMISSION_CONTRACT = (
    "PROJECT7_DIRECT_TFV_POLICY_RETURN_PORTFOLIO_MATCHED_ADMISSION_V4_82CONTROL_109REP"
)


def validate_policy_return_portfolio_record(record: Mapping[str, Any]) -> None:
    validate_policy_return_record(record)
    if str(record.get("candidate_portfolio_contract", "")) != DIRECT_TFV_POLICY_RETURN_PORTFOLIO_CONTRACT:
        raise ValueError("policy-return portfolio record has the wrong masked hybrid candidate contract")
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
    query_set = str(record.get("query_set_id", "")).strip().lower()
    if not source:
        raise ValueError("policy-return portfolio record lacks candidate_source")
    if len(query_set) != 64 or any(ch not in "0123456789abcdef" for ch in query_set):
        raise ValueError("policy-return portfolio record lacks a canonical query_set_id")


def derive_policy_return_portfolio_admission(
    *,
    records: Sequence[Mapping[str, Any]],
    expected_rainfall_groups: Sequence[str],
    policy_return_checkpoint_sha256: str,
    continuation_policy_sha256: str,
    coverage: float,
) -> dict[str, Any]:
    """Calibrate one-sided residuals on the same masked hybrid H10 family used online."""
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
    if PROJECTED_GRADIENT_SOURCE not in sources:
        raise ValueError("portfolio calibration lacks the masked support-constrained H10 gradient family")

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
                "support_constrained_gradient_h10": True,
            },
            "ranking_calibration_scope": (
                "same-prefix H10 hybrid candidate portfolio on the frozen native supervisory-control "
                "subspace after first-move support projection, q95 joint-sequence contraction and "
                "deduplication; rainfall-group max residual is the independent split-conformal unit"
            ),
            "gradient_free_dimension": 82,
            "gradient_tensor_channels": 109,
            "gradient_action_horizon": "H10_ONLY",
            "online_lbfgsb_used": False,
        }
    )
    return payload


__all__ = [
    "DIRECT_TFV_POLICY_RETURN_PORTFOLIO_ADMISSION_CONTRACT",
    "derive_policy_return_portfolio_admission",
    "validate_policy_return_portfolio_record",
]
