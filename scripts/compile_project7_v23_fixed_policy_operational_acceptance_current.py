"""Compile publication-grade fixed-policy operational acceptance for Project7 V23.

This command keeps MODEL_ACCEPTANCE_CONTRACT_V4 unchanged.  It records the legacy Step2 component
result exactly as observed, but now requires that result to be bound to the same frozen Direct-TFV V5
Step2 checkpoint used by V23, V15 and V21.  Step1 remains a structural hard gate.  A failed standalone
Step2 ranking diagnostic is retained as a paper limitation, not relabelled as a pass and not used as a
Policy-Lock gate for the already-frozen end-to-end controller.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from rtc.direct_tfv_policy_return import sha256_file
from rtc.project7_v23_step2_lineage import (
    V23_STEP2_CHECKPOINT_SHA256,
    validate_v23_step2_component_diagnostic,
    validate_v23_step2_lineage_evidence,
)


CONTRACT = "PROJECT7_V23_FIXED_POLICY_OPERATIONAL_ACCEPTANCE_EVIDENCE_V2_STEP2_LINEAGE_BOUND"
OPERATIONAL_CONTRACT = "PROJECT7_V23_FIXED_POLICY_OPERATIONAL_ACCEPTANCE_CONTRACT_V1"
LEGACY_MODEL_CONTRACT = "MODEL_ACCEPTANCE_CONTRACT_V4_DIMENSIONLESS_PREREGISTERED"
REQUIRED_MODE = "FIXED_POLICY_NO_RETRAIN"
REQUIRED_STEP3_DISPOSITION = "FROZEN_V15_V21_FIXED_POLICY_NO_RETRAIN"


def _json(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {source}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operational-acceptance-contract", required=True)
    parser.add_argument("--legacy-model-acceptance-contract", required=True)
    parser.add_argument("--formal-protocol", required=True)
    parser.add_argument("--step1-unobserved-depth-nse", type=float, required=True)
    parser.add_argument("--step2-lineage-evidence", required=True)
    parser.add_argument("--step2-component-diagnostic", required=True)
    parser.add_argument("--source-evidence", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    operational_path = Path(args.operational_acceptance_contract).resolve()
    legacy_path = Path(args.legacy_model_acceptance_contract).resolve()
    protocol_path = Path(args.formal_protocol).resolve()
    lineage_path = Path(args.step2_lineage_evidence).resolve()
    diagnostic_path = Path(args.step2_component_diagnostic).resolve()
    operational = _json(operational_path)
    legacy = _json(legacy_path)
    protocol = _json(protocol_path)
    lineage = _json(lineage_path)
    diagnostic = _json(diagnostic_path)

    if operational.get("contract") != OPERATIONAL_CONTRACT:
        raise ValueError("wrong fixed-policy operational acceptance contract")
    if legacy.get("contract") != LEGACY_MODEL_CONTRACT:
        raise ValueError("wrong legacy MODEL_ACCEPTANCE_CONTRACT_V4")
    if protocol.get("formal_mode") != REQUIRED_MODE:
        raise RuntimeError("operational amendment is valid only for FIXED_POLICY_NO_RETRAIN")
    if protocol.get("final_opened") is not False:
        raise RuntimeError("operational acceptance refuses a protocol after Final exposure")
    if protocol.get("validation_can_update_model_parameters") is not False:
        raise RuntimeError("fixed-policy Validation must not update model parameters")
    if protocol.get("final_can_tune_any_model_threshold_or_candidate") is not False:
        raise RuntimeError("Final tuning firewall was weakened")

    validate_v23_step2_lineage_evidence(lineage)
    validate_v23_step2_component_diagnostic(diagnostic)
    if str(diagnostic.get("step2_lineage_evidence_sha256", "")).lower() != sha256_file(
        lineage_path
    ).lower():
        raise RuntimeError("Step2 component diagnostic was compiled from another lineage audit")
    if str(diagnostic.get("step2_checkpoint_sha256", "")).lower() != V23_STEP2_CHECKPOINT_SHA256:
        raise RuntimeError("Formal Step2 diagnostic is not for the V23 V5 checkpoint")

    step1_value = float(args.step1_unobserved_depth_nse)
    step2_value = float(diagnostic["tfv_exact_truth_rank_correlation"])
    if not math.isfinite(step1_value) or not math.isfinite(step2_value):
        raise ValueError("acceptance metrics must be finite")
    step1_threshold = float(operational["step1"]["minimum"])
    legacy_step2_threshold = float(legacy["step2"]["minimum"]["tfv_exact_truth_rank_correlation"])
    step1_pass = step1_value >= step1_threshold
    legacy_step2_pass = step2_value >= legacy_step2_threshold

    evidence_files = [Path(value).resolve() for value in args.source_evidence]
    for path in evidence_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    paper_claim_restrictions: list[str] = []
    if not legacy_step2_pass:
        paper_claim_restrictions.extend(
            [
                "DO_NOT_CLAIM_STEP2_STANDALONE_TFV_RANKING_ACCEPTANCE",
                "DO_NOT_CLAIM_STEP2_AS_AN_INDEPENDENT_HIGH_ACCURACY_TFV_SURROGATE",
                "REPORT_LEGACY_STEP2_V4_FAILURE_AND_V5_BOUND_NUMERIC_DIAGNOSTICS",
                "LIMIT_PRIMARY_EFFICACY_CLAIM_TO_POLICY_LOCKED_END_TO_END_CLOSED_LOOP_SWMM",
            ]
        )

    payload = {
        "contract": CONTRACT,
        "operational_acceptance_contract": OPERATIONAL_CONTRACT,
        "operational_acceptance_contract_sha256": sha256_file(operational_path),
        "legacy_model_acceptance_contract": LEGACY_MODEL_CONTRACT,
        "legacy_model_acceptance_contract_sha256": sha256_file(legacy_path),
        "formal_protocol_sha256": sha256_file(protocol_path),
        "formal_mode": REQUIRED_MODE,
        "research_estimand": "CLOSED_LOOP_SYSTEM_WIDE_TFV_UNDER_AUTHORITATIVE_SWMM",
        "step1_unobserved_depth_nse": step1_value,
        "step1_threshold": step1_threshold,
        "step1_accepted": step1_pass,
        "step2_runtime_lineage_accepted": True,
        "step2_checkpoint_sha256": V23_STEP2_CHECKPOINT_SHA256,
        "step2_checkpoint_path": str(lineage["step2_checkpoint_path"]),
        "step2_lineage_evidence_path": str(lineage_path),
        "step2_lineage_evidence_sha256": sha256_file(lineage_path),
        "step2_component_diagnostic_path": str(diagnostic_path),
        "step2_component_diagnostic_sha256": sha256_file(diagnostic_path),
        "step2_tfv_exact_truth_rank_correlation": step2_value,
        "step2_legacy_threshold": legacy_step2_threshold,
        "step2_legacy_component_accepted": legacy_step2_pass,
        "step2_required_for_policy_lock": False,
        "step2_failure_retained_as_publication_limitation": bool(not legacy_step2_pass),
        "step2_standalone_surrogate_claim_allowed": bool(legacy_step2_pass),
        "step2_query_balanced_top1": diagnostic.get("query_balanced_top1"),
        "step2_mean_selected_regret_m3": diagnostic.get("mean_selected_regret_m3"),
        "step2_retrained": False,
        "step3_retrained": False,
        "step3_disposition": REQUIRED_STEP3_DISPOSITION,
        "policy_parameters_changed_after_component_diagnostic_failure": False,
        "hard_thresholds_lowered_after_results": False,
        "legacy_step2_threshold_relabelled_or_lowered": False,
        "protocol_amendment_after_legacy_component_failure": True,
        "validation_required_before_policy_lock": True,
        "validation_tfv_is_primary_effect_estimate_not_tuning_gate": True,
        "validation_hard_gates": [
            "PFV_SAFETY_ALL_EVENTS",
            "ENGINEERING_EXECUTION_ALL_EVENTS",
            "FINAL_REMAINS_SEALED",
            "NO_POLICY_MUTATION",
        ],
        "paper_claim_restrictions": paper_claim_restrictions,
        "source_evidence_paths": [str(path) for path in evidence_files],
        "source_evidence_sha256": [sha256_file(path) for path in evidence_files],
        "final_truth_opened": False,
        "accepted_for_operational_validation": bool(step1_pass),
        "accepted_for_policy_lock_candidate": bool(step1_pass),
    }

    destination = Path(args.out).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["accepted_for_operational_validation"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
