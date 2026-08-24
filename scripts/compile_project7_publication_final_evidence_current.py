"""Compile the final publication-facing Project7 evidence package.

This is an evidence compiler, not a trainer and not a SWMM runner.  It binds the frozen controller
contract, Step2 lineage/limitation, Formal Validation6, Policy Lock, Final6 comparison and event-level
statistics into one fail-closed paper-reporting artifact.  Comparator claims are classified from the
locked Final6 bootstrap intervals; no universal-superiority claim is manufactured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rtc.project7_publication_final import (
    PUBLICATION_FINAL_CONTRACT,
    validate_publication_controller_contract,
    validate_publication_policy_lock,
    validate_publication_validation,
)


FINAL_COMPARISON_CONTRACT = "PROJECT7_V23_POLICY_LOCKED_FINAL6_COMPARISON_V1"
PUBLICATION_STATISTICS_CONTRACT = "PROJECT7_V23_FINAL6_PUBLICATION_STATISTICS_V1"
STEP2_LINEAGE_CONTRACT = "PROJECT7_V23_FROZEN_STEP2_V5_LINEAGE_EVIDENCE_V1"
STEP2_DIAGNOSTIC_CONTRACT = "PROJECT7_V23_STEP2_V5_COMPONENT_DIAGNOSTIC_V1"
FINAL_PACKAGE_CONTRACT = "PROJECT7_PUBLICATION_FINAL_EVIDENCE_PACKAGE_V1"
BASELINES = ("no_control", "internal_rtc", "auto_rbc", "efd")


def _json(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {source}")
    return payload


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _claim_status(row: dict[str, Any]) -> str:
    ci = row.get("event_balanced_mean_tfv_reduction_pct_bootstrap95_ci")
    if not isinstance(ci, list) or len(ci) != 2:
        raise ValueError("publication statistics lack a two-sided TFV bootstrap interval")
    lower, upper = float(ci[0]), float(ci[1])
    mean = float(row["event_balanced_mean_tfv_reduction_pct"])
    if lower > 0.0 and mean > 0.0:
        return "SUPPORTED_POSITIVE_TFV_REDUCTION"
    if upper < 0.0 and mean < 0.0:
        return "SUPPORTED_COMPARATOR_OUTPERFORMS_PROPOSED"
    if mean > 0.0:
        return "POSITIVE_DIRECTION_UNCERTAIN"
    if mean < 0.0:
        return "NEGATIVE_DIRECTION_UNCERTAIN"
    return "NO_DIRECTION"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller-contract", required=True)
    parser.add_argument("--step2-lineage", required=True)
    parser.add_argument("--step2-diagnostic", required=True)
    parser.add_argument("--validation-evidence", required=True)
    parser.add_argument("--policy-lock", required=True)
    parser.add_argument("--final-comparison", required=True)
    parser.add_argument("--publication-statistics", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = {
        "controller_contract": Path(args.controller_contract).resolve(),
        "step2_lineage": Path(args.step2_lineage).resolve(),
        "step2_diagnostic": Path(args.step2_diagnostic).resolve(),
        "validation_evidence": Path(args.validation_evidence).resolve(),
        "policy_lock": Path(args.policy_lock).resolve(),
        "final_comparison": Path(args.final_comparison).resolve(),
        "publication_statistics": Path(args.publication_statistics).resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    controller = _json(paths["controller_contract"])
    lineage = _json(paths["step2_lineage"])
    diagnostic = _json(paths["step2_diagnostic"])
    validation = _json(paths["validation_evidence"])
    lock = _json(paths["policy_lock"])
    final = _json(paths["final_comparison"])
    statistics = _json(paths["publication_statistics"])

    validate_publication_controller_contract(controller)
    validate_publication_validation(validation)
    validate_publication_policy_lock(lock)

    if lineage.get("contract") != STEP2_LINEAGE_CONTRACT or lineage.get("lineage_pass") is not True:
        raise RuntimeError("publication package lacks passing V23/V5 Step2 lineage evidence")
    if diagnostic.get("contract") != STEP2_DIAGNOSTIC_CONTRACT:
        raise ValueError("wrong V23/V5 Step2 component diagnostic contract")
    if diagnostic.get("standalone_acceptance_pass") is not False:
        raise RuntimeError("publication package must retain the observed Step2 standalone failure")
    if diagnostic.get("step2_retrained") is not False:
        raise RuntimeError("publication Step2 was retrained after the fixed-policy decision")

    if final.get("contract") != FINAL_COMPARISON_CONTRACT:
        raise ValueError("wrong locked Final6 comparison contract")
    if final.get("formal_evidence") is not True or final.get("policy_locked") is not True:
        raise RuntimeError("Final6 is not Policy-Locked Formal evidence")
    if int(final.get("event_count", 0)) != 6:
        raise RuntimeError("publication Final requires exactly six locked events")
    if final.get("pfv_safety_all_events_pass") is not True:
        raise RuntimeError("Final6 failed the frozen Priority8 PFV safety contract")
    if final.get("engineering_all_events_pass") is not True:
        raise RuntimeError("Final6 failed engineering execution")
    if final.get("final_results_used_for_training") is not False:
        raise RuntimeError("Final6 was used for training")
    if final.get("final_results_used_for_tuning") is not False:
        raise RuntimeError("Final6 was used for tuning")

    if statistics.get("contract") != PUBLICATION_STATISTICS_CONTRACT:
        raise ValueError("wrong publication statistics contract")
    if statistics.get("statistical_unit") != "RAINFALL_EVENT":
        raise RuntimeError("publication statistics changed the independent statistical unit")
    if int(statistics.get("event_count", 0)) != 6:
        raise RuntimeError("publication statistics do not match Final6")
    if _sha(paths["final_comparison"]).lower() != str(
        statistics.get("source_final_comparison_sha256", "")
    ).lower():
        raise RuntimeError("publication statistics are not derived from the supplied Final6 evidence")

    claims: dict[str, Any] = {}
    comparison_rows = statistics.get("comparisons")
    if not isinstance(comparison_rows, dict):
        raise ValueError("publication statistics lack comparator rows")
    for baseline in BASELINES:
        row = comparison_rows.get(baseline)
        if not isinstance(row, dict):
            raise ValueError(f"publication statistics lack comparator: {baseline}")
        claims[baseline] = {
            "claim_status": _claim_status(row),
            "wins": int(row["wins"]),
            "losses": int(row["losses"]),
            "ties": int(row["ties"]),
            "event_balanced_mean_tfv_reduction_pct": float(
                row["event_balanced_mean_tfv_reduction_pct"]
            ),
            "bootstrap95_ci": list(
                row["event_balanced_mean_tfv_reduction_pct_bootstrap95_ci"]
            ),
            "aggregate_volume_tfv_reduction_pct": float(
                row["aggregate_volume_tfv_reduction_pct"]
            ),
            "exact_two_sided_sign_test_pvalue": row.get("exact_two_sided_sign_test_pvalue"),
        }

    payload = {
        "contract": FINAL_PACKAGE_CONTRACT,
        "controller_contract": PUBLICATION_FINAL_CONTRACT,
        "research_goal": controller["research_goal"],
        "step1": controller["step1"],
        "step2": {
            **controller["step2"],
            "lineage_evidence_sha256": _sha(paths["step2_lineage"]),
            "component_diagnostic_sha256": _sha(paths["step2_diagnostic"]),
            "observed_standalone_exact_tfv_rank_correlation": diagnostic.get(
                "tfv_exact_truth_rank_correlation"
            ),
        },
        "step3": controller["step3"],
        "engineering": controller["engineering"],
        "formal": {
            **controller["formal"],
            "validation_sha256": _sha(paths["validation_evidence"]),
            "policy_lock_sha256": _sha(paths["policy_lock"]),
            "final_comparison_sha256": _sha(paths["final_comparison"]),
            "publication_statistics_sha256": _sha(paths["publication_statistics"]),
            "pfv_safety_all_events_pass": True,
            "engineering_all_events_pass": True,
            "ready_for_paper_reporting": True,
        },
        "comparator_claims": claims,
        "claim_boundaries": controller["claim_boundaries"],
        "paper_interpretation": {
            "controller_type": "FINITE_CANDIDATE_RECEDING_HORIZON_LEARNING_ASSISTED_RTC",
            "primary_claim": "END_TO_END_POLICY_LOCKED_AUTHORITATIVE_SWMM_CONTROL_PERFORMANCE",
            "step2_limitation_must_be_reported": True,
            "priority8_safety_is_noninferiority_not_minimization": True,
            "global_peak_is_report_only": True,
            "universal_baseline_superiority_claim_allowed": False,
            "continuous_gradient_mpc_claim_allowed": False,
        },
        "source_paths": {key: str(path) for key, path in paths.items()},
        "source_sha256": {key: _sha(path) for key, path in paths.items()},
    }

    destination = Path(args.out).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
