"""Create the immutable Project7 V23 Policy Lock after Formal validation.

The fixed-policy path permits a failed legacy Step2 standalone diagnostic only when the evidence is
explicitly bound to the exact V5 Step2 checkpoint used by V23/V15/V21.  This tool never trains a model
or runs SWMM; Final remains sealed at lock creation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from rtc.direct_tfv_policy_return import sha256_file
from rtc.project7_v23_formal_reuse import V23_FORMAL_PROTOCOL_CONTRACT, validate_frozen_split
from rtc.project7_v23_step2_lineage import V23_STEP2_CHECKPOINT_SHA256


POLICY_LOCK_CONTRACT = "PROJECT7_V23_POLICY_LOCK_V1"
MODEL_ACCEPTANCE_EVIDENCE_CONTRACT = "PROJECT7_V23_FORMAL_MODEL_ACCEPTANCE_EVIDENCE_V1"
OPERATIONAL_ACCEPTANCE_EVIDENCE_CONTRACT = (
    "PROJECT7_V23_FIXED_POLICY_OPERATIONAL_ACCEPTANCE_EVIDENCE_V2_STEP2_LINEAGE_BOUND"
)
VALIDATION_EVIDENCE_CONTRACT = "PROJECT7_V23_FORMAL_DEVELOPMENT_VALIDATION_EVIDENCE_V1"


def _json(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {source}")
    return payload


def _git_head(repository_root: Path) -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    if len(value) != 40:
        raise RuntimeError("could not resolve repository HEAD")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repository_root,
        text=True,
    ).splitlines()
    allowed_docs = (
        "docs/EDGE_FEATURE_NORMALIZATION_V44.json",
        "docs/STEP2_EDGE_HYDRAULIC_LINEAGE_AUDIT_V44.json",
        "docs/STEP2_EDGE_HYDRAULIC_LINEAGE_AUDIT_V44.md",
    )
    policy_dirty = [line for line in status if not any(doc in line for doc in allowed_docs)]
    if policy_dirty:
        raise RuntimeError(f"policy-relevant working tree is dirty: {policy_dirty[:5]}")
    return value


def _source_tree_digest(repository_root: Path) -> str:
    paths = (
        "src/rtc/direct_tfv_policy_return_portfolio_v23.py",
        "src/rtc/direct_tfv_operational_v23_runtime.py",
        "src/rtc/direct_tfv_operational_v21_runtime_v2.py",
        "src/rtc/direct_tfv_policy_return_selected_boundary_v21.py",
        "src/rtc/direct_tfv_policy_return_query_margin_v17.py",
        "src/rtc/project7_v23_formal_reuse.py",
        "src/rtc/project7_v23_step2_lineage.py",
        "src/rtc/baselines.py",
        "src/rtc/rule_baselines.py",
        "src/rtc/production_guard.py",
        "scripts/run_policy_direct_tfv_operational_v23_development.py",
        "scripts/run_policy_direct_tfv_v23_locked_current.py",
        "scripts/run_project7_v23_formal_final_current.py",
        "scripts/audit_project7_v23_step2_lineage_current.py",
        "scripts/compile_project7_v23_step2_v5_component_diagnostic_current.py",
        "scripts/compile_project7_v23_fixed_policy_operational_acceptance_current.py",
        "configs/project7_v23_fixed_policy_operational_acceptance_contract_v1.json",
    )
    digest = hashlib.sha256()
    for relative in paths:
        path = repository_root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_acceptance(
    acceptance: dict[str, Any],
    *,
    protocol_mode: str,
) -> tuple[str, str, bool | None, bool, list[str]]:
    contract = str(acceptance.get("contract", ""))
    if acceptance.get("final_truth_opened") is not False:
        raise RuntimeError("acceptance evidence accessed Final")
    if acceptance.get("hard_thresholds_lowered_after_results") is not False:
        raise RuntimeError("acceptance thresholds were modified after results")

    if contract == MODEL_ACCEPTANCE_EVIDENCE_CONTRACT:
        if acceptance.get("accepted_for_policy_lock") is not True:
            raise RuntimeError("legacy model-acceptance evidence did not authorize Policy Lock")
        if acceptance.get("step1_accepted") is not True or acceptance.get("step2_accepted") is not True:
            raise RuntimeError("legacy path requires Step1/Step2 preregistered acceptance")
        disposition = str(acceptance.get("step3_disposition", ""))
        allowed = {
            "EXACT_MATCH_MINIMAL_RETRAIN_VALIDATED",
            "EXACT_MATCH_CURRENT_V15_V21_VALIDATED_NO_RETRAIN",
            "FROZEN_V15_V21_FIXED_POLICY_NO_RETRAIN",
        }
        if disposition not in allowed:
            raise RuntimeError(f"unsupported Step3 Formal disposition: {disposition}")
        return "LEGACY_COMPONENT_STRICT_V4", disposition, True, True, []

    if contract != OPERATIONAL_ACCEPTANCE_EVIDENCE_CONTRACT:
        raise ValueError("unsupported model/operational acceptance evidence contract")
    if protocol_mode != "FIXED_POLICY_NO_RETRAIN" or acceptance.get("formal_mode") != protocol_mode:
        raise RuntimeError("fixed-policy operational acceptance used outside FIXED_POLICY_NO_RETRAIN")
    if acceptance.get("accepted_for_policy_lock_candidate") is not True:
        raise RuntimeError("operational acceptance did not authorize validation/lock candidacy")
    if acceptance.get("step1_accepted") is not True:
        raise RuntimeError("Step1 structural acceptance remains mandatory")
    if acceptance.get("step2_runtime_lineage_accepted") is not True:
        raise RuntimeError("Policy Lock requires V23 Step2 runtime-lineage evidence")
    if str(acceptance.get("step2_checkpoint_sha256", "")).lower() != V23_STEP2_CHECKPOINT_SHA256:
        raise RuntimeError("Policy Lock refuses Step2 evidence from another checkpoint")
    if acceptance.get("step2_retrained") is not False:
        raise RuntimeError("fixed V23 Policy Lock refuses Step2 retraining")
    if acceptance.get("step2_required_for_policy_lock") is not False:
        raise RuntimeError("operational acceptance unexpectedly made legacy Step2 a lock gate")
    if acceptance.get("policy_parameters_changed_after_component_diagnostic_failure") is not False:
        raise RuntimeError("policy changed after component diagnostics")
    if acceptance.get("legacy_step2_threshold_relabelled_or_lowered") is not False:
        raise RuntimeError("legacy Step2 threshold was relabelled/lowered")
    if acceptance.get("step3_retrained") is not False:
        raise RuntimeError("fixed-policy operational acceptance forbids Step3 retraining")
    disposition = str(acceptance.get("step3_disposition", ""))
    if disposition != "FROZEN_V15_V21_FIXED_POLICY_NO_RETRAIN":
        raise RuntimeError("wrong fixed-policy Step3 disposition")
    step2_legacy_pass = acceptance.get("step2_legacy_component_accepted")
    if not isinstance(step2_legacy_pass, bool):
        raise RuntimeError("operational acceptance must report legacy Step2 PASS/FAIL explicitly")
    if not step2_legacy_pass:
        if acceptance.get("step2_failure_retained_as_publication_limitation") is not True:
            raise RuntimeError("failed legacy Step2 diagnostic was not retained as a limitation")
        if acceptance.get("step2_standalone_surrogate_claim_allowed") is not False:
            raise RuntimeError("failed legacy Step2 cannot support a standalone surrogate claim")
    restrictions = [str(value) for value in acceptance.get("paper_claim_restrictions", ())]
    return "FIXED_POLICY_END_TO_END_OPERATIONAL", disposition, step2_legacy_pass, False, restrictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-protocol", required=True)
    parser.add_argument("--formal-split-contract", required=True)
    parser.add_argument("--model-acceptance-evidence", required=True)
    parser.add_argument("--development-validation-evidence", required=True)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--v15-rank-checkpoint", required=True)
    parser.add_argument("--v21-boundary-checkpoint", required=True)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    repository_root = Path(args.repository_root).resolve()
    protocol_path = Path(args.formal_protocol).resolve()
    split_path = Path(args.formal_split_contract).resolve()
    acceptance_path = Path(args.model_acceptance_evidence).resolve()
    validation_path = Path(args.development_validation_evidence).resolve()
    asset_manifest_path = Path(args.asset_manifest).resolve()
    v15_path = Path(args.v15_rank_checkpoint).resolve()
    v21_path = Path(args.v21_boundary_checkpoint).resolve()
    for path in (
        protocol_path,
        split_path,
        acceptance_path,
        validation_path,
        asset_manifest_path,
        v15_path,
        v21_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    protocol = _json(protocol_path)
    split = _json(split_path)
    acceptance = _json(acceptance_path)
    validation = _json(validation_path)
    if protocol.get("contract") != V23_FORMAL_PROTOCOL_CONTRACT:
        raise ValueError("wrong V23 Formal protocol contract")
    if protocol.get("formal_protocol_frozen") is not True or protocol.get("final_opened") is not False:
        raise RuntimeError("Formal protocol is not frozen with Final sealed")
    if protocol.get("calibration_role_removed") is not True:
        raise RuntimeError("Policy Lock refuses a reintroduced calibration role")
    roles = validate_frozen_split(split)
    protocol_mode = str(protocol.get("formal_mode", ""))
    acceptance_basis, disposition, step2_pass, step2_gate, restrictions = _validate_acceptance(
        acceptance, protocol_mode=protocol_mode
    )

    if validation.get("contract") != VALIDATION_EVIDENCE_CONTRACT:
        raise ValueError("wrong Formal Development-Validation evidence contract")
    if set(validation.get("event_ids", ())) != set(roles["development_validation"]):
        raise RuntimeError("Formal validation did not use exactly the frozen six Validation events")
    if validation.get("final_truth_opened") is not False:
        raise RuntimeError("Formal validation accessed Final")
    if validation.get("pfv_safety_all_events_pass") is not True:
        raise RuntimeError("PFV safety failed during Formal Development-Validation")
    if validation.get("engineering_all_events_pass") is not True:
        raise RuntimeError("engineering execution failed during Formal Development-Validation")
    if validation.get("policy_changed_after_validation_started") is not False:
        raise RuntimeError("policy changed during Formal validation")

    head = _git_head(repository_root)
    lock = {
        "contract": POLICY_LOCK_CONTRACT,
        "locked": True,
        "formal_mode": protocol_mode,
        "acceptance_basis": acceptance_basis,
        "step3_disposition": disposition,
        "step2_checkpoint_sha256": acceptance.get("step2_checkpoint_sha256"),
        "step2_runtime_lineage_accepted": acceptance.get("step2_runtime_lineage_accepted"),
        "step2_legacy_component_accepted": step2_pass,
        "step2_required_for_policy_lock": step2_gate,
        "paper_claim_restrictions": restrictions,
        "repository_head_sha": head,
        "policy_source_tree_sha256": _source_tree_digest(repository_root),
        "formal_protocol_path": str(protocol_path),
        "formal_protocol_sha256": sha256_file(protocol_path),
        "formal_split_contract_path": str(split_path),
        "formal_split_contract_sha256": sha256_file(split_path),
        "model_acceptance_evidence_path": str(acceptance_path),
        "model_acceptance_evidence_sha256": sha256_file(acceptance_path),
        "development_validation_evidence_path": str(validation_path),
        "development_validation_evidence_sha256": sha256_file(validation_path),
        "asset_manifest_path": str(asset_manifest_path),
        "asset_manifest_sha256": sha256_file(asset_manifest_path),
        "v15_rank_checkpoint_path": str(v15_path),
        "v15_rank_checkpoint_sha256": sha256_file(v15_path),
        "v21_boundary_checkpoint_path": str(v21_path),
        "v21_boundary_checkpoint_sha256": sha256_file(v21_path),
        "candidate_generator_contract": "PROJECT7_OPERATIONAL_V23_STRESS_ADAPTIVE_ABSOLUTE_FILL_RELEASE_TARGET_V1",
        "runtime_contract": "PROJECT7_OPERATIONAL_DEVELOPMENT_V23_STRONG_STORM_HYDRAULIC_CANDIDATE_V1",
        "q95_support_frozen": True,
        "max_setting_delta_per_update": 0.5,
        "supervisory_controls": 82,
        "model_action_channels": 109,
        "passive_channels": 27,
        "competitive_comparators": ["no_control", "internal_rtc", "auto_rbc", "efd"],
        "primary_control_metric": "SYSTEM_WIDE_TFV_M3_AUTHORITATIVE_SWMM",
        "pfv_safety_contract": "PFV_PROPOSED <= 100_M3 + 1.05 * PFV_NO_CONTROL",
        "global_peak_role": "REPORT_ONLY",
        "final_event_ids": list(roles["final"]),
        "final_event_count": 6,
        "final_opened_at_lock": False,
        "policy_mutation_after_lock_forbidden": True,
        "final_can_tune_policy": False,
        "final_must_run_complete_panel_regardless_of_relative_tfv_performance": True,
        "ready_for_final": True,
    }
    destination = Path(args.out).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "policy_lock_path": str(destination),
        "policy_lock_sha256": sha256_file(destination),
        "repository_head_sha": head,
        "formal_mode": protocol_mode,
        "acceptance_basis": acceptance_basis,
        "step2_checkpoint_sha256": lock["step2_checkpoint_sha256"],
        "step2_legacy_component_accepted": step2_pass,
        "step3_disposition": disposition,
        "final_event_ids": lock["final_event_ids"],
        "ready_for_final": True,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
