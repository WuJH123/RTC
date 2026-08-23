"""Frozen Step2 lineage contract for the publication-facing Project7 V23 controller.

V23 was developed and evaluated with one concrete Direct-TFV V5 Step2 checkpoint.  Formal evidence
must therefore refer to that same checkpoint rather than whichever historical Step2 report happens to
have the largest metric.  This module freezes the checkpoint and downstream V15/V21 lineage identities;
it does not train, score, or modify any model.
"""
from __future__ import annotations

from typing import Any, Mapping


V23_STEP2_LINEAGE_EVIDENCE_CONTRACT = "PROJECT7_V23_FROZEN_STEP2_V5_LINEAGE_EVIDENCE_V1"
V23_STEP2_COMPONENT_DIAGNOSTIC_CONTRACT = "PROJECT7_V23_STEP2_V5_COMPONENT_DIAGNOSTIC_V1"
V23_STEP2_CHECKPOINT_SHA256 = "3a05704812a07a914d0ce9d8d026f6c84a4dbed646743f95d27726b29c3a544a"
V23_STEP2_CHECKPOINT_BASENAME = "step2_direct_tfv_value_dev.pt"
V23_STEP2_RUN_DIRECTORY = "direct_tfv_core_v5_dev_20260817_144117"
V23_V15_CHECKPOINT_SHA256 = "faef4de0ee0c8bfccd7d6156fc0057ac5ea6226c7564076278cbfe124f43556e"
V23_V21_CHECKPOINT_SHA256 = "89f2c26439a0ca94b9417ba3f625d04d8a9b1782521a32a602770d9b0957aa5a"


def _sha(value: object) -> str:
    return str(value or "").strip().lower()


def validate_v23_step2_lineage_evidence(payload: Mapping[str, Any]) -> None:
    if payload.get("contract") != V23_STEP2_LINEAGE_EVIDENCE_CONTRACT:
        raise ValueError("wrong V23 Step2 lineage-evidence contract")
    if payload.get("lineage_pass") is not True:
        raise RuntimeError("V23 Step2 lineage evidence did not pass")
    if _sha(payload.get("step2_checkpoint_sha256")) != V23_STEP2_CHECKPOINT_SHA256:
        raise RuntimeError("Formal Step2 evidence is not bound to the V23 V5 checkpoint")
    if _sha(payload.get("asset_manifest_step2_sha256")) != V23_STEP2_CHECKPOINT_SHA256:
        raise RuntimeError("Practical asset manifest does not resolve to V23 V5 Step2")
    if _sha(payload.get("v15_base_step2_sha256")) != V23_STEP2_CHECKPOINT_SHA256:
        raise RuntimeError("V15 rank is not based on the V23 V5 Step2 checkpoint")
    if _sha(payload.get("v21_base_step2_sha256")) != V23_STEP2_CHECKPOINT_SHA256:
        raise RuntimeError("V21 boundary is not based on the V23 V5 Step2 checkpoint")
    if _sha(payload.get("v15_checkpoint_sha256")) != V23_V15_CHECKPOINT_SHA256:
        raise RuntimeError("Formal lineage did not use the frozen V15 rank checkpoint")
    if _sha(payload.get("v21_checkpoint_sha256")) != V23_V21_CHECKPOINT_SHA256:
        raise RuntimeError("Formal lineage did not use the frozen V21 boundary checkpoint")
    if _sha(payload.get("v21_rank_source_checkpoint_sha256")) != V23_V15_CHECKPOINT_SHA256:
        raise RuntimeError("V21 boundary points at another rank-source checkpoint")
    if payload.get("step2_retrained_for_formal") is not False:
        raise RuntimeError("current V23 fixed-policy Formal path forbids Step2 retraining")


def validate_v23_step2_component_diagnostic(payload: Mapping[str, Any]) -> None:
    if payload.get("contract") != V23_STEP2_COMPONENT_DIAGNOSTIC_CONTRACT:
        raise ValueError("wrong V23 Step2 component-diagnostic contract")
    if _sha(payload.get("step2_checkpoint_sha256")) != V23_STEP2_CHECKPOINT_SHA256:
        raise RuntimeError("Step2 component diagnostic is not bound to V23 V5")
    if payload.get("metric_role") != "LEGACY_COMPONENT_DIAGNOSTIC_NOT_POLICY_LOCK_HARD_GATE":
        raise RuntimeError("V23 Step2 diagnostic role changed")
    if payload.get("new_swmm_truth_generated") is not False:
        raise RuntimeError("Step2 diagnostic must reuse existing evidence only")
    if payload.get("step2_retrained") is not False:
        raise RuntimeError("Step2 diagnostic unexpectedly retrained the fixed V23 policy")


__all__ = [
    "V23_STEP2_CHECKPOINT_BASENAME",
    "V23_STEP2_CHECKPOINT_SHA256",
    "V23_STEP2_COMPONENT_DIAGNOSTIC_CONTRACT",
    "V23_STEP2_LINEAGE_EVIDENCE_CONTRACT",
    "V23_STEP2_RUN_DIRECTORY",
    "V23_V15_CHECKPOINT_SHA256",
    "V23_V21_CHECKPOINT_SHA256",
    "validate_v23_step2_component_diagnostic",
    "validate_v23_step2_lineage_evidence",
]
