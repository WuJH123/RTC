"""Audit what Project7 Step2 data exist, what are actually trained, and what must remain held out.

The report distinguishes branch rows, counterfactual groups and unique hydraulic state
keys. It is outcome-read-only except for loading cache metadata required to count group
membership; it does not train a model or launch SWMM.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtc.step2_d4_cache_v125 import D4_SOURCE_KIND
from rtc.step2_train_response_v60 import V60TrainCache, deterministic_rainfall_split_v60
from rtc.step2_v120_data_contract import (
    SOURCE_D2_BRANCHES,
    SOURCE_D2_GROUPS,
    TRAIN_D2_BRANCHES,
    D2_GROUPS,
    DEVELOPMENT_VALIDATION_D2_GROUPS,
    TARGETED_D3_BRANCHES,
    D3_GROUPS,
    verify_d2_source_audit,
)

CONTRACT = "PROJECT7_V126_STEP2_DATA_USAGE_CENSUS_V2_STATE_SUPPORT"


def _branch_count(cache: V60TrainCache, names: list[str]) -> int:
    return int(sum(len(cache.entry(name).indices) for name in names))


def _rain(cache: V60TrainCache, names: list[str]) -> set[str]:
    return {str(cache.entry(name).rainfall_group) for name in names}


def _state_keys(cache: V60TrainCache, names: list[str]) -> set[str]:
    """Counterfactual state identity independent of source-kind and action branch."""
    return {
        "::".join(
            (
                str(cache.entry(name).rainfall_group),
                str(cache.entry(name).event_id),
                str(cache.entry(name).checkpoint_id),
            )
        )
        for name in names
    }


def _support_summary(cache: V60TrainCache, names: list[str]) -> dict[str, object]:
    states = _state_keys(cache, names)
    branches = _branch_count(cache, names)
    return {
        "groups": len(names),
        "branches": branches,
        "unique_state_keys": len(states),
        "branches_per_unique_state": (float(branches) / len(states) if states else 0.0),
        "rainfall_groups": sorted(_rain(cache, names)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--canonical-cache", required=True)
    p.add_argument("--d4-fit-cache", required=True)
    p.add_argument("--d4-audit-cache", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--d2-source-audit")
    p.add_argument("--split-contract")
    p.add_argument("--legacy-cache-manifest")
    args = p.parse_args()

    base = V60TrainCache(args.canonical_cache)
    d2 = base.names("D2")
    targeted_d3 = base.targeted_d3_names()
    legacy_in_base = base.legacy_d3_names()
    if (len(d2), len(targeted_d3)) != (D2_GROUPS, D3_GROUPS):
        raise ValueError(
            f"canonical cache must contain {D2_GROUPS}/{D3_GROUPS} D2/targeted-D3 groups; "
            f"got {len(d2)}/{len(targeted_d3)}"
        )
    names = sorted(d2 + targeted_d3)
    fit, holdout = deterministic_rainfall_split_v60(base, names=names, holdout_fraction=0.20)
    fit_d2 = [name for name in fit if name.startswith("D2::")]
    fit_d3 = [name for name in fit if name.startswith("D3::")]
    hold_d2 = [name for name in holdout if name.startswith("D2::")]
    hold_d3 = [name for name in holdout if name.startswith("D3::")]

    d4_fit = V60TrainCache(args.d4_fit_cache)
    d4_audit = V60TrainCache(args.d4_audit_cache)
    fit_d4 = d4_fit.names(D4_SOURCE_KIND)
    audit_d4 = d4_audit.names(D4_SOURCE_KIND)
    if _rain(d4_fit, fit_d4) & _rain(d4_audit, audit_d4):
        raise ValueError("D4 FIT/AUDIT rainfall leakage")

    source_audit = None
    if bool(args.d2_source_audit) != bool(args.split_contract):
        raise ValueError("--d2-source-audit and --split-contract must be supplied together")
    if args.d2_source_audit:
        source_audit = verify_d2_source_audit(
            args.d2_source_audit, split_contract_path=args.split_contract
        )

    legacy_payload: dict[str, object] = {
        "provided": False,
        "status": "NOT_IN_CANONICAL_TRAINING",
        **_support_summary(base, legacy_in_base),
    }
    if args.legacy_cache_manifest:
        legacy = V60TrainCache(args.legacy_cache_manifest)
        legacy_names = legacy.legacy_d3_names()
        # If a historical cache uses roles now recognised as targeted, still report all D3
        # rows but never silently promote them into V126 fitting.
        all_legacy_d3 = legacy.names("D3")
        legacy_states = _state_keys(legacy, all_legacy_d3)
        train_states = _state_keys(base, fit)
        holdout_states = _state_keys(base, holdout)
        legacy_payload = {
            "provided": True,
            "status": "LINEAGE_AUDIT_REQUIRED_BEFORE_AUXILIARY_PRETRAINING",
            "all_d3": _support_summary(legacy, all_legacy_d3),
            "explicit_legacy_groups": len(legacy_names),
            "explicit_legacy_branches": _branch_count(legacy, legacy_names),
            "state_overlap_with_canonical_trainfit": len(legacy_states & train_states),
            "state_overlap_with_canonical_internal_holdout": len(legacy_states & holdout_states),
            "rainfall_overlap_with_canonical_trainfit": sorted(
                _rain(legacy, all_legacy_d3) & _rain(base, fit)
            ),
            "rainfall_overlap_with_canonical_internal_holdout": sorted(
                _rain(legacy, all_legacy_d3) & _rain(base, holdout)
            ),
            "used_for_v126_training": False,
        }

    fit_d2_states = _state_keys(base, fit_d2)
    fit_d3_states = _state_keys(base, fit_d3)
    hold_d2_states = _state_keys(base, hold_d2)
    hold_d3_states = _state_keys(base, hold_d3)
    d4_fit_states = _state_keys(d4_fit, fit_d4)
    d4_audit_states = _state_keys(d4_audit, audit_d4)
    train_unique_states = fit_d2_states | fit_d3_states | d4_fit_states

    payload = {
        "contract": CONTRACT,
        "important_unit_note": (
            "branch rows are not independent hydraulic states; state support is counted by "
            "rainfall_group::event_id::checkpoint_id"
        ),
        "d2_source_population": {
            "authoritative_source_branches": SOURCE_D2_BRANCHES,
            "authoritative_source_groups": SOURCE_D2_GROUPS,
            "frozen_train_eligible_branches": TRAIN_D2_BRANCHES,
            "frozen_train_eligible_groups": D2_GROUPS,
            "development_validation_groups": DEVELOPMENT_VALIDATION_D2_GROUPS,
            "development_validation_branches": SOURCE_D2_BRANCHES - TRAIN_D2_BRANCHES,
            "source_audit_verified": source_audit is not None,
            "development_validation_allowed_in_training": False,
        },
        "canonical_base_cache": {
            "d2": _support_summary(base, d2),
            "targeted_d3": _support_summary(base, targeted_d3),
            "expected_targeted_d3_branches": TARGETED_D3_BRANCHES,
            "d2_d3_unique_state_overlap": len(_state_keys(base, d2) & _state_keys(base, targeted_d3)),
            "legacy_d3_groups_in_canonical_cache": len(legacy_in_base),
            "legacy_d3_branches_in_canonical_cache": _branch_count(base, legacy_in_base),
        },
        "actual_training_eligibility": {
            "trainfit_d2": _support_summary(base, fit_d2),
            "trainfit_targeted_d3": _support_summary(base, fit_d3),
            "trainfit_d2_d3_state_overlap": len(fit_d2_states & fit_d3_states),
            "d4_fit": _support_summary(d4_fit, fit_d4),
            "d4_fit_state_overlap_with_base_trainfit": len(d4_fit_states & (fit_d2_states | fit_d3_states)),
            "total_branch_rows_used_by_v126_if_all_fit_sources_train": (
                _branch_count(base, fit_d2)
                + _branch_count(base, fit_d3)
                + _branch_count(d4_fit, fit_d4)
            ),
            "total_unique_hydraulic_state_keys_across_fit_sources": len(train_unique_states),
            "internal_holdout_d2": _support_summary(base, hold_d2),
            "internal_holdout_targeted_d3": _support_summary(base, hold_d3),
            "internal_holdout_d2_d3_state_overlap": len(hold_d2_states & hold_d3_states),
            "internal_holdout_allowed_in_training": False,
        },
        "d4": {
            "fit": _support_summary(d4_fit, fit_d4),
            "audit": _support_summary(d4_audit, audit_d4),
            "fit_audit_state_overlap": len(d4_fit_states & d4_audit_states),
            "audit_allowed_in_training": False,
        },
        "legacy_d3": legacy_payload,
        "v125_observed_training_semantics": (
            "D4-FIT was appended to the generic D3 stage; V126 separates D4 into a final "
            "anchor-advantage fine-tune stage"
        ),
        "boundary": {
            "new_swmm": False,
            "training": False,
            "validation_outcomes_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    out.with_suffix(".md").write_text(
        "\n".join(
            [
                "# Project7 V126 Step2 data usage census",
                "",
                f"D2 source: {SOURCE_D2_BRANCHES} branches / {SOURCE_D2_GROUPS} groups.",
                f"D2 frozen train-eligible: {TRAIN_D2_BRANCHES} branches / {D2_GROUPS} groups; "
                f"the remaining {SOURCE_D2_BRANCHES-TRAIN_D2_BRANCHES} branches are development-validation and must not train.",
                f"Canonical targeted D3: {_branch_count(base, targeted_d3)} branches / {len(targeted_d3)} groups.",
                f"Actual TrainFit branch rows: D2 {_branch_count(base, fit_d2)}; D3 {_branch_count(base, fit_d3)}; D4 {_branch_count(d4_fit, fit_d4)}.",
                f"TrainFit D2/D3 shared hydraulic states: {len(fit_d2_states & fit_d3_states)}.",
                f"D4-FIT states already present in base TrainFit: {len(d4_fit_states & (fit_d2_states | fit_d3_states))} / {len(d4_fit_states)}.",
                f"Total nominal fit branch rows: {payload['actual_training_eligibility']['total_branch_rows_used_by_v126_if_all_fit_sources_train']}; unique hydraulic state keys: {len(train_unique_states)}.",
                f"D4 AUDIT: {_branch_count(d4_audit, audit_d4)} branches / {len(audit_d4)} groups / {len(d4_audit_states)} state keys.",
                f"Legacy D3 status: {legacy_payload['status']}.",
                "",
                "Branch counts must not be interpreted as independent training states. Holdout/AUDIT rows remain excluded.",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
