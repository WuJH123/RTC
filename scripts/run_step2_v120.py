"""Strict public entrypoint for the causal execution-bound V120 trainer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from rtc.step2_train_response_v60 import V60TrainCache
from rtc.step2_v120_data_contract import (
    INTERNAL_HOLDOUT_FRACTION,
    SOURCE_D2_BRANCHES,
    STATE_DOMAIN_CONTRACT,
    TARGETED_D3_BRANCHES,
    TRAIN_D2_BRANCHES,
    finite_auxiliary_value_metrics,
    sha256_file,
    validate_canonical_cache_population,
    validate_internal_holdout_fraction,
    verify_d2_source_audit,
)
from rtc.step2_v120_train_helpers import load_frozen_train_events_v120
from run_step2_v120_causal import main as _legacy_main

BUNDLE_NAME = "step2_v120_execution_bound_causal_bundle.pt"
REPORT_NAME = "STEP2_V120_EXECUTION_BOUND_CAUSAL_REPORT.json"


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Strict canonical Project7 V120 trainer")
    parser.add_argument("--graph", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--d2-source-audit", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--swmm-engine-version", required=True)
    parser.add_argument(
        "--split-contract",
        default=str(repo / "configs" / "project7_v069_split_contract.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--holdout-fraction", type=float, default=INTERNAL_HOLDOUT_FRACTION)
    args = parser.parse_args()

    fraction = validate_internal_holdout_fraction(args.holdout_fraction)
    _, frozen_train = load_frozen_train_events_v120(args.split_contract)
    audit = verify_d2_source_audit(args.d2_source_audit, split_contract_path=args.split_contract)
    cache = V60TrainCache(args.cache_manifest)
    if cache.legacy_d3_names():
        raise ValueError("V120 public trainer rejects legacy dense D3")
    d2, d3 = cache.names("D2"), cache.targeted_d3_names()
    population = validate_canonical_cache_population(cache, d2, d3)
    events = {cache.entry(name).event_id for name in d2 + d3}
    if events != frozen_train:
        raise ValueError("V120 cache event population differs from frozen Train18")

    old_argv = sys.argv
    try:
        sys.argv = [
            old_argv[0],
            "--graph", args.graph,
            "--cache-manifest", args.cache_manifest,
            "--out-dir", args.out_dir,
            "--swmm-engine-version", args.swmm_engine_version,
            "--split-contract", args.split_contract,
            "--device", args.device,
            "--seed", str(args.seed),
            "--holdout-fraction", str(fraction),
        ]
        _legacy_main()
    finally:
        sys.argv = old_argv

    out = Path(args.out_dir)
    bundle_path = out / BUNDLE_NAME
    payload = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("V120 trainer did not produce a dictionary bundle")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not isinstance(metrics.get("holdout_d2"), dict):
        raise ValueError("V120 bundle lacks holdout D2 metrics")
    d2_ok, d2_reasons = finite_auxiliary_value_metrics(metrics["holdout_d2"])
    prior_gate = payload.get("value_gate")
    d3_ok = bool(isinstance(prior_gate, dict) and prior_gate.get("passed") is True)
    combined_ok = bool(d3_ok and d2_ok)

    lineage = payload.setdefault("lineage", {})
    census = payload.setdefault("data_census", {})
    split = payload.setdefault("split", {})
    if not all(isinstance(value, dict) for value in (lineage, census, split)):
        raise ValueError("V120 bundle evidence sections are invalid")
    lineage["d2_source_audit_sha256"] = sha256_file(args.d2_source_audit)
    lineage["d2_source_index_sha256"] = str(audit["source_index_sha256"])
    lineage["strict_training_entrypoint_sha256"] = sha256_file(Path(__file__))
    census.update({
        "source_d2_authoritative_branch_census": SOURCE_D2_BRANCHES,
        "source_d2_audit_contract": str(audit["contract"]),
        "eligible_cache_d2_groups": population["d2_groups"],
        "eligible_cache_d2_branches": population["d2_branches"],
        "eligible_cache_d2_candidates": population["d2_candidates"],
        "expected_eligible_d2_branches": TRAIN_D2_BRANCHES,
        "targeted_d3_groups": population["d3_groups"],
        "targeted_d3_branches": population["d3_branches"],
        "targeted_d3_candidates": population["d3_candidates"],
        "expected_targeted_d3_branches": TARGETED_D3_BRANCHES,
        "branches_per_group": population["branches_per_group"],
        "candidates_per_group": population["candidates_per_group"],
    })
    split["internal_holdout_fraction"] = fraction
    payload["state_input"] = {
        "contract": STATE_DOMAIN_CONTRACT,
        "training_state_source": "authoritative_SWMM_checkpoint_current_state",
        "internal_value_gate_state_source": "authoritative_SWMM_checkpoint_current_state",
        "runtime_state_source": "frozen_Step1_reconstruction_from_sparse_sensors",
        "future_SWMM_state_used_online": False,
        "state_domain_shift_present": True,
        "internal_gate_scope": "oracle-current-state Train-only screening, not low-sensor closed-loop efficacy evidence",
    }
    payload["runtime_compatible"] = combined_ok
    payload["value_gate"] = {
        "passed": combined_ok,
        "primary_source": "holdout_d3_authoritative_current_state",
        "primary_d3": {"passed": d3_ok},
        "auxiliary_d2_integrity": {"passed": d2_ok, "reasons": d2_reasons},
        "low_sensor_closed_loop_efficacy_proven": False,
    }
    torch.save(payload, bundle_path)
    report = {key: value for key, value in payload.items() if key != "state_dict"}
    (out / REPORT_NAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not combined_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
