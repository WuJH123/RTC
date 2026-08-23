"""Compile the legacy Step2 component diagnostic and bind it to V23's frozen V5 checkpoint.

The numeric diagnostic remains a publication limitation/diagnostic in FIXED_POLICY_NO_RETRAIN mode.
This command prevents metrics from another historical Step2 version from being attached to V23.  It
uses only existing evidence files and never runs SWMM or retrains Step2.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from rtc.direct_tfv_policy_return import sha256_file
from rtc.project7_v23_step2_lineage import (
    V23_STEP2_CHECKPOINT_SHA256,
    V23_STEP2_COMPONENT_DIAGNOSTIC_CONTRACT,
    V23_STEP2_RUN_DIRECTORY,
    validate_v23_step2_component_diagnostic,
    validate_v23_step2_lineage_evidence,
)


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _source_bound_to_v5(path: Path) -> bool:
    if V23_STEP2_RUN_DIRECTORY in path.parts:
        return True
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return V23_STEP2_CHECKPOINT_SHA256 in text.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step2-lineage-evidence", required=True)
    parser.add_argument("--tfv-exact-truth-rank-correlation", type=float, required=True)
    parser.add_argument("--query-balanced-top1", type=float)
    parser.add_argument("--mean-selected-regret-m3", type=float)
    parser.add_argument("--source-evidence", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    lineage_path = Path(args.step2_lineage_evidence).resolve()
    lineage = _json(lineage_path)
    validate_v23_step2_lineage_evidence(lineage)
    source_paths = [Path(value).resolve() for value in args.source_evidence]
    if not source_paths:
        raise ValueError("at least one V5-specific Step2 evidence file is required")
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    bound = [path for path in source_paths if _source_bound_to_v5(path)]
    if not bound:
        raise RuntimeError(
            "none of the Step2 diagnostic sources is bound to the frozen V5 run; "
            "do not substitute V4/V41/other historical metrics"
        )

    rank = float(args.tfv_exact_truth_rank_correlation)
    if not math.isfinite(rank):
        raise ValueError("Step2 TFV rank correlation must be finite")
    top1 = None if args.query_balanced_top1 is None else float(args.query_balanced_top1)
    regret = None if args.mean_selected_regret_m3 is None else float(args.mean_selected_regret_m3)
    if top1 is not None and (not math.isfinite(top1) or not 0.0 <= top1 <= 1.0):
        raise ValueError("query-balanced top1 must lie in [0,1]")
    if regret is not None and (not math.isfinite(regret) or regret < 0.0):
        raise ValueError("selected regret must be finite and nonnegative")

    payload = {
        "contract": V23_STEP2_COMPONENT_DIAGNOSTIC_CONTRACT,
        "metric_role": "LEGACY_COMPONENT_DIAGNOSTIC_NOT_POLICY_LOCK_HARD_GATE",
        "step2_checkpoint_path": str(lineage["step2_checkpoint_path"]),
        "step2_checkpoint_sha256": str(lineage["step2_checkpoint_sha256"]),
        "step2_lineage_evidence_path": str(lineage_path),
        "step2_lineage_evidence_sha256": sha256_file(lineage_path),
        "tfv_exact_truth_rank_correlation": rank,
        "query_balanced_top1": top1,
        "mean_selected_regret_m3": regret,
        "source_evidence_paths": [str(path) for path in source_paths],
        "source_evidence_sha256": [sha256_file(path) for path in source_paths],
        "v5_bound_source_evidence_paths": [str(path) for path in bound],
        "step2_retrained": False,
        "new_swmm_truth_generated": False,
        "new_training_data_generated": False,
        "standalone_acceptance_threshold": 0.70,
        "standalone_acceptance_pass": bool(rank >= 0.70),
    }
    validate_v23_step2_component_diagnostic(payload)
    destination = Path(args.out).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
