"""Strict public entrypoint for the causal execution-bound V120 trainer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rtc.step2_train_response_v60 import V60TrainCache
from rtc.step2_v120_data_contract import (
    INTERNAL_HOLDOUT_FRACTION,
    validate_canonical_cache_population,
    validate_internal_holdout_fraction,
    verify_d2_source_audit,
)
from rtc.step2_v120_train_helpers import load_frozen_train_events_v120
from run_step2_v120_causal import main as _legacy_main


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
    verify_d2_source_audit(args.d2_source_audit, split_contract_path=args.split_contract)
    cache = V60TrainCache(args.cache_manifest)
    if cache.legacy_d3_names():
        raise ValueError("V120 public trainer rejects legacy dense D3")
    d2, d3 = cache.names("D2"), cache.targeted_d3_names()
    validate_canonical_cache_population(cache, d2, d3)
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


if __name__ == "__main__":
    main()
