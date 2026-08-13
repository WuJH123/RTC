"""Strict promotion wrapper for low-sensor V120 evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .promote_v120 import PROMOTION_CONTRACT, promote_v120
from .step2_v120_data_contract import STATE_DOMAIN_CONTRACT, sha256_file


def _json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote strict low-sensor V120 bundle")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--runtime-acceptance", required=True)
    parser.add_argument("--development-run-index", required=True)
    parser.add_argument("--step1", required=True)
    parser.add_argument("--d2-source-audit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise ValueError("V120 bundle is invalid")
    lineage = bundle.get("lineage")
    state_input = bundle.get("state_input")
    if not isinstance(lineage, dict) or str(lineage.get("d2_source_audit_sha256", "")) != sha256_file(args.d2_source_audit):
        raise ValueError("V120 promotion D2 source audit differs from training")
    if not isinstance(state_input, dict) or state_input.get("contract") != STATE_DOMAIN_CONTRACT:
        raise ValueError("V120 bundle lacks explicit train/runtime state-domain contract")

    step1_sha = sha256_file(args.step1)
    index = pd.read_csv(args.development_run_index)
    proposed = index[index["strategy"].astype(str) == "proposed"]
    if proposed.empty:
        raise ValueError("V120 promotion requires Proposed development evidence")
    for _, row in proposed.iterrows():
        meta = _json(str(row["metadata_path"]))
        if str(meta.get("step1_model_sha256", "")) != step1_sha:
            raise ValueError("development Proposed run used another Step1 checkpoint")

    promote_v120(
        bundle_path=args.bundle,
        graph_path=args.graph,
        controller_config_path=args.config,
        runtime_acceptance_path=args.runtime_acceptance,
        development_run_index_path=args.development_run_index,
        output_path=args.out,
    )
    promoted = torch.load(args.out, map_location="cpu", weights_only=False)
    promotion = promoted.get("promotion")
    if not isinstance(promotion, dict) or promotion.get("contract") != PROMOTION_CONTRACT:
        raise ValueError("legacy promotion did not produce expected evidence")
    promotion["step1_model_sha256"] = step1_sha
    promotion["d2_source_audit_sha256"] = sha256_file(args.d2_source_audit)
    promotion["low_sensor_step1_closed_loop_executed"] = True
    torch.save(promoted, args.out)
    print(json.dumps({
        "contract": PROMOTION_CONTRACT,
        "promoted_bundle_sha256": sha256_file(args.out),
        "step1_model_sha256": step1_sha,
        "d2_source_audit_sha256": promotion["d2_source_audit_sha256"],
        "model_parameters_changed": False,
        "thresholds_retuned": False,
    }, indent=2))


if __name__ == "__main__":
    main()
