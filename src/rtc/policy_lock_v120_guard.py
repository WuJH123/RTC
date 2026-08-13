from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .policy_lock_v120 import create_policy_lock_v120
from .step2_v120_data_contract import STATE_DOMAIN_CONTRACT, sha256_file, verify_d2_source_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Create strict V120 Policy Lock")
    parser.add_argument("--artefacts", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    artifacts = json.loads(Path(args.artefacts).read_text(encoding="utf-8"))
    if not isinstance(artifacts, dict):
        raise ValueError("V120 artifacts must be a JSON object")
    required = {"d2_source_audit", "split_contract", "step1_model", "step2_model"}
    missing = sorted(required - set(artifacts))
    if missing:
        raise ValueError(f"strict V120 Policy Lock missing: {missing}")
    audit = str(artifacts["d2_source_audit"])
    verify_d2_source_audit(audit, split_contract_path=str(artifacts["split_contract"]))
    bundle = torch.load(str(artifacts["step2_model"]), map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict):
        raise ValueError("V120 Step2 bundle is invalid")
    lineage, state, promotion = bundle.get("lineage"), bundle.get("state_input"), bundle.get("promotion")
    audit_sha = sha256_file(audit)
    if not isinstance(lineage, dict) or lineage.get("d2_source_audit_sha256") != audit_sha:
        raise ValueError("locked V120 bundle belongs to another D2 audit")
    if not isinstance(state, dict) or state.get("contract") != STATE_DOMAIN_CONTRACT:
        raise ValueError("locked V120 bundle lacks state-domain evidence")
    if not isinstance(promotion, dict) or promotion.get("low_sensor_step1_closed_loop_executed") is not True:
        raise ValueError("V120 promotion lacks Step1 low-sensor closed-loop evidence")
    if promotion.get("step1_model_sha256") != sha256_file(str(artifacts["step1_model"])):
        raise ValueError("V120 promotion used another Step1 checkpoint")
    if promotion.get("d2_source_audit_sha256") != audit_sha:
        raise ValueError("V120 promotion used another D2 audit")
    print(json.dumps(create_policy_lock_v120(
        artefacts_path=args.artefacts,
        output_path=args.out,
    ), indent=2))


if __name__ == "__main__":
    main()
