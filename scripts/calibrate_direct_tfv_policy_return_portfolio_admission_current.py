"""Calibrate one-sided admission on the exact V14 multi-candidate policy-return query family."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtc.direct_tfv_policy_return import (
    DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT,
    sha256_file,
)
from rtc.direct_tfv_policy_return_portfolio_admission import (
    DIRECT_TFV_POLICY_RETURN_PORTFOLIO_ADMISSION_CONTRACT,
    derive_policy_return_portfolio_admission,
    validate_policy_return_portfolio_record,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records-jsonl", required=True)
    p.add_argument("--policy-return-checkpoint", required=True)
    p.add_argument("--continuation-policy-sha256", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--coverage", type=float, default=0.90)
    args = p.parse_args()

    records = []
    for line_number, raw in enumerate(
        Path(args.records_jsonl).read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"policy-return portfolio calibration row {line_number} is not an object")
        validate_policy_return_portfolio_record(row)
        if str(row.get("data_role", "")) != "policy_return_calibration":
            raise ValueError("portfolio calibration JSONL may contain only calibration rows")
        records.append(row)
    if not records:
        raise ValueError("portfolio calibration JSONL is empty")
    groups = sorted({str(row["rainfall_group"]) for row in records})
    payload = derive_policy_return_portfolio_admission(
        records=records,
        expected_rainfall_groups=groups,
        policy_return_checkpoint_sha256=sha256_file(args.policy_return_checkpoint),
        continuation_policy_sha256=args.continuation_policy_sha256,
        coverage=float(args.coverage),
    )
    payload["records_jsonl_sha256"] = sha256_file(args.records_jsonl)
    payload["policy_return_checkpoint_path"] = str(Path(args.policy_return_checkpoint).resolve())
    payload["base_admission_contract_verified"] = (
        payload["contract"] == DIRECT_TFV_POLICY_RETURN_ADMISSION_CONTRACT
    )
    payload["portfolio_admission_contract_verified"] = (
        payload["portfolio_admission_contract"]
        == DIRECT_TFV_POLICY_RETURN_PORTFOLIO_ADMISSION_CONTRACT
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
