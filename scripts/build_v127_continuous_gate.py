"""Compile the frozen Project7 V127 continuous-MPC evidence gate."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rtc.step3_mpc_v127 import Step2GradientEvidenceV127

V127_CONTINUOUS_GATE_CONTRACT = "PROJECT7_V127_CONTINUOUS_MPC_EVIDENCE_GATE_V1"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranking-report", required=True)
    p.add_argument("--d2-gradient-report", required=True)
    p.add_argument("--d5-gradient-report", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ranking = json.loads(Path(args.ranking_report).read_text(encoding="utf-8"))
    d2 = json.loads(Path(args.d2_gradient_report).read_text(encoding="utf-8"))
    d5 = json.loads(Path(args.d5_gradient_report).read_text(encoding="utf-8"))
    holdout = ranking.get("internal_holdout_d3", {})
    d5_audit = d5.get("audit_after", {})
    evidence = Step2GradientEvidenceV127(
        holdout_rank=float(holdout.get("rank", float("nan"))),
        holdout_top1=float(holdout.get("top1", float("nan"))),
        d2_gradient_sign_accuracy=float(d2.get("tfv_gradient_sign_accuracy", float("nan"))),
        d2_gradient_cosine_similarity=float(d2.get("tfv_gradient_cosine_similarity", float("nan"))),
        d5_gradient_sign_accuracy=float(d5_audit.get("tfv_gradient_sign_accuracy", float("nan"))),
        d5_gradient_cosine_similarity=float(d5_audit.get("tfv_gradient_cosine_similarity", float("nan"))),
        causal_step1_state_verified=bool(d2.get("causal_step1_state", False)),
        causal_rainfall_verified=bool(d2.get("causal_rainfall", False)),
    )
    passed = True
    failure = ""
    try:
        evidence.validate()
    except ValueError as exc:
        passed = False
        failure = str(exc)
    payload = {
        "contract": V127_CONTINUOUS_GATE_CONTRACT,
        "passed": passed,
        "verdict": "V127_CONTINUOUS_MPC_AUTHORIZED_DEVELOPMENT" if passed else "V127_CONTINUOUS_MPC_BLOCKED",
        "failure": failure,
        "thresholds": {
            "internal_holdout_d3_rank_min": 0.70,
            "internal_holdout_d3_top1_min": 0.50,
            "d2_gradient_sign_min": 0.70,
            "d2_gradient_cosine_min": 0.60,
            "d5_gradient_sign_min": 0.70,
            "d5_gradient_cosine_min": 0.60,
        },
        "metrics": evidence.__dict__,
        "sources": {
            "ranking_report_sha256": _sha(args.ranking_report),
            "d2_gradient_report_sha256": _sha(args.d2_gradient_report),
            "d5_gradient_report_sha256": _sha(args.d5_gradient_report),
        },
        "boundary": {
            "development_gate_only": True,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
            "policy_lock": False,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
