"""Compile structurally valid V127 continuous-MPC evidence.

Ranking and gradient scores are scientific quality evidence, not universal hard runtime
thresholds.  This compiler therefore verifies causality, finite metrics and exact model
lineage.  It refuses to combine reports from different checkpoints, but it does not lower
or raise an arbitrary numerical score to decide whether the continuous method exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from rtc.step3_mpc_v127 import Step2GradientEvidenceV127

V127_CONTINUOUS_GATE_CONTRACT = "PROJECT7_V127_CONTINUOUS_MPC_EVIDENCE_V2_LINEAGE_BOUND_NOT_SCORE_GATED"


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_sha(value: object, *, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"V127 evidence lacks canonical {label} SHA256")
    return text


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
        d2_gradient_sign_accuracy=float(
            d2.get("tfv_gradient_sign_accuracy", float("nan"))
        ),
        d2_gradient_cosine_similarity=float(
            d2.get("tfv_gradient_cosine_similarity", float("nan"))
        ),
        d5_gradient_sign_accuracy=float(
            d5_audit.get("tfv_gradient_sign_accuracy", float("nan"))
        ),
        d5_gradient_cosine_similarity=float(
            d5_audit.get("tfv_gradient_cosine_similarity", float("nan"))
        ),
        causal_step1_state_verified=bool(d2.get("causal_step1_state", False)),
        causal_rainfall_verified=bool(d2.get("causal_rainfall", False)),
    )
    evidence.validate()

    ranking_step2 = _require_sha(
        ranking.get("step2_sha256"), label="ranking Step2"
    )
    d2_step2 = _require_sha(
        (d2.get("lineage") or {}).get("step2_sha256"), label="D2 Step2"
    )
    d5_step2 = _require_sha(
        d5.get("final_step2_sha256"), label="D5 final Step2"
    )
    if len({ranking_step2, d2_step2, d5_step2}) != 1:
        raise ValueError(
            "V127 evidence reports were produced from different Step2 checkpoints"
        )
    metrics = dict(evidence.__dict__)
    if any(
        not math.isfinite(float(metrics[key]))
        for key in (
            "holdout_rank",
            "holdout_top1",
            "d2_gradient_sign_accuracy",
            "d2_gradient_cosine_similarity",
            "d5_gradient_sign_accuracy",
            "d5_gradient_cosine_similarity",
        )
    ):
        raise ValueError("V127 evidence contains non-finite scientific metrics")

    payload = {
        "contract": V127_CONTINUOUS_GATE_CONTRACT,
        "passed": True,
        "verdict": "V127_CONTINUOUS_EVIDENCE_STRUCTURALLY_VALID",
        "execution_semantics": "continuous MPC remains enabled; quality scores are reported and interpreted, not universal runtime switches",
        "metrics": metrics,
        "step2_sha256": ranking_step2,
        "sources": {
            "ranking_report_sha256": _sha(args.ranking_report),
            "d2_gradient_report_sha256": _sha(args.d2_gradient_report),
            "d5_gradient_report_sha256": _sha(args.d5_gradient_report),
        },
        "boundary": {
            "development_evidence_only": True,
            "performance_threshold_used_as_runtime_switch": False,
            "causal_lineage_required": True,
            "same_step2_checkpoint_required": True,
            "validation_accessed": False,
            "final_accessed": False,
            "formal_accessed": False,
            "policy_lock": False,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
