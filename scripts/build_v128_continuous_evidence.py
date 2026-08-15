"""Compile structurally valid, same-checkpoint V128 continuous-MPC evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from rtc.checkpoint_v128 import V128_CHECKPOINT_CONTRACT
from rtc.engineering_v128 import V128_IDEALIZED_ENVELOPE_SOURCE
from rtc.step2_differentiable_v128 import V128_STEP2_CONTRACT
from rtc.step3_mpc_v127 import Step2GradientEvidenceV127

from audit_step2_v128_d2_gradients_fast import V128_D2_FAST_GRADIENT_AUDIT_CONTRACT
from audit_step2_v128_fast import V128_FAST_RANKING_CONTRACT
from run_step2_v128_d5_gradient_fast import V128_D5_FAST_RUN_CONTRACT

V128_CONTINUOUS_EVIDENCE_CONTRACT = (
    "PROJECT7_V128_CONTINUOUS_MPC_EVIDENCE_V1_TYPED_SAME_CHECKPOINT"
)


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _require_sha(value: object, *, label: str) -> str:
    text = str(value or "").lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"V128 evidence lacks canonical {label} SHA256")
    return text


def _require_contract(payload: dict, expected: str, label: str) -> None:
    if payload.get("contract") != expected:
        raise ValueError(f"V128 {label} report contract mismatch")
    if payload.get("step2_contract") != V128_STEP2_CONTRACT:
        raise ValueError(f"V128 {label} report is not bound to typed Step2")
    if payload.get("checkpoint_contract") != V128_CHECKPOINT_CONTRACT:
        raise ValueError(f"V128 {label} report is not bound to strict V128 checkpoint")


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
    _require_contract(ranking, V128_FAST_RANKING_CONTRACT, "ranking")
    _require_contract(d2, V128_D2_FAST_GRADIENT_AUDIT_CONTRACT, "D2")
    _require_contract(d5, V128_D5_FAST_RUN_CONTRACT, "D5")

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

    ranking_step2 = _require_sha(ranking.get("step2_sha256"), label="ranking Step2")
    d2_step2 = _require_sha(
        (d2.get("lineage") or {}).get("step2_sha256"), label="D2 Step2"
    )
    d5_step2 = _require_sha(d5.get("final_step2_sha256"), label="D5 final Step2")
    if len({ranking_step2, d2_step2, d5_step2}) != 1:
        raise ValueError("V128 ranking/D2/D5 evidence describes different Step2 checkpoints")

    metrics = dict(evidence.__dict__)
    numeric = (
        "holdout_rank",
        "holdout_top1",
        "d2_gradient_sign_accuracy",
        "d2_gradient_cosine_similarity",
        "d5_gradient_sign_accuracy",
        "d5_gradient_cosine_similarity",
    )
    if any(not math.isfinite(float(metrics[key])) for key in numeric):
        raise ValueError("V128 evidence contains non-finite scientific metrics")

    payload = {
        "contract": V128_CONTINUOUS_EVIDENCE_CONTRACT,
        "passed": True,
        "verdict": "V128_CONTINUOUS_EVIDENCE_STRUCTURALLY_VALID",
        "step2_contract": V128_STEP2_CONTRACT,
        "checkpoint_contract": V128_CHECKPOINT_CONTRACT,
        "step2_sha256": ranking_step2,
        "metrics": metrics,
        "d5_decoder_envelope_source": V128_IDEALIZED_ENVELOPE_SOURCE,
        "custom_engineering_envelope_supported_by_this_d5_evidence": False,
        "custom_envelope_rule": (
            "a non-default per-actuator envelope changes the exact fraction-to-target decoder; "
            "regenerate decoder-space D5 before using such an envelope in authoritative runtime"
        ),
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
