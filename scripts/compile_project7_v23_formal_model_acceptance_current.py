"""Compile Project7 V23 Formal model-acceptance evidence from pre-existing evaluations.

The command enforces MODEL_ACCEPTANCE_CONTRACT_V4 exactly; it never lowers thresholds.  In
FIXED_POLICY_NO_RETRAIN mode, V23-specific Step3 retraining metrics are not fabricated from mismatched
truth and the frozen V15/V21 policy is recorded as the Step3 disposition.  In exact-match retraining
mode, the preregistered D2/D3 candidate-ranking thresholds are mandatory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rtc.direct_tfv_policy_return import sha256_file


CONTRACT = "PROJECT7_V23_FORMAL_MODEL_ACCEPTANCE_EVIDENCE_V1"
MODEL_CONTRACT = "MODEL_ACCEPTANCE_CONTRACT_V4_DIMENSIONLESS_PREREGISTERED"


def _json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-acceptance-contract", required=True)
    parser.add_argument("--formal-protocol", required=True)
    parser.add_argument("--step1-unobserved-depth-nse", type=float, required=True)
    parser.add_argument("--step2-tfv-exact-truth-rank-correlation", type=float, required=True)
    parser.add_argument("--d2-tfv-rank-correlation", type=float)
    parser.add_argument("--d2-tfv-top1-hit-rate", type=float)
    parser.add_argument("--d3-tfv-rank-correlation", type=float)
    parser.add_argument("--d3-tfv-top1-hit-rate", type=float)
    parser.add_argument("--source-evidence", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    contract_path = Path(args.model_acceptance_contract).resolve()
    protocol_path = Path(args.formal_protocol).resolve()
    contract = _json(contract_path)
    protocol = _json(protocol_path)
    if contract.get("contract") != MODEL_CONTRACT:
        raise ValueError("wrong preregistered model acceptance contract")
    mode = str(protocol.get("formal_mode", ""))
    if mode not in {"EXACT_MATCH_RETRAIN_ALLOWED", "FIXED_POLICY_NO_RETRAIN"}:
        raise ValueError(f"unsupported Formal mode: {mode}")

    step1_threshold = float(contract["step1"]["minimum"]["unobserved_depth_nse"])
    step2_threshold = float(contract["step2"]["minimum"]["tfv_exact_truth_rank_correlation"])
    step1_pass = float(args.step1_unobserved_depth_nse) >= step1_threshold
    step2_pass = float(args.step2_tfv_exact_truth_rank_correlation) >= step2_threshold

    ranking = contract["candidate_ranking"]["minimum"]
    metrics = {
        "d2_tfv_rank_correlation": args.d2_tfv_rank_correlation,
        "d2_tfv_top1_hit_rate": args.d2_tfv_top1_hit_rate,
        "d3_tfv_rank_correlation": args.d3_tfv_rank_correlation,
        "d3_tfv_top1_hit_rate": args.d3_tfv_top1_hit_rate,
    }
    if mode == "EXACT_MATCH_RETRAIN_ALLOWED":
        missing = [key for key, value in metrics.items() if value is None]
        if missing:
            raise ValueError(f"exact-match retraining mode requires candidate-ranking metrics: {missing}")
        ranking_pass = all(float(metrics[key]) >= float(ranking[key]) for key in metrics)
        disposition = "EXACT_MATCH_MINIMAL_RETRAIN_VALIDATED" if ranking_pass else "RANK_ACCEPTANCE_FAIL"
    else:
        ranking_pass = None
        disposition = "FROZEN_V15_V21_FIXED_POLICY_NO_RETRAIN"

    evidence_files = [Path(value).resolve() for value in args.source_evidence]
    for path in evidence_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = {
        "contract": CONTRACT,
        "model_acceptance_contract": MODEL_CONTRACT,
        "model_acceptance_contract_sha256": sha256_file(contract_path),
        "formal_protocol_sha256": sha256_file(protocol_path),
        "formal_mode": mode,
        "step1_unobserved_depth_nse": float(args.step1_unobserved_depth_nse),
        "step1_threshold": step1_threshold,
        "step1_accepted": step1_pass,
        "step2_tfv_exact_truth_rank_correlation": float(
            args.step2_tfv_exact_truth_rank_correlation
        ),
        "step2_threshold": step2_threshold,
        "step2_accepted": step2_pass,
        "candidate_ranking_metrics": metrics,
        "candidate_ranking_thresholds": ranking,
        "candidate_ranking_accepted": ranking_pass,
        "step3_disposition": disposition,
        "gradient_acceptance_role": "DEVELOPMENT_ABLATION_ONLY_NOT_PRODUCTION_GATE",
        "hard_thresholds_lowered_after_results": False,
        "absolute_error_diagnostics_still_required_for_paper": True,
        "source_evidence_paths": [str(path) for path in evidence_files],
        "source_evidence_sha256": [sha256_file(path) for path in evidence_files],
        "final_truth_opened": False,
        "accepted_for_policy_lock": bool(
            step1_pass
            and step2_pass
            and disposition
            in {
                "EXACT_MATCH_MINIMAL_RETRAIN_VALIDATED",
                "FROZEN_V15_V21_FIXED_POLICY_NO_RETRAIN",
            }
        ),
    }
    destination = Path(args.out).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["accepted_for_policy_lock"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
