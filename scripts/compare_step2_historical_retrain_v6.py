"""Compare frozen V5 and historical-preservation V6 Step2 on the same internal holdout.

This is a Development-only gate.  It never promotes a checkpoint or edits V23 lineage.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


CONTRACT = "PROJECT7_STEP2_HISTORICAL_RETRAIN_V6_COMPARISON_V1"


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _metric(report: dict[str, Any], split: str, name: str) -> float:
    try:
        value = float(report["evaluations"][split][name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing numeric metric evaluations.{split}.{name}") from exc
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric evaluations.{split}.{name}")
    return value


def _no_regression(candidate: float, baseline: float, *, atol: float = 1.0e-12) -> bool:
    return candidate + atol >= baseline


def _no_increase(candidate: float, baseline: float, *, atol: float = 1.0e-12) -> bool:
    return candidate <= baseline + atol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    baseline = _load(args.baseline_report)
    candidate = _load(args.candidate_report)
    if baseline.get("profile") != "dev" or candidate.get("profile") != "dev":
        raise RuntimeError("historical retrain comparison requires full DEV reports")
    if baseline.get("seed") != candidate.get("seed"):
        raise RuntimeError("baseline/candidate seed mismatch")
    if baseline.get("selected_group_counts") != candidate.get("selected_group_counts"):
        raise RuntimeError("baseline/candidate group counts differ")
    baseline_lineage = baseline.get("lineage", {})
    candidate_lineage = candidate.get("lineage", {})
    lineage_keys = (
        "graph_sha256",
        "base_cache_sha256",
        "d4_fit_cache_sha256",
        "d4_audit_cache_sha256",
        "causal_rainfall_sha256",
        "causal_state_store_sha256",
        "step1_model_semantic_sha256",
        "sensor_layout_semantic_sha256",
    )
    mismatches = [key for key in lineage_keys if baseline_lineage.get(key) != candidate_lineage.get(key)]
    if mismatches:
        raise RuntimeError(f"baseline/candidate lineage mismatch: {mismatches}")

    metrics: dict[str, dict[str, float]] = {}
    for split in ("internal_holdout_d2", "internal_holdout_d3"):
        metrics[split] = {}
        for name in (
            "rank",
            "pairwise",
            "sign",
            "top1_fraction",
            "selected_harmful_fraction",
            "selected_true_delta_tfv_m3",
            "delta_tfv_mae_m3",
            "selected_regret_m3",
        ):
            b = _metric(baseline, split, name)
            c = _metric(candidate, split, name)
            metrics[split][f"baseline_{name}"] = b
            metrics[split][f"candidate_{name}"] = c
            metrics[split][f"delta_{name}"] = c - b

    d2 = metrics["internal_holdout_d2"]
    d3 = metrics["internal_holdout_d3"]
    gates = {
        "d2_rank_not_regressed": _no_regression(d2["candidate_rank"], d2["baseline_rank"]),
        "d2_pairwise_not_regressed": _no_regression(d2["candidate_pairwise"], d2["baseline_pairwise"]),
        "d2_harmful_not_increased": _no_increase(
            d2["candidate_selected_harmful_fraction"], d2["baseline_selected_harmful_fraction"]
        ),
        "d3_pairwise_improved": d3["candidate_pairwise"] > d3["baseline_pairwise"],
        "d3_sign_not_regressed": _no_regression(d3["candidate_sign"], d3["baseline_sign"]),
        "d3_regret_improved": d3["candidate_selected_regret_m3"] < d3["baseline_selected_regret_m3"],
        "d3_harmful_not_increased": _no_increase(
            d3["candidate_selected_harmful_fraction"], d3["baseline_selected_harmful_fraction"]
        ),
        "candidate_d3_rank_ge_0_70_diagnostic": d3["candidate_rank"] >= 0.70,
    }
    hard_gate_names = [name for name in gates if not name.endswith("_diagnostic")]
    pass_hard_gates = all(bool(gates[name]) for name in hard_gate_names)

    payload = {
        "contract": CONTRACT,
        "development_only": True,
        "baseline_report": str(Path(args.baseline_report).resolve()),
        "candidate_report": str(Path(args.candidate_report).resolve()),
        "lineage_match": True,
        "metrics": metrics,
        "gates": gates,
        "hard_gate_names": hard_gate_names,
        "pass_existing_data_offline_gate": pass_hard_gates,
        "formal_authorized": False,
        "step3_retraining_or_recalibration_required_before_any_new_formal": pass_hard_gates,
        "interpretation": (
            "Candidate may proceed to downstream Step3 Development retraining/recalibration."
            if pass_hard_gates
            else "Reject this Step2 arm; do not spend SWMM or Formal budget on it."
        ),
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not pass_hard_gates:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
