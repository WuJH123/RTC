"""Evaluate the frozen V123 continuous-gradient gate from causal evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtc.step2_acceptance_v123 import evaluate_continuous_value_gate_v123


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tfv-report", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = json.loads(Path(args.tfv_report).read_text(encoding="utf-8"))
    metric = report["arms"]["B_CAUSAL"]["metrics"]["holdout_d3"]
    # No new gradient evidence is inferred from ranking.  The gradient gate is
    # deliberately fail-closed until an independent authoritative gradient audit exists.
    gate = evaluate_continuous_value_gate_v123(
        causal_input_verified=bool(
            report.get("future_realized_rainfall_used_as_model_input", {}).get(
                "B_CAUSAL"
            )
            is False
        ),
        holdout_rank=float(metric["rank"]),
        holdout_top1=float(metric["top1_rate"]),
        gradient_sign_accuracy=0.0,
        gradient_cosine=0.0,
    )
    gate["gradient_evidence"] = {
        "available": False,
        "reason": "independent causal gradient sign/cosine audit not present; fail closed",
    }
    gate["source_report"] = str(Path(args.tfv_report).resolve())
    path = Path(args.out).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
