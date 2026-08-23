"""Compile publication-facing event-level statistics from locked Project7 V23 Final6 evidence.

This command never runs SWMM and never changes the policy. It reads the immutable Final6 comparison,
keeps rainfall event as the statistical unit, and reports paired effect sizes, exhaustive six-event
bootstrap confidence intervals and exact two-sided sign tests for each competitive baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

from rtc.direct_tfv_policy_return import sha256_file
from rtc.project7_publication_statistics import (
    FINAL_EVENT_COUNT,
    exact_bootstrap_mean_ci,
    exact_two_sided_sign_test_pvalue,
)


FINAL_COMPARISON_CONTRACT = "PROJECT7_V23_POLICY_LOCKED_FINAL6_COMPARISON_V1"
PUBLICATION_STATISTICS_CONTRACT = "PROJECT7_V23_FINAL6_PUBLICATION_STATISTICS_V1"
BASELINES = ("no_control", "internal_rtc", "auto_rbc", "efd")


def _json(path: str | Path) -> dict[str, Any]:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {source}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-comparison", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source = Path(args.final_comparison).resolve()
    final = _json(source)
    if final.get("contract") != FINAL_COMPARISON_CONTRACT:
        raise ValueError("wrong Project7 V23 locked Final6 comparison contract")
    if final.get("formal_evidence") is not True or final.get("policy_locked") is not True:
        raise RuntimeError("publication statistics require Policy-Locked Formal evidence")
    if final.get("final_results_used_for_training") is not False:
        raise RuntimeError("Final results were marked as used for training")
    if final.get("final_results_used_for_tuning") is not False:
        raise RuntimeError("Final results were marked as used for tuning")
    events = list(final.get("event_results", ()))
    if len(events) != FINAL_EVENT_COUNT:
        raise ValueError("publication statistics require exactly six Final events")

    comparisons: dict[str, Any] = {}
    for baseline in BASELINES:
        reductions: list[float] = []
        deltas: list[float] = []
        wins = 0
        losses = 0
        ties = 0
        for event in events:
            row = event.get("comparisons", {}).get(baseline)
            if not isinstance(row, dict):
                raise ValueError(f"Final event lacks baseline comparison: {baseline}")
            reduction = row.get("tfv_reduction_pct")
            if reduction is None:
                raise ValueError(f"Final event has undefined TFV reduction: {baseline}")
            reduction_value = float(reduction)
            delta_value = float(row["proposed_minus_baseline_tfv_m3"])
            reductions.append(reduction_value)
            deltas.append(delta_value)
            if delta_value < -1.0e-9:
                wins += 1
            elif delta_value > 1.0e-9:
                losses += 1
            else:
                ties += 1

        reduction_ci = exact_bootstrap_mean_ci(reductions)
        delta_ci = exact_bootstrap_mean_ci(deltas)
        comparisons[baseline] = {
            "event_count": FINAL_EVENT_COUNT,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "event_balanced_mean_tfv_reduction_pct": statistics.mean(reductions),
            "event_balanced_median_tfv_reduction_pct": statistics.median(reductions),
            "event_balanced_mean_tfv_reduction_pct_bootstrap95_ci": list(reduction_ci),
            "paired_mean_proposed_minus_baseline_tfv_m3": statistics.mean(deltas),
            "paired_median_proposed_minus_baseline_tfv_m3": statistics.median(deltas),
            "paired_mean_delta_tfv_m3_bootstrap95_ci": list(delta_ci),
            "exact_two_sided_sign_test_pvalue": exact_two_sided_sign_test_pvalue(
                wins=wins,
                losses=losses,
            ),
            "aggregate_volume_tfv_reduction_pct": final["aggregate"][baseline][
                "aggregate_volume_tfv_reduction_pct"
            ],
            "direction_positive_for_proposed": bool(statistics.mean(reductions) > 0.0),
        }

    payload = {
        "contract": PUBLICATION_STATISTICS_CONTRACT,
        "source_final_comparison_path": str(source),
        "source_final_comparison_sha256": sha256_file(source),
        "statistical_unit": "RAINFALL_EVENT",
        "event_count": FINAL_EVENT_COUNT,
        "decision_rows_are_independent_replicates": False,
        "bootstrap_method": "EXHAUSTIVE_NONPARAMETRIC_EVENT_BOOTSTRAP_6_POWER_6",
        "bootstrap_resample_count": FINAL_EVENT_COUNT**FINAL_EVENT_COUNT,
        "sign_test": "EXACT_TWO_SIDED_BINOMIAL_SIGN_TEST_EXCLUDING_TIES",
        "inferential_results_are_secondary_to_effect_SIZE_AND_DIRECTION": True,
        "comparisons": comparisons,
        "pfv_safety_pass_count": int(final.get("pfv_safety_pass_count", 0)),
        "pfv_safety_all_events_pass": bool(final.get("pfv_safety_all_events_pass")),
        "engineering_all_events_pass": bool(final.get("engineering_all_events_pass")),
        "ready_for_paper_reporting": bool(final.get("ready_for_paper_reporting")),
        "final_results_used_for_training": False,
        "final_results_used_for_tuning": False,
    }

    destination = Path(args.out).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
