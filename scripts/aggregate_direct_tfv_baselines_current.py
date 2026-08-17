"""Aggregate multiple provenance-verified Direct-TFV Development baseline comparisons event-wise."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from rtc.baseline_panel import (
    DIRECT_TFV_BASELINE_PANEL_CONTRACT,
    SCIENTIFIC_COMPARATOR_IDS,
)


DIRECT_TFV_BASELINE_PANEL_AGGREGATE_CONTRACT = (
    "PROJECT7_DIRECT_TFV_DEVELOPMENT_BASELINE_PANEL_EVENT_BALANCED_V1"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", action="append", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    comparisons: list[dict[str, Any]] = []
    seen_events: set[str] = set()
    for raw_path in args.comparison:
        path = Path(raw_path).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"comparison must be a JSON object: {path}")
        if payload.get("contract") != DIRECT_TFV_BASELINE_PANEL_CONTRACT:
            raise ValueError(f"not a current Direct-TFV baseline comparison: {path}")
        if payload.get("baseline_provenance_verified_all") is not True:
            raise ValueError(f"comparison lacks verified baseline provenance: {path}")
        event_id = str(payload.get("event_id", ""))
        if not event_id or event_id in seen_events:
            raise ValueError(f"comparison event_id is missing or duplicated: {event_id!r}")
        seen_events.add(event_id)
        comparisons.append(payload)

    event_count = len(comparisons)
    long_rows: list[dict[str, Any]] = []
    by_strategy: dict[str, list[dict[str, Any]]] = {}
    for comparison in comparisons:
        event_id = str(comparison["event_id"])
        rows = comparison.get("rows")
        if not isinstance(rows, list):
            raise ValueError(f"comparison {event_id} lacks rows")
        for raw_row in rows:
            if not isinstance(raw_row, dict):
                raise ValueError(f"comparison {event_id} contains a non-object row")
            row = {"event_id": event_id, **raw_row}
            long_rows.append(row)
            by_strategy.setdefault(str(row["strategy"]), []).append(row)

    summaries: dict[str, Any] = {}
    for strategy, rows in sorted(by_strategy.items()):
        if len(rows) != event_count:
            raise ValueError(f"strategy {strategy} is missing from one or more events")
        tfv = [float(row["tfv_m3"]) for row in rows]
        reductions = [
            float(row["tfv_reduction_vs_no_control_pct"])
            for row in rows
            if row.get("tfv_reduction_vs_no_control_pct") is not None
        ]
        summaries[strategy] = {
            "event_count": event_count,
            "mean_tfv_m3": statistics.fmean(tfv),
            "median_tfv_m3": statistics.median(tfv),
            "mean_tfv_reduction_vs_no_control_pct": (
                statistics.fmean(reductions) if reductions else None
            ),
            "median_tfv_reduction_vs_no_control_pct": (
                statistics.median(reductions) if reductions else None
            ),
            "minimum_tfv_reduction_vs_no_control_pct": min(reductions) if reductions else None,
            "maximum_tfv_reduction_vs_no_control_pct": max(reductions) if reductions else None,
        }

    no_control_wins = sum(bool(item.get("proposed_beats_no_control")) for item in comparisons)
    strong_wins = {
        strategy: sum(
            bool(item.get("proposed_beats_scientific_comparator", {}).get(strategy))
            for item in comparisons
        )
        for strategy in SCIENTIFIC_COMPARATOR_IDS
    }
    if no_control_wins == event_count and all(value == event_count for value in strong_wins.values()):
        classification = "DEVELOPMENT_METHOD_ADVANTAGE_SUPPORTED"
    elif no_control_wins == event_count:
        classification = "RULE_BASELINE_COMPETITIVENESS_LIMITED"
    else:
        classification = "DEVELOPMENT_NO_CONTROL_BENEFIT_INCONSISTENT"

    proposed_total = sum(float(row["tfv_m3"]) for row in by_strategy["proposed"])
    no_control_total = sum(float(row["tfv_m3"]) for row in by_strategy["no_control"])
    pooled_reduction = (
        100.0 * (no_control_total - proposed_total) / no_control_total
        if no_control_total > 0.0
        else None
    )
    payload = {
        "contract": DIRECT_TFV_BASELINE_PANEL_AGGREGATE_CONTRACT,
        "development_only": True,
        "event_balanced_primary": True,
        "event_ids": sorted(seen_events),
        "event_count": event_count,
        "strategy_summary": summaries,
        "proposed_no_control_win_count": no_control_wins,
        "proposed_scientific_comparator_win_count": strong_wins,
        "pooled_tfv_reduction_vs_no_control_pct_secondary": pooled_reduction,
        "global_peak_role": "report_only",
        "classification": classification,
    }

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in long_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(long_rows)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
