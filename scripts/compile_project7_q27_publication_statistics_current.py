"""Compile locked Project7 Q27 publication statistics from authoritative event-level results.

The CSV must contain exactly the Policy-Lock event panel.  Required columns are event_id,
proposed_tfv_m3, no_control_tfv_m3, matched_auto_rbc_tfv_m3, matched_efd_tfv_m3,
matched_internal_tfv_m3, proposed_pfv_m3, no_control_pfv_m3 and engineering_pass.  The script never
changes the policy and reports neutral/negative results without suppressing them.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from rtc.project7_q27_publication import (
    POLICY_LOCK_CONTRACT,
    STATISTICS_CONTRACT,
    paired_tfv_statistics,
    validate_final_table_event_set,
)


COMPARATORS = {
    "no_control": "no_control_tfv_m3",
    "matched_auto_rbc": "matched_auto_rbc_tfv_m3",
    "matched_efd": "matched_efd_tfv_m3",
    "matched_internal": "matched_internal_tfv_m3",
}


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid/missing numeric field {key} for {row.get('event_id', '')}") from exc


def _bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "pass", "passed", "yes"}:
        return True
    if text in {"0", "false", "fail", "failed", "no"}:
        return False
    raise ValueError(f"cannot parse boolean: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-lock", required=True)
    parser.add_argument("--event-table", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260830)
    args = parser.parse_args()

    lock = json.loads(Path(args.policy_lock).read_text(encoding="utf-8"))
    if lock.get("contract") != POLICY_LOCK_CONTRACT or lock.get("ready_for_final") is not True:
        raise ValueError("statistics require a valid ready Q27 Policy Lock")
    locked = [str(value) for value in lock.get("locked_final_event_ids", [])]
    with Path(args.event_table).open("rt", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    validate_final_table_event_set(rows, locked)
    order = {event: index for index, event in enumerate(locked)}
    rows.sort(key=lambda row: order[str(row["event_id"])])

    proposed = [_float(row, "proposed_tfv_m3") for row in rows]
    comparator_stats = {
        name: paired_tfv_statistics(
            proposed,
            [_float(row, field) for row in rows],
            repetitions=int(args.bootstrap_repetitions),
            seed=int(args.bootstrap_seed),
        )
        for name, field in COMPARATORS.items()
    }
    pfv_pass = [
        _float(row, "proposed_pfv_m3") <= 1.05 * _float(row, "no_control_pfv_m3") + 1.0e-9
        for row in rows
    ]
    engineering = [_bool(row["engineering_pass"]) for row in rows]
    primary = comparator_stats["matched_auto_rbc"]
    payload: dict[str, Any] = {
        "contract": STATISTICS_CONTRACT,
        "policy_lock": str(Path(args.policy_lock).resolve()),
        "event_table": str(Path(args.event_table).resolve()),
        "event_count": len(rows),
        "event_ids": locked,
        "comparators": comparator_stats,
        "primary_active_comparator": "matched_auto_rbc",
        "primary_claim_classification": primary["claim_classification"],
        "pfv_safety_pass_count": int(sum(pfv_pass)),
        "pfv_safety_event_count": len(pfv_pass),
        "pfv_safety_all": all(pfv_pass),
        "engineering_pass_count": int(sum(engineering)),
        "engineering_event_count": len(engineering),
        "engineering_pass_all": all(engineering),
        "global_peak_role": "REPORT_ONLY",
        "final_results_used_for_training": False,
        "final_results_used_for_tuning": False,
        "neutral_or_negative_results_suppressed": False,
        "ready_for_paper_reporting": True,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
