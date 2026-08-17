"""Compare one clean Direct-TFV Development event with a provenance-verified six-baseline panel."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rtc.baseline_panel import (
    CURRENT_SIX_BASELINE_DEVELOPMENT_CONTRACT,
    build_direct_tfv_baseline_comparison,
)
from rtc.baselines import FORMAL_FIXED_BASELINE_IDS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposed-metadata", required=True)
    parser.add_argument("--baseline-panel", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    args = parser.parse_args()

    panel_path = Path(args.baseline_panel).resolve()
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    if not isinstance(panel, dict):
        raise ValueError("baseline panel must be a JSON object")
    if panel.get("contract") != CURRENT_SIX_BASELINE_DEVELOPMENT_CONTRACT:
        raise ValueError("comparison requires the current six-baseline Development panel")
    if panel.get("baseline_provenance_verified_all") is not True:
        raise ValueError("baseline panel is not provenance verified")
    rows = panel.get("rows")
    if not isinstance(rows, list):
        raise ValueError("baseline panel lacks rows")
    metadata_by_strategy = {
        str(row["strategy"]): str(row["metadata_path"])
        for row in rows
        if isinstance(row, dict) and row.get("strategy") in FORMAL_FIXED_BASELINE_IDS
    }
    if set(metadata_by_strategy) != set(FORMAL_FIXED_BASELINE_IDS):
        raise ValueError("baseline panel does not contain exactly the six fixed strategies")

    payload = build_direct_tfv_baseline_comparison(
        proposed_metadata=args.proposed_metadata,
        baseline_metadata_by_strategy=metadata_by_strategy,
    )
    payload["event_id"] = str(panel.get("event_id", ""))
    if not payload["event_id"]:
        raise ValueError("baseline panel lacks event_id")
    payload["baseline_panel_path"] = str(panel_path)

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table_rows = payload["rows"]
    assert isinstance(table_rows, list) and table_rows
    fields: list[str] = []
    for row in table_rows:
        assert isinstance(row, dict)
        for key in row:
            if key not in fields:
                fields.append(key)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table_rows)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
