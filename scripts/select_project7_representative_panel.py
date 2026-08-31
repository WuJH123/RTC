"""Select a compact, outcome-independent Project7 evaluation panel."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rtc.project7_representative_panel import (
    DEFAULT_REPRESENTATIVE_EVENT_COUNT,
    select_representative_panel,
    selected_rows,
)


def _read_training_ids(path: str | None, event_id_column: str) -> tuple[str, ...]:
    if not path:
        return ()
    table = pd.read_csv(path)
    if event_id_column not in table.columns:
        raise ValueError(f"training manifest is missing {event_id_column!r}")
    return tuple(table[event_id_column].dropna().astype(str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV containing the successfully completed event universe")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-events", type=int, default=DEFAULT_REPRESENTATIVE_EVENT_COUNT)
    parser.add_argument("--event-id-column", default="event_id")
    parser.add_argument("--training-events", help="optional CSV used only to assert zero train/evaluation event overlap")
    args = parser.parse_args()

    rows = pd.read_csv(args.input)
    training_ids = _read_training_ids(args.training_events, args.event_id_column)
    panel = select_representative_panel(
        rows,
        target_event_count=args.target_events,
        event_id_column=args.event_id_column,
        training_event_ids=training_ids,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected = selected_rows(rows, panel, event_id_column=args.event_id_column)
    selected.to_csv(out / "PROJECT7_REPRESENTATIVE_PANEL.csv", index=False)
    payload = {
        "contract": panel.contract,
        "input_event_count": panel.input_event_count,
        "target_event_count": panel.target_event_count,
        "selected_event_ids": list(panel.selected_event_ids),
        "family_counts": panel.family_counts,
        "descriptor_columns": list(panel.descriptor_columns),
        "selection_uses_hydraulic_outcomes": False,
        "selection_uses_strategy_performance": False,
        "training_evaluation_overlap_checked": bool(args.training_events),
        "full_panel_remains_optional_sensitivity_analysis": True,
    }
    (out / "PROJECT7_REPRESENTATIVE_PANEL.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
