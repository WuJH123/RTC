"""Evaluate the fail-closed Project7 V30 pre-Policy-Lock scientific gate from a CSV panel."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rtc.project7_v30_scientific_gate import (
    ScientificPanelRow,
    evaluate_v30_scientific_gate,
)


def _bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "pass", "passed"}:
        return True
    if normalized in {"0", "false", "no", "fail", "failed"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def _load(path: Path) -> list[ScientificPanelRow]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "event_id",
            "strategy",
            "partition",
            "source_inp_sha256",
            "tfv_m3",
            "pfv_m3",
            "engineering_pass",
            "action_decisions",
            "decision_count",
            "ever_changed_actuator_count",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError("scientific gate CSV missing columns: " + ",".join(sorted(missing)))
        rows: list[ScientificPanelRow] = []
        for raw in reader:
            rows.append(
                ScientificPanelRow(
                    event_id=str(raw["event_id"]).strip(),
                    strategy=str(raw["strategy"]).strip(),
                    partition=str(raw["partition"]).strip(),
                    source_inp_sha256=str(raw["source_inp_sha256"]).strip(),
                    tfv_m3=float(raw["tfv_m3"]),
                    pfv_m3=float(raw["pfv_m3"]),
                    engineering_pass=_bool(str(raw["engineering_pass"])),
                    action_decisions=int(raw["action_decisions"]),
                    decision_count=int(raw["decision_count"]),
                    ever_changed_actuator_count=int(raw["ever_changed_actuator_count"]),
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    panel = Path(args.panel_csv).resolve()
    if not panel.is_file():
        raise FileNotFoundError(panel)
    result = evaluate_v30_scientific_gate(_load(panel))
    payload = {
        "contract": result.contract,
        "passed": result.passed,
        "event_count": result.event_count,
        "issues": list(result.issues),
        "diagnostics": result.diagnostics,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not result.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
