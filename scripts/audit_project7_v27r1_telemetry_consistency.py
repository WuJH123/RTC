"""Audit Direct-TFV telemetry against the numerical target command written to SWMM.

The V27 Benchmark exposed a reporting-only adapter bug: the portfolio layer looked for a field that
was stripped by the runtime wrapper, so every decision was labelled HOLD although 144 commands
actually changed.  This read-only audit treats the target command as authoritative and checks that
source/action-class telemetry agrees with the numerical setting change.  It is intentionally not a
scientific performance gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import numpy as np


def _rows(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            values.append(value)
    return values


def _vector(value: Any) -> np.ndarray | None:
    try:
        out = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if out.shape != (109,) or not np.isfinite(out).all():
        return None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tolerance", type=float, default=1.0e-7)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    total = 0
    actual_actions = 0
    logged_actions = 0
    class_mismatches = 0
    source_mismatches = 0
    missing_reference = 0
    by_file: list[dict[str, Any]] = []
    selected_sources: Counter[str] = Counter()
    for path in sorted(root.rglob("*.decisions.jsonl")):
        rows = _rows(path)
        file_total = file_actual = file_class_bad = file_source_bad = 0
        previous: np.ndarray | None = None
        for row in rows:
            diagnostics = row.get("diagnostics") if isinstance(row.get("diagnostics"), dict) else {}
            command = _vector(row.get("settings"))
            if command is None:
                # Closed-loop JSON stores settings as an actuator-id mapping.  Preserve stable mapping
                # order when all 109 values are present.
                settings = row.get("settings")
                if isinstance(settings, dict) and len(settings) == 109:
                    try:
                        command = np.asarray([float(value) for value in settings.values()], dtype=np.float64)
                    except (TypeError, ValueError):
                        command = None
            if command is None:
                continue
            if previous is None:
                previous = _vector(diagnostics.get("hold_reference_settings"))
            if previous is None:
                missing_reference += 1
                previous = command.copy()
            delta_l1 = float(np.abs(command - previous).sum())
            actual = delta_l1 > float(args.tolerance)
            logged_class = str(diagnostics.get("calibrated_runtime_action_class", "")).upper()
            logged = logged_class == "ACTION"
            source = str(row.get("source", ""))
            source_action = source == "MPC_DIRECT_TFV_RECEDING"
            selected = str(diagnostics.get("policy_return_portfolio_selected_source", ""))
            if selected:
                selected_sources[selected] += 1
            total += 1
            file_total += 1
            actual_actions += int(actual)
            file_actual += int(actual)
            logged_actions += int(logged)
            if logged != actual:
                class_mismatches += 1
                file_class_bad += 1
            if source_action != actual:
                source_mismatches += 1
                file_source_bad += 1
            previous = command.copy()
        by_file.append(
            {
                "path": str(path),
                "decision_count": file_total,
                "actual_action_count": file_actual,
                "class_mismatch_count": file_class_bad,
                "source_mismatch_count": file_source_bad,
            }
        )

    if total == 0:
        raise ValueError("no usable Direct-TFV decisions were found")
    payload = {
        "contract": "PROJECT7_V27R1_TARGET_COMMAND_TELEMETRY_CONSISTENCY_AUDIT_V1",
        "decision_count": total,
        "actual_action_count": actual_actions,
        "actual_hold_count": total - actual_actions,
        "logged_action_count": logged_actions,
        "action_class_mismatch_count": class_mismatches,
        "action_class_mismatch_fraction": class_mismatches / total,
        "top_level_source_mismatch_count": source_mismatches,
        "top_level_source_mismatch_fraction": source_mismatches / total,
        "missing_initial_reference_count": missing_reference,
        "selected_source_counts": dict(sorted(selected_sources.items())),
        "files": by_file,
        "settings_are_authoritative_for_this_audit": True,
        "audit_is_reporting_only": True,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(out), **payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
