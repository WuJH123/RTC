"""Summarize V27 decision diagnostics from closed-loop JSONL outputs.

This is a read-only Development audit.  It quantifies whether q95 contraction changes the value-model
preferred candidate, whether reporting-value clipping is frequent, and how often the Auto-RBC shadow
candidate is available/selected.  None of these diagnostics blocks execution.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterator


PREFIX = "V27_DECISION_AWARE|"


def _strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _strings(child)


def _diag(row: dict[str, Any]) -> dict[str, str] | None:
    for value in _strings(row):
        if not value.startswith(PREFIX):
            continue
        result: dict[str, str] = {}
        for token in value[len(PREFIX) :].split("|"):
            if "=" not in token:
                continue
            key, raw = token.split("=", 1)
            result[key] = raw
        return result
    return None


def _as_bool(value: str | None) -> bool:
    return str(value).strip().lower() == "true"


def _as_int(value: str | None) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    diagnostics: list[dict[str, str]] = []
    files: list[str] = []
    for path in sorted(root.rglob("*.jsonl")):
        found = False
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            parsed = _diag(row)
            if parsed is None:
                continue
            diagnostics.append(parsed)
            found = True
        if found:
            files.append(str(path))
    if not diagnostics:
        raise ValueError("no V27 decision diagnostics were found")

    transitions: Counter[str] = Counter()
    supported_sources: Counter[str] = Counter()
    raw_sources: Counter[str] = Counter()
    for row in diagnostics:
        raw = row.get("raw_best", "")
        supported = row.get("supported_best", "")
        raw_sources[raw] += 1
        supported_sources[supported] += 1
        transitions[f"{raw}->{supported}"] += 1
    payload = {
        "contract": "PROJECT7_V27_DECISION_DIAGNOSTIC_AUDIT_V1",
        "decision_count": len(diagnostics),
        "source_jsonl_files": files,
        "report_clip_hit_candidate_total": sum(_as_int(row.get("clip_hits")) for row in diagnostics),
        "raw_report_clip_hit_candidate_total": sum(_as_int(row.get("raw_clip_hits")) for row in diagnostics),
        "q95_binding_candidate_total": sum(_as_int(row.get("q95_binding_candidates")) for row in diagnostics),
        "q95_selection_changed_decision_count": sum(_as_bool(row.get("q95_selection_changed")) for row in diagnostics),
        "q95_selection_changed_fraction": sum(_as_bool(row.get("q95_selection_changed")) for row in diagnostics) / len(diagnostics),
        "auto_rbc_shadow_present_decision_count": sum(_as_bool(row.get("shadow_present")) for row in diagnostics),
        "auto_rbc_shadow_selected_decision_count": sum(_as_bool(row.get("shadow_selected")) for row in diagnostics),
        "auto_rbc_shadow_duplicate_decision_count": sum(_as_bool(row.get("shadow_duplicate")) for row in diagnostics),
        "raw_best_source_counts": dict(sorted(raw_sources.items())),
        "supported_best_source_counts": dict(sorted(supported_sources.items())),
        "q95_source_transition_counts": dict(sorted(transitions.items())),
        "diagnostics_are_reporting_only": True,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"audit": str(out), **payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
