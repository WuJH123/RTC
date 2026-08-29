"""Compile publication-facing V29 regime/model and Benchmark5 mechanism evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _recursive_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, dict):
        for name, child in value.items():
            if str(name) == key:
                found.append(child)
            found.extend(_recursive_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_recursive_values(child, key))
    return found


def _first(value: Any, key: str, default: Any = None) -> Any:
    values = _recursive_values(value, key)
    return values[0] if values else default


def _decision_files(root: Path) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for metadata_path in sorted(root.rglob("*.json")):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict) or payload.get("strategy") != "proposed":
            continue
        requested = payload.get("decision_path")
        if not requested:
            continue
        path = Path(str(requested))
        if not path.is_file():
            matches = list(metadata_path.parent.glob("*decision*.jsonl"))
            if len(matches) == 1:
                path = matches[0]
            else:
                continue
        result.append((metadata_path.parent.name, path.resolve()))
    dedup: dict[tuple[str, str], tuple[str, Path]] = {}
    for item in result:
        dedup[(item[0], str(item[1]))] = item
    return list(dedup.values())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-report", required=True)
    parser.add_argument("--benchmark-root", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    model_report = json.loads(Path(args.model_report).read_text(encoding="utf-8"))
    comparison = json.loads(Path(args.comparison).read_text(encoding="utf-8"))
    root = Path(args.benchmark_root).resolve()
    events: dict[str, Any] = {}
    all_stress: list[float] = []
    all_actions = 0
    all_holds = 0
    utility_present = 0
    utility_selected = 0
    utility_retained_fraction: list[float] = []

    for event_id, path in _decision_files(root):
        rows = _read_jsonl(path)
        stress: list[float] = []
        actions = 0
        holds = 0
        sources: dict[str, int] = {}
        event_utility_present = 0
        event_utility_selected = 0
        event_utility_fraction: list[float] = []
        for row in rows:
            action_class = str(_first(row, "v29_action_class", ""))
            if action_class not in {"ACTION", "HOLD"}:
                continue
            if action_class == "ACTION":
                actions += 1
            else:
                holds += 1
            source = str(_first(row, "v29_selected_source", "HOLD"))
            sources[source] = sources.get(source, 0) + 1
            candidates = _first(row, "v29_candidate_telemetry", [])
            if isinstance(candidates, list):
                local_stress = [
                    float(candidate.get("network_stress_q75"))
                    for candidate in candidates
                    if isinstance(candidate, dict)
                    and candidate.get("network_stress_q75") is not None
                ]
                if local_stress:
                    stress.append(float(np.median(local_stress)))
            present = bool(_first(row, "v29_utility_shadow_present", False))
            selected = bool(_first(row, "v29_utility_shadow_selected", False))
            event_utility_present += int(present)
            event_utility_selected += int(selected)
            fraction = _first(row, "v29_utility_shadow_retained_utility_fraction", None)
            if fraction is not None:
                event_utility_fraction.append(float(fraction))

        total = actions + holds
        events[event_id] = {
            "decision_count": int(total),
            "action_count": int(actions),
            "hold_count": int(holds),
            "action_fraction": float(actions / total) if total else 0.0,
            "stress_min": float(np.min(stress)) if stress else None,
            "stress_median": float(np.median(stress)) if stress else None,
            "stress_max": float(np.max(stress)) if stress else None,
            "selected_source_counts": dict(sorted(sources.items())),
            "utility_shadow_present_count": int(event_utility_present),
            "utility_shadow_selected_count": int(event_utility_selected),
            "utility_shadow_retained_utility_fraction_mean": (
                float(np.mean(event_utility_fraction)) if event_utility_fraction else None
            ),
        }
        all_stress.extend(stress)
        all_actions += actions
        all_holds += holds
        utility_present += event_utility_present
        utility_selected += event_utility_selected
        utility_retained_fraction.extend(event_utility_fraction)

    split_reports = model_report.get("split_reports", {})
    regime_summary = {
        split: payload.get("stress_regime_metrics", {})
        for split, payload in split_reports.items()
        if isinstance(payload, dict)
    }
    payload = {
        "contract": "PROJECT7_V29_REGIME_MECHANISM_PUBLICATION_AUDIT_V1",
        "model_contract": model_report.get("model_contract"),
        "selected_ridge": model_report.get("selected_ridge"),
        "selected_shrinkage": model_report.get("selected_shrinkage"),
        "validation_used_for_model_selection": model_report.get("validation_used_for_model_selection"),
        "test_used_for_model_selection": model_report.get("test_used_for_training_or_model_selection"),
        "return_period_used_as_feature": model_report.get("return_period_used_as_feature"),
        "event_duration_used_as_feature": model_report.get("event_duration_used_as_feature"),
        "split_stress_regime_metrics": regime_summary,
        "benchmark_event_mechanism": events,
        "benchmark_total_action_count": int(all_actions),
        "benchmark_total_hold_count": int(all_holds),
        "benchmark_action_fraction": float(all_actions / max(1, all_actions + all_holds)),
        "benchmark_stress_min": float(np.min(all_stress)) if all_stress else None,
        "benchmark_stress_median": float(np.median(all_stress)) if all_stress else None,
        "benchmark_stress_max": float(np.max(all_stress)) if all_stress else None,
        "utility_shadow_present_count": int(utility_present),
        "utility_shadow_selected_count": int(utility_selected),
        "utility_shadow_retained_utility_fraction_mean": (
            float(np.mean(utility_retained_fraction)) if utility_retained_fraction else None
        ),
        "benchmark_comparison": comparison,
        "scientific_note": (
            "Return period and event duration are reporting strata only. "
            "They are never policy inputs; regime adaptation is continuous in causal hydraulic state."
        ),
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
