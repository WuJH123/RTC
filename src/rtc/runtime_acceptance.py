from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .causal_timing import timing_from_controller_config


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path: str | Path) -> dict[str, object]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON must be an object: {path}")
    return value


def _decisions(metadata_path: str | Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    meta_path = Path(metadata_path)
    meta = _json(meta_path)
    decision_name = meta.get("decision_file")
    if not decision_name:
        raise ValueError(f"closed-loop metadata lacks decision_file: {metadata_path}")
    decision_path = meta_path.parent / str(decision_name)
    rows: list[dict[str, object]] = []
    for raw in decision_path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            item = json.loads(raw)
            if not isinstance(item, dict):
                raise ValueError("decision log row must be a JSON object")
            rows.append(item)
    return meta, rows


def build_runtime_acceptance(
    *,
    run_index_path: str | Path,
    controller_config_path: str | Path,
) -> dict[str, object]:
    config = _json(controller_config_path)
    timing = timing_from_controller_config(config)
    timing.validate(require_full_history_before_first_control=True)
    controller_section = config.get("controller")
    if not isinstance(controller_section, dict):
        raise ValueError("controller config lacks controller section")
    raw_budget = controller_section.get("decision_runtime_budget_seconds")
    if raw_budget is None:
        raise ValueError("Formal runtime acceptance requires decision_runtime_budget_seconds")
    budget = float(raw_budget)
    if not 0 < budget < timing.control_update_seconds:
        raise ValueError("decision runtime budget must be positive and smaller than control interval")

    index = pd.read_csv(run_index_path)
    required = {"strategy", "metadata_path"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"development run index missing columns: {missing}")
    proposed = index[index["strategy"].astype(str) == "proposed"].copy()
    if proposed.empty:
        raise ValueError("development run index contains no Proposed runs")

    runtimes: list[float] = []
    source_counts: dict[str, int] = {}
    grid_violations = 0
    first_decision_violations = 0
    missing_runtime_diagnostics = 0
    decisions_total = 0
    event_rows: list[dict[str, object]] = []
    start = timing.control_start_seconds
    update = timing.control_update_seconds

    for _, item in proposed.iterrows():
        meta, rows = _decisions(str(item["metadata_path"]))
        if int(meta.get("control_update_seconds", -1)) != update:
            raise ValueError("development run control_update differs from frozen controller config")
        if int(meta.get("observation_update_seconds", -1)) != timing.model_step_seconds:
            raise ValueError("development run observation cadence differs from model step")
        if int(meta.get("control_start_minutes", -1)) * 60 != start:
            raise ValueError("development run control start differs from frozen controller config")
        if int(meta.get("initial_observation_elapsed_seconds", -1)) != 0:
            raise ValueError("development run lacks the causal t=0 initial observation")
        if not rows:
            raise ValueError(f"Proposed development run has no decisions: {item['metadata_path']}")
        elapsed = [int(row["elapsed_seconds"]) for row in rows]
        if elapsed[0] != start:
            first_decision_violations += 1
        event_grid = sum(1 for t in elapsed if t < start or (t - start) % update != 0)
        grid_violations += event_grid
        event_runtime: list[float] = []
        for row in rows:
            source = str(row.get("source", ""))
            source_counts[source] = source_counts.get(source, 0) + 1
            diagnostics = row.get("diagnostics")
            if not isinstance(diagnostics, dict) or "decision_runtime_seconds" not in diagnostics:
                # Warm-up/readback can return before optimization; these are separately fatal
                # below. All normal MPC/optimizer decisions must report wall-clock runtime.
                if source not in {"FALLBACK_HISTORY_WARMUP", "FALLBACK_READBACK"}:
                    missing_runtime_diagnostics += 1
                continue
            runtime = float(diagnostics["decision_runtime_seconds"])
            if not np.isfinite(runtime) or runtime < 0:
                raise ValueError("invalid decision_runtime_seconds in development log")
            runtimes.append(runtime)
            event_runtime.append(runtime)
        decisions_total += len(rows)
        event_rows.append(
            {
                "event_id": str(item.get("event_id", "")),
                "decisions": len(rows),
                "grid_violations": event_grid,
                "max_runtime_seconds": max(event_runtime) if event_runtime else np.nan,
            }
        )

    fatal_sources = {
        "FALLBACK_HISTORY_WARMUP",
        "FALLBACK_READBACK",
        "FALLBACK_RUNTIME_ERROR",
        "FALLBACK_COMPUTE_DEADLINE",
    }
    fatal_count = sum(source_counts.get(name, 0) for name in fatal_sources)
    max_runtime = float(max(runtimes)) if runtimes else np.inf
    p95 = float(np.quantile(runtimes, 0.95)) if runtimes else np.inf
    p99 = float(np.quantile(runtimes, 0.99)) if runtimes else np.inf
    passed = bool(
        grid_violations == 0
        and first_decision_violations == 0
        and missing_runtime_diagnostics == 0
        and fatal_count == 0
        and max_runtime <= budget
    )
    return {
        "contract": "DEVELOPMENT_REALTIME_EXECUTION_ACCEPTANCE_V1",
        "passed": passed,
        "controller_config_sha256": _sha(controller_config_path),
        "run_index_sha256": _sha(run_index_path),
        "timing": timing.as_dict(),
        "decision_runtime_budget_seconds": budget,
        "metrics": {
            "proposed_events": int(len(proposed)),
            "decisions": int(decisions_total),
            "decision_runtime_p95_seconds": p95,
            "decision_runtime_p99_seconds": p99,
            "decision_runtime_max_seconds": max_runtime,
            "control_grid_violations": int(grid_violations),
            "first_decision_violations": int(first_decision_violations),
            "missing_runtime_diagnostics": int(missing_runtime_diagnostics),
            "fatal_runtime_fallbacks": int(fatal_count),
            "optimizer_candidate_fallbacks": int(source_counts.get("FALLBACK", 0)),
        },
        "decision_source_counts": source_counts,
        "fatal_sources": sorted(fatal_sources),
        "events": event_rows,
        "interpretation": "Policy Lock requires real development decisions on the frozen event clock, complete causal history, successful write/readback, no runtime/deadline fallback, and wall-clock compute within budget.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed real-time execution gate for Proposed development closed-loop runs"
    )
    parser.add_argument("--run-index", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = build_runtime_acceptance(
        run_index_path=args.run_index,
        controller_config_path=args.config,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["passed"] is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
