"""Post-run real-time acceptance evidence for Project7 V128 authoritative control."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

V128_RUNTIME_ACCEPTANCE_CONTRACT = (
    "PROJECT7_V128_AUTHORITATIVE_600S_RUNTIME_ACCEPTANCE_V1"
)


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(values, q)) if values.size else float("nan")


def audit_v128_runtime_decisions(
    decision_path: str | Path,
    *,
    control_update_seconds: int = 600,
) -> dict[str, Any]:
    path = Path(decision_path)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("V128 runtime acceptance requires at least one decision")
    elapsed = np.asarray([int(row["elapsed_seconds"]) for row in rows], dtype=np.int64)
    if np.any(np.diff(elapsed) <= 0):
        raise ValueError("V128 decision timestamps are not strictly increasing")
    intervals = np.diff(elapsed)
    interval_pass = bool(
        intervals.size == 0 or np.all(intervals == int(control_update_seconds))
    )

    runtimes: list[float] = []
    optimizer_runtimes: list[float] = []
    deadline = 0
    fallback = 0
    score_execute_failures = 0
    continuity_failures = 0
    sources: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source", "UNKNOWN"))
        sources[source] = sources.get(source, 0) + 1
        if "FALLBACK" in source or source.startswith("RBC_SAFETY"):
            fallback += 1
        diagnostics = row.get("diagnostics") or {}
        runtime = diagnostics.get("decision_runtime_seconds")
        if runtime is not None:
            value = float(runtime)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("V128 decision runtime contains invalid values")
            runtimes.append(value)
        optimizer = diagnostics.get("optimizer_elapsed_seconds")
        if optimizer is not None:
            value = float(optimizer)
            if np.isfinite(value) and value >= 0.0:
                optimizer_runtimes.append(value)
        deadline += int(
            bool(
                diagnostics.get("optimizer_deadline_exceeded", False)
                or diagnostics.get("continuous_optimizer_deadline_exceeded", False)
            )
        )
        if diagnostics.get("score_equals_execute") is False:
            score_execute_failures += 1
        if diagnostics.get("score_equals_execute_under_engineering_envelope") is False:
            score_execute_failures += 1
        if diagnostics.get("continuity_guard_passed") is False:
            continuity_failures += 1

    runtime_array = np.asarray(runtimes, dtype=float)
    if runtime_array.size != len(rows):
        raise ValueError("V128 every authoritative decision must report wall-clock runtime")
    optimizer_array = np.asarray(optimizer_runtimes, dtype=float)
    max_runtime = float(runtime_array.max())
    hard_realtime_pass = bool(max_runtime < float(control_update_seconds))
    payload: dict[str, Any] = {
        "contract": V128_RUNTIME_ACCEPTANCE_CONTRACT,
        "passed": bool(
            hard_realtime_pass
            and interval_pass
            and score_execute_failures == 0
            and continuity_failures == 0
        ),
        "decision_count": len(rows),
        "control_update_seconds": int(control_update_seconds),
        "decision_interval_exact_600s": interval_pass,
        "decision_runtime_seconds": {
            "mean": float(runtime_array.mean()),
            "p50": _quantile(runtime_array, 0.50),
            "p95": _quantile(runtime_array, 0.95),
            "max": max_runtime,
        },
        "optimizer_runtime_seconds": {
            "count": int(optimizer_array.size),
            "mean": float(optimizer_array.mean()) if optimizer_array.size else None,
            "p50": _quantile(optimizer_array, 0.50) if optimizer_array.size else None,
            "p95": _quantile(optimizer_array, 0.95) if optimizer_array.size else None,
            "max": float(optimizer_array.max()) if optimizer_array.size else None,
        },
        "hard_realtime_max_lt_600s": hard_realtime_pass,
        "deadline_exceeded_decisions": int(deadline),
        "fallback_decisions": int(fallback),
        "score_execute_failures": int(score_execute_failures),
        "continuity_failures": int(continuity_failures),
        "sources": sources,
        "interpretation": (
            "passed proves the measured authoritative run completed every logged decision "
            "inside the 600-s control period; it is not a universal worst-case industrial "
            "hard-real-time guarantee on other hardware/events"
        ),
    }
    return payload


__all__ = ["V128_RUNTIME_ACCEPTANCE_CONTRACT", "audit_v128_runtime_decisions"]
