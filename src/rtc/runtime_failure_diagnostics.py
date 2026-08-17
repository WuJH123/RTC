from __future__ import annotations

from typing import Any, Mapping, Sequence


DIRECT_TFV_RUNTIME_FAILURE_DIAGNOSTIC_CONTRACT = (
    "PROJECT7_DIRECT_TFV_RUNTIME_FAILURE_DIAGNOSTIC_V1"
)
_ACCELERATOR_MARKERS = (
    "cuda error",
    "acceleratorerror",
    "cuda out of memory",
    "cublas",
    "cudnn",
    "device-side assert",
)


def classify_fallback_failure(row: Mapping[str, Any]) -> str:
    source = str(row.get("source", ""))
    if not source.startswith("FALLBACK_"):
        return "not_fallback"
    diagnostics = row.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return "unknown_runtime_failure"
    error_type = str(diagnostics.get("error_type", "")).lower()
    error = str(diagnostics.get("error", "")).lower()
    text = f"{error_type} {error}"
    if any(marker in text for marker in _ACCELERATOR_MARKERS):
        return "accelerator_environment"
    if source == "FALLBACK_RUNTIME_ERROR":
        return "controller_runtime"
    return "policy_or_guard_fallback"


def summarize_runtime_failures(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {
        "accelerator_environment": 0,
        "controller_runtime": 0,
        "policy_or_guard_fallback": 0,
        "unknown_runtime_failure": 0,
    }
    elapsed_by_domain: dict[str, list[int]] = {key: [] for key in counts}
    for row in rows:
        domain = classify_fallback_failure(row)
        if domain not in counts:
            continue
        counts[domain] += 1
        elapsed_by_domain[domain].append(int(row.get("elapsed_seconds", -1)))
    total = sum(counts.values())
    if total == 0:
        classification = "NO_RUNTIME_FALLBACK"
    elif counts["accelerator_environment"] == total:
        classification = "ACCELERATOR_ENVIRONMENT_ONLY"
    elif counts["controller_runtime"] > 0 or counts["unknown_runtime_failure"] > 0:
        classification = "CONTROLLER_RUNTIME_FAILURE_PRESENT"
    else:
        classification = "POLICY_OR_GUARD_FALLBACK_PRESENT"
    ranges = {
        domain: {
            "first_elapsed_seconds": min(values) if values else None,
            "last_elapsed_seconds": max(values) if values else None,
        }
        for domain, values in elapsed_by_domain.items()
    }
    return {
        "contract": DIRECT_TFV_RUNTIME_FAILURE_DIAGNOSTIC_CONTRACT,
        "fallback_count": total,
        "counts_by_domain": counts,
        "elapsed_range_by_domain": ranges,
        "classification": classification,
        "scientific_policy_failure_inferred_from_accelerator_error": False,
    }


__all__ = [
    "DIRECT_TFV_RUNTIME_FAILURE_DIAGNOSTIC_CONTRACT",
    "classify_fallback_failure",
    "summarize_runtime_failures",
]
