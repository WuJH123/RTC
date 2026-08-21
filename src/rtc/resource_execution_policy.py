"""Current Project7 execution policy for machine-resource telemetry.

Resource observations are telemetry only.  The paper-facing workflow must not pause, stop, skip a
rainfall group, or change scientific behavior because a host-memory, paging, GPU-memory, or GPU-
utilization threshold was crossed.  The operating system / CUDA runtime / PySWMM may still raise a
real allocation or simulation exception; those are ordinary hard failures and are not converted into
pre-emptive resource guards here.
"""
from __future__ import annotations

from dataclasses import dataclass


PROJECT7_RESOURCE_EXECUTION_POLICY = "PROJECT7_RESOURCE_TELEMETRY_ONLY_NO_PREEMPTIVE_STOP_V1"
RESOURCE_TRIGGERED_EARLY_STOP_ENABLED = False
RESOURCE_TELEMETRY_CAN_CHANGE_SCIENTIFIC_EXECUTION = False


@dataclass(frozen=True)
class ResourceTelemetry:
    """Optional runtime observations retained for diagnostics only."""

    free_ram_gb: float | None = None
    pages_per_second: float | None = None
    pagefile_usage_percent: float | None = None
    gpu_memory_used_mb: float | None = None
    gpu_memory_free_mb: float | None = None
    gpu_utilization_percent: float | None = None


def resource_telemetry_requires_stop(_telemetry: ResourceTelemetry | None = None) -> bool:
    """Return False for every telemetry value by contract.

    Callers may record the supplied values, but they must not use them to interrupt an authoritative
    parent/query, skip a planned rainfall group, or alter the frozen Project7 policy-return workflow.
    Actual exceptions such as CUDA OOM, host allocation failure, or PySWMM errors continue to surface
    through their native exception paths.
    """

    return False
