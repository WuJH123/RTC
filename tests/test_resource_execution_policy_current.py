from __future__ import annotations

from rtc.resource_execution_policy import (
    PROJECT7_RESOURCE_EXECUTION_POLICY,
    RESOURCE_TELEMETRY_CAN_CHANGE_SCIENTIFIC_EXECUTION,
    RESOURCE_TRIGGERED_EARLY_STOP_ENABLED,
    ResourceTelemetry,
    resource_telemetry_requires_stop,
)


def test_current_resource_policy_never_preemptively_stops_for_telemetry() -> None:
    assert PROJECT7_RESOURCE_EXECUTION_POLICY == (
        "PROJECT7_RESOURCE_TELEMETRY_ONLY_NO_PREEMPTIVE_STOP_V1"
    )
    assert RESOURCE_TRIGGERED_EARLY_STOP_ENABLED is False
    assert RESOURCE_TELEMETRY_CAN_CHANGE_SCIENTIFIC_EXECUTION is False

    extreme = ResourceTelemetry(
        free_ram_gb=0.0,
        pages_per_second=1.0e9,
        pagefile_usage_percent=100.0,
        gpu_memory_used_mb=8192.0,
        gpu_memory_free_mb=0.0,
        gpu_utilization_percent=100.0,
    )
    assert resource_telemetry_requires_stop(extreme) is False
    assert resource_telemetry_requires_stop(None) is False
