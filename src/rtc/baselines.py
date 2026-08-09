from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .inp_runtime import build_runtime_inp


@dataclass(frozen=True)
class BaselineDefinition:
    baseline_id: str
    description: str
    python_override: bool
    native_controls_enabled: bool


BASELINES = {
    "proposed": BaselineDefinition(
        "proposed",
        "Sparse-state + differentiable TFV-first continuous MPC on the controls-disabled physical base",
        True,
        False,
    ),
    "internal_rtc": BaselineDefinition(
        "internal_rtc",
        "Frozen SWMM native [CONTROLS], with no Python actuator overrides",
        False,
        True,
    ),
    "no_control": BaselineDefinition(
        "no_control",
        "Same physical network/forcing with native [CONTROLS] disabled and no Python writes",
        False,
        False,
    ),
    "hold": BaselineDefinition(
        "hold",
        "Controls-disabled base; hold actuator readback observed at the first control decision",
        True,
        False,
    ),
    "all_open": BaselineDefinition(
        "all_open",
        "Diagnostic only on controls-disabled base: command every eligible setting to 1.0",
        True,
        False,
    ),
    "all_closed": BaselineDefinition(
        "all_closed",
        "Diagnostic only on controls-disabled base: command every eligible setting to 0.0",
        True,
        False,
    ),
}

# Backward-compatible aliases for pre-audit manifests. New Formal evidence must use the
# explicit names above so No-control can never be confused with Internal-RTC.
LEGACY_BASELINE_ALIASES = {
    "native_rules": "internal_rtc",
    "passive_no_rtc": "no_control",
}


def canonical_baseline_id(value: str) -> str:
    key = str(value).strip()
    return LEGACY_BASELINE_ALIASES.get(key, key)


def write_no_control_inp(
    source: str | Path,
    destination: str | Path,
    *,
    swmm_threads: int | None = None,
) -> Path:
    """Create the scientific No-control INP.

    No-control means: identical physical network and forcing, native ``[CONTROLS]``
    disabled, and no Python control writes. It is deliberately *not* all-open/all-closed.
    Pump curves, initial status, storage geometry and all non-policy physics are retained.
    """

    result = build_runtime_inp(
        source,
        destination,
        native_controls=False,
        swmm_threads=swmm_threads,
    )
    return Path(result.runtime_path)


def write_passive_no_rtc_inp(source: str | Path, destination: str | Path) -> Path:
    """Deprecated compatibility alias for :func:`write_no_control_inp`."""

    return write_no_control_inp(source, destination)
