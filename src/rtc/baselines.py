from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BaselineDefinition:
    baseline_id: str
    description: str
    python_override: bool
    native_controls_enabled: bool


BASELINES = {
    "proposed": BaselineDefinition(
        "proposed",
        "Sparse-state + differentiable world-model + site-safe continuous MPC",
        True,
        True,
    ),
    "native_rules": BaselineDefinition(
        "native_rules",
        "Frozen SWMM native [CONTROLS], with no Python action overrides",
        False,
        True,
    ),
    "passive_no_rtc": BaselineDefinition(
        "passive_no_rtc",
        "Native RTC [CONTROLS] removed; retain frozen physical network/default device semantics",
        False,
        False,
    ),
    "hold": BaselineDefinition(
        "hold",
        "Hold the actuator readback observed at the evaluation start/checkpoint",
        True,
        True,
    ),
    "all_open": BaselineDefinition(
        "all_open",
        "Diagnostic only: command every eligible continuous setting to 1.0",
        True,
        True,
    ),
    "all_closed": BaselineDefinition(
        "all_closed",
        "Diagnostic only: command every eligible continuous setting to 0.0",
        True,
        True,
    ),
}


def write_passive_no_rtc_inp(source: str | Path, destination: str | Path) -> Path:
    """Create the passive/no-RTC baseline by removing only the [CONTROLS] section.

    This intentionally does *not* set every actuator to 1.0 or 0.0. Pump curves,
    startup/shutoff depths, network geometry, initial settings and all non-RTC physics
    remain exactly as encoded in the frozen INP.
    """

    src = Path(source)
    dst = Path(destination)
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    output: list[str] = []
    in_controls = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].strip().upper()
            in_controls = section == "CONTROLS"
            if in_controls:
                output.append("[CONTROLS]\n")
                output.append("; disabled for PASSIVE_NO_RTC scientific baseline\n")
                continue
        if not in_controls:
            output.append(line)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("".join(output), encoding="utf-8")
    return dst
