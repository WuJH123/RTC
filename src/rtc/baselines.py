from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .closed_loop import CausalObservation, ControllerAction
from .inp_runtime import build_runtime_inp
from .rule_baselines import AutoRBCController, EFDController, rule_baseline_sensor_nodes


@dataclass(frozen=True)
class BaselineDefinition:
    baseline_id: str
    description: str
    python_override: bool
    native_controls_enabled: bool
    formal_comparator: bool = True


BASELINES = {
    "proposed": BaselineDefinition(
        "proposed",
        "Sparse causal Step1 state reconstruction plus learned 109-actuator pairwise delta-TFV value and support-aware receding control",
        True,
        False,
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
        "No-supervisory-RTC: [CONTROLS] disabled, no Python writes; intrinsic pump startup/shutoff logic remains physical/local behavior",
        False,
        False,
    ),
    "auto_rbc": BaselineDefinition(
        "auto_rbc",
        "Automatically parameterized causal local rule-based control using actuator-adjacent filling, downstream congestion and type-aware SWMM release-setting semantics",
        True,
        False,
    ),
    "efd": BaselineDefinition(
        "efd",
        "Storage Equal Filling Degree comparator using causal storage volume/capacity and type-aware outgoing-release settings",
        True,
        False,
    ),
    "all_open": BaselineDefinition(
        "all_open",
        "Legacy-name diagnostic extreme: command every eligible SWMM SETTING to its numerical maximum 1.0. This is ALL-MAX-SETTING, not a universal physical all-open/max-release state because WEIR SETTING has different semantics.",
        True,
        False,
        False,
    ),
    "all_closed": BaselineDefinition(
        "all_closed",
        "Legacy-name diagnostic extreme: command every eligible SWMM SETTING to its numerical minimum 0.0. This is ALL-MIN-SETTING, not a universal physical all-closed/min-release state because actuator SETTING semantics differ by type.",
        True,
        False,
        False,
    ),
    "hold": BaselineDefinition(
        "hold",
        "Debug-only frozen-readback policy. Excluded from the formal competitive comparison because on a controls-disabled base it can collapse to No-control.",
        True,
        False,
        False,
    ),
}

# Keep the six-strategy evidence panel for compatibility and transparency. Publication claims use
# COMPETITIVE_BASELINE_IDS; numerical setting extremes remain visible diagnostic evidence.
FORMAL_FIXED_BASELINE_IDS = (
    "no_control",
    "internal_rtc",
    "auto_rbc",
    "efd",
    "all_open",
    "all_closed",
)
COMPETITIVE_BASELINE_IDS = (
    "no_control",
    "internal_rtc",
    "auto_rbc",
    "efd",
)
DIAGNOSTIC_FIXED_BASELINE_IDS = ("all_open", "all_closed", "hold")
SUPPORTED_FIXED_BASELINE_IDS = FORMAL_FIXED_BASELINE_IDS + ("hold",)
FIXED_BASELINE_IDS = FORMAL_FIXED_BASELINE_IDS

LEGACY_BASELINE_ALIASES = {
    "native_rules": "internal_rtc",
    "passive_no_rtc": "no_control",
    "auto_rrc": "auto_rbc",
    "Auto-RRC": "auto_rbc",
    "Auto-RBC": "auto_rbc",
    "EFD": "efd",
}


def canonical_baseline_id(value: str) -> str:
    key = str(value).strip()
    return LEGACY_BASELINE_ALIASES.get(key, key.lower())


def write_no_control_inp(
    source: str | Path,
    destination: str | Path,
    *,
    swmm_threads: int | None = None,
) -> Path:
    """Create the scientific No-supervisory-RTC INP.

    This removes user-defined supervisory ``[CONTROLS]`` only. It intentionally preserves
    pump curves, initial pump status, intrinsic [PUMPS] Startup/Shutoff depths, storage
    geometry, regulator physics and all forcing. No Python control writes are made. It is
    therefore neither All-max-setting nor All-min-setting and remains an operationally meaningful
    reference representing the network without supervisory RTC.
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


def constant_setting_controller(
    value: float, source: str
) -> Callable[[CausalObservation], ControllerAction]:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError("constant baseline setting must be inside [0,1]")

    def controller(obs: CausalObservation) -> ControllerAction:
        return ControllerAction(
            settings={aid: float(value) for aid in obs.actuator_ids},
            source=source,
        )

    return controller


def frozen_hold_controller() -> Callable[[CausalObservation], ControllerAction]:
    frozen: np.ndarray | None = None

    def controller(obs: CausalObservation) -> ControllerAction:
        nonlocal frozen
        if frozen is None:
            frozen = np.asarray(obs.actuator_current_setting, dtype=float).copy()
        return ControllerAction(
            settings=dict(zip(obs.actuator_ids, frozen, strict=True)),
            source="FROZEN_HOLD",
        )

    return controller


def baseline_sensor_nodes(strategy: str, inp_path: str | Path) -> tuple[str, ...]:
    strategy = canonical_baseline_id(strategy)
    if strategy in {"auto_rbc", "efd"}:
        return rule_baseline_sensor_nodes(strategy, inp_path)
    return ()


def fixed_baseline_controller(
    strategy: str,
    *,
    inp_path: str | Path | None = None,
    max_delta_per_update: float | None = None,
):
    """Return the deterministic Python controller for a fixed/reference strategy."""
    strategy = canonical_baseline_id(strategy)
    if strategy in {"no_control", "internal_rtc"}:
        return None
    if strategy == "hold":
        return frozen_hold_controller()
    if strategy == "all_open":
        return constant_setting_controller(1.0, "ALL_OPEN")
    if strategy == "all_closed":
        return constant_setting_controller(0.0, "ALL_CLOSED")
    if strategy == "auto_rbc":
        if inp_path is None:
            raise ValueError("Auto-RBC requires the source event INP")
        return AutoRBCController(
            inp_path,
            max_delta_per_update=max_delta_per_update,
        )
    if strategy == "efd":
        if inp_path is None:
            raise ValueError("EFD requires the source event INP")
        return EFDController(
            inp_path,
            max_delta_per_update=max_delta_per_update,
        )
    raise ValueError(f"not a supported fixed baseline strategy: {strategy}")
