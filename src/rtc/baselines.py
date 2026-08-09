from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .closed_loop import CausalObservation, ControllerAction
from .inp_runtime import build_runtime_inp


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
        "Sparse-state + differentiable TFV-first continuous MPC on the controls-disabled physical base",
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
    "all_open": BaselineDefinition(
        "all_open",
        "Diagnostic extreme policy on controls-disabled base: command every eligible setting to 1.0 from the first common control decision",
        True,
        False,
    ),
    "all_closed": BaselineDefinition(
        "all_closed",
        "Diagnostic extreme policy on controls-disabled base: command every eligible setting to 0.0 from the first common control decision",
        True,
        False,
    ),
    "hold": BaselineDefinition(
        "hold",
        "Debug-only frozen-readback policy. Excluded from the Formal comparison matrix because on a controls-disabled base it can collapse to No-control.",
        True,
        False,
        False,
    ),
}

# Formal comparison matrix. Hold remains callable for engineering/debug experiments, but is
# deliberately excluded so the paper does not count a duplicate No-control-like strategy.
FORMAL_FIXED_BASELINE_IDS = ("no_control", "internal_rtc", "all_open", "all_closed")
DIAGNOSTIC_FIXED_BASELINE_IDS = ("hold",)
SUPPORTED_FIXED_BASELINE_IDS = FORMAL_FIXED_BASELINE_IDS + DIAGNOSTIC_FIXED_BASELINE_IDS
# Backward-compatible name used by older callers: now means the Formal fixed set.
FIXED_BASELINE_IDS = FORMAL_FIXED_BASELINE_IDS

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
    """Create the scientific No-supervisory-RTC INP.

    This removes user-defined supervisory ``[CONTROLS]`` only. It intentionally preserves
    pump curves, initial pump status, intrinsic [PUMPS] Startup/Shutoff depths, storage
    geometry, regulator physics and all forcing. No Python control writes are made. It is
    therefore neither All-open nor All-closed and remains an operationally meaningful
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


def constant_setting_controller(value: float, source: str) -> Callable[[CausalObservation], ControllerAction]:
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


def fixed_baseline_controller(strategy: str):
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
    raise ValueError(f"not a supported fixed baseline strategy: {strategy}")
