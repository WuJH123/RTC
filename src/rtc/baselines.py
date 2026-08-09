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
    "auto_rbc": BaselineDefinition(
        "auto_rbc",
        "Automatically parameterized causal rule-based control from actuator-adjacent normalized node depths; no forecast or event tuning",
        True,
        False,
    ),
    "efd": BaselineDefinition(
        "efd",
        "Storage-aware Equal Filling Degree control using current normalized storage depths and writable storage outflows",
        True,
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

# Formal comparison matrix. Auto-RBC and EFD are genuine dynamic rule-based comparators;
# All-open/All-closed remain diagnostic extremes. Hold remains debug-only.
FORMAL_FIXED_BASELINE_IDS = (
    "no_control",
    "internal_rtc",
    "auto_rbc",
    "efd",
    "all_open",
    "all_closed",
)
DIAGNOSTIC_FIXED_BASELINE_IDS = ("hold",)
SUPPORTED_FIXED_BASELINE_IDS = FORMAL_FIXED_BASELINE_IDS + DIAGNOSTIC_FIXED_BASELINE_IDS
FIXED_BASELINE_IDS = FORMAL_FIXED_BASELINE_IDS

# Compatibility aliases. The commonly used term is Auto-RBC (rule-based control); accept
# Auto-RRC as a user-facing spelling but canonicalize all scientific evidence to auto_rbc.
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
