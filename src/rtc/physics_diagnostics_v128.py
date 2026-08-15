"""Physics-readiness diagnostics for current Project7 V128.

Do not turn an incomplete water-balance proxy into a training loss.  Current node state stores
volume, total inflow, total outflow and flooding, but not every SWMM node loss term and not
ordinary conduit dynamic flow.  This module quantifies a continuity proxy and explicitly gates
future conduit-flow supervision until authoritative fields exist in the frozen data contract.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

PHYSICS_DIAGNOSTICS_CONTRACT = "PROJECT7_V128_PHYSICS_READINESS_DIAGNOSTICS_V1"


def node_continuity_proxy(
    initial_state: np.ndarray,
    states: np.ndarray,
    *,
    dt_seconds: float = 300.0,
    volume_index: int = 3,
    total_inflow_index: int = 4,
    total_outflow_index: int = 5,
    flooding_index: int = 2,
) -> dict[str, float | int | bool | str]:
    """Compute a diagnostic-only node storage continuity proxy.

    Proxy equation: dV/dt ~= total_inflow - total_outflow - flooding.
    It is intentionally *not* an exact SWMM mass-balance loss because the current Step2 state
    does not expose all node loss terms (e.g. storage evaporation/exfiltration).  The result can
    flag gross inconsistencies but must not be optimized until the missing terms are recorded.
    """
    x0 = np.asarray(initial_state, dtype=np.float64)
    x = np.asarray(states, dtype=np.float64)
    if x.ndim != 3 or x0.shape != x.shape[1:]:
        raise ValueError("continuity proxy expects initial [N,S] and states [H,N,S]")
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be positive")
    channels = x.shape[-1]
    required = (volume_index, total_inflow_index, total_outflow_index, flooding_index)
    if min(required) < 0 or max(required) >= channels:
        raise ValueError("continuity proxy state-channel index outside state tensor")
    previous_volume = np.concatenate((x0[None, :, volume_index], x[:-1, :, volume_index]), axis=0)
    dv_dt = (x[:, :, volume_index] - previous_volume) / float(dt_seconds)
    rhs = x[:, :, total_inflow_index] - x[:, :, total_outflow_index] - x[:, :, flooding_index]
    residual = dv_dt - rhs
    scale = np.maximum(np.abs(dv_dt) + np.abs(rhs), 1.0e-6)
    return {
        "contract": PHYSICS_DIAGNOSTICS_CONTRACT,
        "training_loss_enabled": False,
        "exact_swmm_mass_balance": False,
        "reason_not_exact": "current Step2 state omits some SWMM node loss terms and ordinary conduit flow",
        "samples": int(residual.size),
        "residual_mae_m3s": float(np.mean(np.abs(residual))),
        "residual_rmse_m3s": float(np.sqrt(np.mean(np.square(residual)))),
        "relative_abs_residual_mean": float(np.mean(np.abs(residual) / scale)),
    }


def conduit_flow_supervision_readiness(arrays: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-open diagnostic: report whether authoritative ordinary-link flow labels exist."""
    keys = set(str(key) for key in arrays)
    candidates = (
        "target_link_flows",
        "target_link_flow_m3s",
        "target_conduit_flows",
        "target_conduit_flow_m3s",
    )
    present = [key for key in candidates if key in keys]
    return {
        "contract": PHYSICS_DIAGNOSTICS_CONTRACT,
        "ordinary_conduit_flow_supervision_available": bool(present),
        "matching_fields": present,
        "training_enabled": False,
        "required_action": (
            "If no matching field exists, do not fabricate conduit-flow supervision. Only extend "
            "the SWMM data contract after a development edge-physics ablation demonstrates value."
        ),
    }


__all__ = [
    "PHYSICS_DIAGNOSTICS_CONTRACT",
    "conduit_flow_supervision_readiness",
    "node_continuity_proxy",
]
