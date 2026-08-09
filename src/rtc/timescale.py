from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HydraulicResponse:
    response_delay_seconds: float
    settling_seconds: float
    peak_absolute_effect: float
    responsive: bool


def identify_response_time(
    elapsed_seconds: np.ndarray,
    center_series: np.ndarray,
    perturbed_series: np.ndarray,
    *,
    absolute_threshold: float = 1e-6,
    relative_threshold: float = 0.10,
    settling_fraction: float = 0.10,
) -> HydraulicResponse:
    """Estimate delay and settling time from same-checkpoint D2 trajectories."""

    t = np.asarray(elapsed_seconds, dtype=float).reshape(-1)
    center = np.asarray(center_series, dtype=float).reshape(-1)
    perturbed = np.asarray(perturbed_series, dtype=float).reshape(-1)
    if not (t.size == center.size == perturbed.size) or t.size < 2:
        raise ValueError("response series must have equal length >= 2")
    if np.any(np.diff(t) <= 0):
        raise ValueError("elapsed_seconds must be strictly increasing")
    effect = perturbed - center
    peak = float(np.max(np.abs(effect)))
    threshold = max(float(absolute_threshold), float(relative_threshold) * peak)
    if peak <= threshold or not np.isfinite(peak):
        return HydraulicResponse(float("nan"), float("nan"), peak, False)

    active = np.where(np.abs(effect) >= threshold)[0]
    first = int(active[0])
    response_delay = float(t[first] - t[0])

    final_effect = float(np.median(effect[max(first, len(effect) - 3) :]))
    tolerance = max(absolute_threshold, settling_fraction * max(abs(final_effect), peak))
    settling = float(t[-1] - t[0])
    for i in range(first, len(effect)):
        if np.all(np.abs(effect[i:] - final_effect) <= tolerance):
            settling = float(t[i] - t[0])
            break
    return HydraulicResponse(response_delay, settling, peak, True)


def recommend_time_scales(
    responses: list[HydraulicResponse],
    *,
    minimum_control_minutes: int = 5,
    horizon_multiplier: float = 2.5,
) -> dict[str, int]:
    """Produce conservative engineering starting values from responsive D2 probes."""

    valid = [r for r in responses if r.responsive and np.isfinite(r.settling_seconds)]
    if not valid:
        raise ValueError("no responsive actuator probes available")
    delay = np.median([r.response_delay_seconds for r in valid]) / 60.0
    settle90 = np.quantile([r.settling_seconds for r in valid], 0.90) / 60.0
    model_step = max(1, int(round(max(1.0, delay / 2.0))))
    control_update = max(minimum_control_minutes, int(round(max(delay, model_step))))
    horizon = max(control_update, int(np.ceil(settle90 * horizon_multiplier / control_update)) * control_update)
    return {
        "model_step_minutes": model_step,
        "control_update_minutes": control_update,
        "prediction_horizon_minutes": horizon,
    }
