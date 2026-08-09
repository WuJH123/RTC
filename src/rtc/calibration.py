from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


def _finite_sample_upper_quantile(values: np.ndarray, coverage: float) -> np.ndarray:
    """Distribution-free split-conformal upper quantile along sample axis 0."""

    x = np.asarray(values, dtype=float)
    if x.ndim < 1 or x.shape[0] < 2:
        raise ValueError("calibration requires at least two samples")
    if not 0.5 < coverage < 1.0:
        raise ValueError("coverage must lie in (0.5, 1)")
    # q_level = ceil((n+1)*coverage)/n, clipped to an available empirical quantile.
    n = x.shape[0]
    level = min(1.0, np.ceil((n + 1) * coverage) / n)
    return np.quantile(x, level, axis=0, method="higher")


@dataclass(frozen=True)
class SafetyCalibration:
    """Frozen model-error envelope; rainfall-scenario risk is handled separately."""

    priority_nodes: tuple[str, ...]
    coverage: float
    flood_error_ucb_m3: tuple[float, ...]
    depth_error_ucb_m: tuple[float, ...]
    calibration_sample_count: int
    calibration_rainfall_groups: tuple[str, ...]
    source_sha256: str
    contract: str = "SITEWISE_SPLIT_CONFORMAL_SAFETY_V1"

    def validate(self) -> None:
        n = len(self.priority_nodes)
        if n == 0:
            raise ValueError("priority nodes are required")
        if len(self.flood_error_ucb_m3) != n or len(self.depth_error_ucb_m) != n:
            raise ValueError("calibration vectors must match priority-node count")
        if any(not np.isfinite(v) for v in self.flood_error_ucb_m3 + self.depth_error_ucb_m):
            raise ValueError("calibration values must be finite")
        if self.calibration_sample_count < 2:
            raise ValueError("insufficient calibration samples")
        if not self.calibration_rainfall_groups:
            raise ValueError("calibration rainfall groups are required")

    def to_json(self, path: str | Path) -> None:
        self.validate()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "SafetyCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in (
            "priority_nodes",
            "flood_error_ucb_m3",
            "depth_error_ucb_m",
            "calibration_rainfall_groups",
        ):
            payload[key] = tuple(payload[key])
        obj = cls(**payload)
        obj.validate()
        return obj


def fit_sitewise_safety_calibration(
    *,
    priority_nodes: tuple[str, ...],
    predicted_flood_deterioration_m3: np.ndarray,
    true_flood_deterioration_m3: np.ndarray,
    predicted_depth_deterioration_m: np.ndarray,
    true_depth_deterioration_m: np.ndarray,
    rainfall_groups: np.ndarray,
    coverage: float = 0.95,
) -> SafetyCalibration:
    """Fit one-sided error envelopes from a calibration-only rainfall partition.

    Residuals are ``truth - prediction``. The resulting UCB is added to the model's
    predicted deterioration at runtime. This is intentionally distinct from taking a
    quantile across rainfall forecast scenarios.
    """

    pred_v = np.asarray(predicted_flood_deterioration_m3, dtype=float)
    true_v = np.asarray(true_flood_deterioration_m3, dtype=float)
    pred_h = np.asarray(predicted_depth_deterioration_m, dtype=float)
    true_h = np.asarray(true_depth_deterioration_m, dtype=float)
    if pred_v.shape != true_v.shape or pred_h.shape != true_h.shape:
        raise ValueError("prediction/truth shape mismatch")
    if pred_v.ndim != 2 or pred_h.ndim != 2:
        raise ValueError("expected [sample, priority_site] arrays")
    if pred_v.shape != pred_h.shape or pred_v.shape[1] != len(priority_nodes):
        raise ValueError("site dimension does not match priority nodes")
    groups = np.asarray(rainfall_groups).astype(str).reshape(-1)
    if groups.size != pred_v.shape[0]:
        raise ValueError("rainfall_groups length mismatch")

    flood_q = _finite_sample_upper_quantile(true_v - pred_v, coverage)
    depth_q = _finite_sample_upper_quantile(true_h - pred_h, coverage)
    digest = hashlib.sha256()
    for arr in (pred_v, true_v, pred_h, true_h):
        digest.update(np.ascontiguousarray(arr).tobytes())
    digest.update("\n".join(groups.tolist()).encode("utf-8"))

    result = SafetyCalibration(
        priority_nodes=tuple(priority_nodes),
        coverage=float(coverage),
        flood_error_ucb_m3=tuple(float(x) for x in flood_q),
        depth_error_ucb_m=tuple(float(x) for x in depth_q),
        calibration_sample_count=int(pred_v.shape[0]),
        calibration_rainfall_groups=tuple(sorted(set(groups.tolist()))),
        source_sha256=digest.hexdigest(),
    )
    result.validate()
    return result


def calibrated_sitewise_ucb(
    predicted_deterioration: np.ndarray,
    error_ucb: np.ndarray,
    *,
    scenario_quantile: float,
) -> np.ndarray:
    """Combine forecast-scenario risk with independently calibrated model error."""

    pred = np.asarray(predicted_deterioration, dtype=float)
    err = np.asarray(error_ucb, dtype=float)
    if pred.ndim != 2:
        raise ValueError("predicted deterioration must be [scenario, site]")
    if err.shape != (pred.shape[1],):
        raise ValueError("error UCB must be one value per site")
    if not 0.5 < scenario_quantile < 1.0:
        raise ValueError("scenario_quantile must lie in (0.5, 1)")
    return np.quantile(pred, scenario_quantile, axis=0) + err
